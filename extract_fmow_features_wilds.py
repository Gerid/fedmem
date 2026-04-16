"""Extract ResNet18 features from pre-staged WILDS-fMoW v1.1 images.

This is a drop-in replacement for the HuggingFace-based `extract_fmow_features.py`
that reads directly from the WILDS v1.1 directory layout already sitting under
``.test_fmow_cache_nonexist/fmow_v1.1/``:

    .test_fmow_cache_nonexist/fmow_v1.1/
    ├── rgb_metadata.csv         # 523,846 rows
    └── images/rgb_img_{i}.png   # 1:1 row ↔ file, already 224×224 RGB

The output follows the exact cache-file format that
``fedprotrack/real_data/fmow.py::_cache_file`` expects so that
``run_fmow_all_baselines.py`` / ``run_fmow_real_budget.py`` can consume the
features without any code changes:

    .feature_cache/fmow_c{cid}_nc{n_classes}_nf{n_features}_fseed{fseed}.npz

Sampling and transform semantics mirror ``_load_fmow_wilds_real`` in
``fedprotrack/real_data/fmow.py`` so that cache files produced here are
indistinguishable (up to GPU floating-point noise) from cache files produced
by the online WILDS path.

Usage
-----
    conda run -n base python extract_fmow_features_wilds.py

Configuration is via module-level constants — edit them if you need a
different (n_concepts, n_features, n_classes, feature_seed) tuple.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18

# ── Configuration ─────────────────────────────────────────────────────
# These MUST match the FMOWConfig / benchmark-script defaults so the
# resulting cache is auto-loaded by run_cifar100_neurips_benchmark.py
# (--dataset fmow) and run_fmow_all_baselines.py.
#
# We emit BOTH nf=64 (fmow.py FMOWConfig default) and nf=128 (neurips
# benchmark script's --n-features default, matching CIFAR-100 protocol)
# in a single forward pass so either entrypoint can load the cache.
N_CONCEPTS = 4
N_CLASSES = 10
N_FEATURES_LIST = [64, 128]
FEATURE_SEED = 3141
BATCH_SIZE = 256

# Global cap matches _load_fmow_wilds_real (max_per_concept * 10 = 50_000).
# This is a GLOBAL subsample size, not per-concept.
GLOBAL_SUBSAMPLE_CAP = 50_000

# Source (WILDS v1.1 layout, already staged locally).
FMOW_ROOT = Path("E:/fedprotrack/.test_fmow_cache_nonexist/fmow_v1.1")
METADATA_CSV = FMOW_ROOT / "rgb_metadata.csv"
IMAGES_DIR = FMOW_ROOT / "images"

# Destination (matches _cache_file format in fmow.py).
FEATURE_CACHE_DIR = Path("E:/fedprotrack/.feature_cache")

# fMoW year range (used for the equal-width concept binning helper below).
YEAR_MIN = 2002
YEAR_MAX = 2017


def _year_to_concept(years: np.ndarray, n_concepts: int) -> np.ndarray:
    """Equal-width year binning — mirrors fmow.py::_year_to_concept."""
    year_min = int(years.min())
    year_max = int(years.max())
    if year_max == year_min:
        return np.zeros(len(years), dtype=np.int64)
    bins = np.linspace(year_min, year_max + 1, n_concepts + 1)
    concept_ids = np.digitize(years, bins) - 1
    return np.clip(concept_ids, 0, n_concepts - 1).astype(np.int64)


def _cache_file(cid: int, n_features: int) -> Path:
    """Same path convention as fmow.py::_cache_file."""
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"fmow_c{cid}_nc{N_CLASSES}_nf{n_features}_fseed{FEATURE_SEED}"
    return FEATURE_CACHE_DIR / f"{tag}.npz"


def _load_metadata() -> pd.DataFrame:
    """Load and filter rgb_metadata.csv.

    Returns a DataFrame with one row per included sample, plus a
    ``row_idx`` column giving the original 0-indexed row number so the
    corresponding ``images/rgb_img_{row_idx}.png`` can be found.
    """
    print(f"Loading metadata: {METADATA_CSV}")
    t0 = time.time()
    df = pd.read_csv(METADATA_CSV)
    df["row_idx"] = np.arange(len(df), dtype=np.int64)
    print(f"  {len(df)} rows loaded ({time.time() - t0:.1f}s)")

    # Restrict to the train split + visible==True, mirroring
    # _load_fmow_wilds_real which calls dataset.get_subset('train') and
    # the standard WILDS fMoW convention of dropping cloud-covered images.
    df = df[df["split"] == "train"]
    df = df[df["visible"] == True]  # noqa: E712 (explicit truth check)
    print(f"  after split=train + visible=True: {len(df)} rows")

    # Parse year from ISO timestamp.
    df["year"] = df["timestamp"].str[:4].astype(int)
    return df.reset_index(drop=True)


def _select_top_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the top-N most frequent categories and remap labels."""
    counts = Counter(df["category"])
    top = [cat for cat, _ in counts.most_common(N_CLASSES)]
    # Match fmow.py ordering: sorted() then enumerate -> new label.
    label_map = {cat: new for new, cat in enumerate(sorted(top))}
    print(f"  top-{N_CLASSES} classes: {sorted(top)}")
    mask = df["category"].isin(label_map)
    kept = df[mask].copy()
    kept["label"] = kept["category"].map(label_map).astype(np.int64)
    print(f"  after top-{N_CLASSES} filter: {len(kept)} rows")
    return kept.reset_index(drop=True)


def _global_subsample(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """Global cap mirroring _load_fmow_wilds_real's max_per_concept*10 logic."""
    if len(df) <= GLOBAL_SUBSAMPLE_CAP:
        return df.reset_index(drop=True)
    keep = rng.choice(len(df), GLOBAL_SUBSAMPLE_CAP, replace=False)
    keep.sort()  # preserve chronological order for reproducibility
    subsampled = df.iloc[keep].reset_index(drop=True)
    print(f"  globally subsampled: {len(df)} -> {len(subsampled)}")
    return subsampled


def _extract_resnet_features(
    df: pd.DataFrame,
    device: torch.device,
) -> np.ndarray:
    """Batched ResNet18 forward over the selected rows."""
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    n = len(df)
    print(f"\nExtracting ResNet18 features over {n} images on {device}")
    print(f"  batch_size={BATCH_SIZE}")
    t0 = time.time()

    feature_parts: list[np.ndarray] = []
    batch_tensors: list[torch.Tensor] = []
    n_done = 0
    failed = 0

    def _flush() -> None:
        nonlocal n_done
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device, non_blocking=True)
        with torch.inference_mode():
            feats = backbone(batch).cpu().numpy().astype(np.float32)
        feature_parts.append(feats)
        n_done += len(batch_tensors)
        batch_tensors.clear()
        if (n_done // BATCH_SIZE) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (n - n_done) / max(rate, 1e-6)
            print(f"  {n_done}/{n}  rate={rate:.0f} img/s  eta={eta:.0f}s")

    for row_idx in df["row_idx"].values:
        img_path = IMAGES_DIR / f"rgb_img_{int(row_idx)}.png"
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor = preprocess(img)
            batch_tensors.append(tensor)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  WARN: failed to load {img_path.name}: {e}")
            # Pad with zeros so the feature array stays aligned with the df
            batch_tensors.append(torch.zeros(3, 224, 224))
        if len(batch_tensors) >= BATCH_SIZE:
            _flush()
    _flush()

    features = np.concatenate(feature_parts, axis=0)
    print(f"  extracted shape={features.shape} in {time.time() - t0:.0f}s"
          f"  ({failed} load failures)")
    return features


def main() -> None:
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {METADATA_CSV}")
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Missing images dir: {IMAGES_DIR}")

    # Early exit if cache already complete for ALL nf variants.
    if all(
        _cache_file(cid, nf).exists()
        for cid in range(N_CONCEPTS)
        for nf in N_FEATURES_LIST
    ):
        print("Feature cache already complete:")
        for nf in N_FEATURES_LIST:
            for cid in range(N_CONCEPTS):
                path = _cache_file(cid, nf)
                with np.load(path) as d:
                    print(f"  {path.name}: X={d['X'].shape}, y={d['y'].shape}")
        return

    rng = np.random.RandomState(FEATURE_SEED)

    # Step 1: metadata.
    df = _load_metadata()

    # Step 2: top-N classes.
    df = _select_top_classes(df)

    # Step 3: global subsample.
    df = _global_subsample(df, rng)

    # Step 4: year → concept binning.
    years = df["year"].values.astype(np.int64)
    concept_ids = _year_to_concept(years, N_CONCEPTS)
    df["concept_id"] = concept_ids
    print("\nPer-concept sample counts:")
    for cid in range(N_CONCEPTS):
        mask = concept_ids == cid
        yrs = years[mask]
        if mask.sum() > 0:
            print(f"  concept {cid}: {mask.sum()} samples "
                  f"(years {yrs.min()}-{yrs.max()})")
        else:
            print(f"  concept {cid}: 0 samples  (empty bin)")

    # Step 5: ResNet18 features.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_512 = _extract_resnet_features(df, device)

    labels = df["label"].values.astype(np.int64)

    # Step 6+7: fit a PCA per target dim (shared 512-d features) and save.
    for n_features in N_FEATURES_LIST:
        print(f"\n--- Building nf={n_features} cache ---")
        print(f"Fitting PCA (n_components={n_features}, random_state={FEATURE_SEED})")
        t0 = time.time()
        pca = PCA(n_components=n_features, random_state=FEATURE_SEED)
        features_pca = pca.fit_transform(features_512).astype(np.float32)
        print(f"  explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"  transformed shape={features_pca.shape}  ({time.time() - t0:.0f}s)")

        for cid in range(N_CONCEPTS):
            mask = concept_ids == cid
            if mask.sum() == 0:
                print(f"  concept {cid}: SKIP (empty)")
                continue
            X = features_pca[mask]
            y = labels[mask]
            path = _cache_file(cid, n_features)
            np.savez_compressed(path, X=X, y=y)
            print(f"  {path.name}: X={X.shape}, y={y.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
