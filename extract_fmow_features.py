"""Extract ResNet18 features from real FMOW val images, grouped by year-concept.

Strategy: extract tar.gz once to disk, then batch-process all images with GPU.
Output: .fmow_real_features/concept_{cid}_nc{n_classes}_nf{n_features}.npz
"""

from __future__ import annotations

import json
import os
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from torchvision.models import ResNet18_Weights, resnet18

# --- Configuration ---
N_CONCEPTS = 4
N_CLASSES = 10
N_FEATURES = 64
BATCH_SIZE = 256
FEATURE_CACHE_DIR = Path(".fmow_real_features")
HF_CACHE_BASE = Path(
    ".fmow_hf_cache/datasets--jbourcier--fmow-rgb-baseline/"
    "snapshots/448b87b29c1e56b704a2b6ab7093560db63e223e"
)
EXTRACT_DIR = Path(".fmow_val_extracted")

YEAR_MIN = 2002
YEAR_MAX = 2017


def year_to_concept(year: int, n_concepts: int) -> int:
    """Map a year to a concept ID via equal-width binning."""
    if year <= YEAR_MIN:
        return 0
    if year >= YEAR_MAX:
        return n_concepts - 1
    span = YEAR_MAX - YEAR_MIN + 1
    bin_width = span / n_concepts
    return min(int((year - YEAR_MIN) / bin_width), n_concepts - 1)


def extract_tar_if_needed() -> None:
    """Extract val-images.tar.gz and val-metadata.tar.gz to disk once."""
    if (EXTRACT_DIR / "_extraction_done").exists():
        print("Tar already extracted, skipping.")
        return

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ["val-metadata.tar.gz", "val-images.tar.gz"]:
        tar_path = HF_CACHE_BASE / name
        print(f"Extracting {name} ({tar_path.stat().st_size / 1e9:.2f} GB)...")
        t0 = time.time()
        with tarfile.open(str(tar_path), "r:gz") as tar:
            tar.extractall(str(EXTRACT_DIR), filter="data")
        print(f"  Done in {time.time() - t0:.0f}s")

    # Mark extraction complete
    (EXTRACT_DIR / "_extraction_done").touch()


def parse_metadata() -> list[dict]:
    """Parse all val metadata JSON files from extracted directory."""
    print("Parsing metadata...")
    t0 = time.time()
    records: list[dict] = []

    val_dir = EXTRACT_DIR / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"Expected {val_dir} after extraction")

    for category_dir in sorted(val_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for json_path in category_dir.rglob("*.json"):
            with open(json_path) as f:
                data = json.load(f)

            ts = data.get("timestamp", "")
            year = int(ts[:4]) if ts and len(ts) >= 4 else 0
            img_path = json_path.with_suffix(".jpg")

            if img_path.exists():
                records.append({
                    "img_path": str(img_path),
                    "category": category,
                    "year": year,
                })

    print(f"  Parsed {len(records)} records in {time.time() - t0:.1f}s")
    return records


def main() -> None:
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache
    cache_tag = f"nc{N_CLASSES}_nf{N_FEATURES}_nconcepts{N_CONCEPTS}"
    all_cached = all(
        (FEATURE_CACHE_DIR / f"concept_{cid}_{cache_tag}.npz").exists()
        for cid in range(N_CONCEPTS)
    )
    if all_cached:
        print("Features already cached!")
        for cid in range(N_CONCEPTS):
            path = FEATURE_CACHE_DIR / f"concept_{cid}_{cache_tag}.npz"
            with np.load(path) as data:
                print(f"  Concept {cid}: X={data['X'].shape}, y={data['y'].shape}")
        return

    # --- Step 1: Extract tar to disk ---
    print("=" * 60)
    print("Step 1: Extract tar archives to disk")
    print("=" * 60)
    extract_tar_if_needed()

    # --- Step 2: Parse metadata ---
    print("\n" + "=" * 60)
    print("Step 2: Parse metadata")
    print("=" * 60)
    records = parse_metadata()

    # --- Step 3: Filter top-N classes ---
    print(f"\nStep 3: Filter to top-{N_CLASSES} classes")
    cat_counts = Counter(r["category"] for r in records)
    top_cats = [cat for cat, _ in cat_counts.most_common(N_CLASSES)]
    cat_to_label = {cat: i for i, cat in enumerate(sorted(top_cats))}
    print(f"  Classes: {list(cat_to_label.keys())}")

    filtered = [r for r in records if r["category"] in cat_to_label]
    for r in filtered:
        r["concept_id"] = year_to_concept(r["year"], N_CONCEPTS)
        r["label"] = cat_to_label[r["category"]]

    print(f"  Total: {len(filtered)} samples")

    concept_counts = Counter(r["concept_id"] for r in filtered)
    for cid in sorted(concept_counts):
        years_in = [r["year"] for r in filtered if r["concept_id"] == cid]
        print(f"  Concept {cid}: {concept_counts[cid]} samples "
              f"(years {min(years_in)}-{max(years_in)})")

    # --- Step 4: Extract features with GPU ---
    print(f"\nStep 4: Extract ResNet18 features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    # Process ALL images at once, then split by concept
    print(f"\n  Loading and preprocessing {len(filtered)} images...")
    t0 = time.time()

    all_tensors: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_concepts: list[int] = []
    loaded = 0
    failed = 0

    for i, r in enumerate(filtered):
        try:
            img = Image.open(r["img_path"]).convert("RGB")
            tensor = preprocess(img)
            all_tensors.append(tensor)
            all_labels.append(r["label"])
            all_concepts.append(r["concept_id"])
            loaded += 1
        except Exception:
            failed += 1

        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{len(filtered)} loaded ({time.time()-t0:.0f}s)")

    print(f"  Loaded {loaded} images, {failed} failed ({time.time()-t0:.0f}s)")

    # Batch feature extraction
    print(f"\n  Extracting features (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    stacked = torch.stack(all_tensors)
    all_features: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(stacked), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(stacked))
            batch = stacked[start:end].to(device, non_blocking=True)
            feats = backbone(batch).cpu().numpy().astype(np.float32)
            all_features.append(feats)
            if (start // BATCH_SIZE + 1) % 20 == 0:
                print(f"    Batch {start//BATCH_SIZE+1}/"
                      f"{(len(stacked)+BATCH_SIZE-1)//BATCH_SIZE} "
                      f"({time.time()-t0:.0f}s)")

    features = np.concatenate(all_features, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    concepts = np.array(all_concepts, dtype=np.int64)
    print(f"  Features: {features.shape} ({time.time()-t0:.0f}s)")

    # --- Step 5: PCA ---
    print(f"\nStep 5: PCA to {N_FEATURES} dims")
    pca = PCA(n_components=N_FEATURES, random_state=42)
    features_pca = pca.fit_transform(features).astype(np.float32)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # --- Step 6: Save per-concept pools ---
    print(f"\nStep 6: Save per-concept pools")
    for cid in range(N_CONCEPTS):
        mask = concepts == cid
        X = features_pca[mask]
        y = labels[mask]
        path = FEATURE_CACHE_DIR / f"concept_{cid}_{cache_tag}.npz"
        np.savez_compressed(path, X=X, y=y)
        print(f"  Concept {cid}: X={X.shape}, y={y.shape} -> {path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
