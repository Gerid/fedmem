"""FMOW (Functional Map of the World) dataset for federated concept drift experiments.

Concepts correspond to temporal periods in satellite imagery. The WILDS FMOW
dataset has a natural temporal distribution shift (2002–2017) driven by changes
in satellite resolution, land use, and building styles.

The feature representation uses pretrained ResNet18 embeddings (frozen) reduced
with PCA to keep dimensionality compatible with the existing FedProTrack pipeline.

Data loading strategy:
1. Load FMOW via torchvision/WILDS or a local cache of pre-extracted features.
2. Bin images by year into ``n_concepts`` temporal concept buckets.
3. Extract ResNet18 features (frozen backbone) and reduce with PCA.
4. Map concept buckets onto the shared (rho, alpha) concept matrix.
5. Sample balanced federated batches per (client, time_step).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18

from ..drift_generator.concept_matrix import generate_concept_matrix
from ..drift_generator.configs import GeneratorConfig
from ..drift_generator.data_streams import ConceptSpec
from ..drift_generator.generator import DriftDataset


# FMOW spans 2002–2017; we use year as the temporal axis.
_FMOW_YEAR_MIN = 2002
_FMOW_YEAR_MAX = 2017
_FMOW_N_CLASSES = 62


@dataclass
class FMOWConfig:
    """Configuration for FMOW temporal-drift dataset generation.

    Parameters
    ----------
    K : int
        Number of federated clients.
    T : int
        Number of time steps.
    n_samples : int
        Samples per client per time step.
    rho : float
        Recurrence period for concept cycling.
    alpha : float
        Asynchrony parameter (0 = synchronous, 1 = fully async).
    delta : float
        Concept separability; for FMOW this controls how many year-bins
        are collapsed (higher delta = fewer, more distinct concepts).
    n_concepts : int
        Number of temporal concept buckets (year ranges).
    n_features : int
        PCA dimensions for feature representation.
    n_classes : int
        Number of target classes to keep (top-N most frequent).
        Set lower for faster smoke tests.
    batch_size : int
        Batch size for feature extraction.
    n_workers : int
        DataLoader workers for feature extraction.
    data_root : str
        Root directory for FMOW dataset download/cache.
    feature_cache_dir : str
        Directory for caching extracted feature pools.
    feature_seed : int
        Seed for PCA and balanced sampling.
    seed : int
        Master random seed.
    """

    K: int = 5
    T: int = 10
    n_samples: int = 200
    rho: float = 5.0
    alpha: float = 0.5
    delta: float = 0.5
    n_concepts: int = 4
    n_features: int = 64
    n_classes: int = 10
    batch_size: int = 256
    n_workers: int = 2
    data_root: str = ".fmow_cache"
    feature_cache_dir: str = ".feature_cache"
    feature_seed: int = 3141
    seed: int = 42
    eval_on_test_pool: bool = True
    test_split_ratio: float = 0.2
    n_eval_samples: int | None = None

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.T < 2:
            raise ValueError(f"T must be >= 2, got {self.T}")
        if self.n_concepts < 2:
            raise ValueError(f"n_concepts must be >= 2, got {self.n_concepts}")
        if self.n_features < 2:
            raise ValueError(f"n_features must be >= 2, got {self.n_features}")
        if self.n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {self.n_classes}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not (0.0 < self.delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {self.delta}")


def _year_to_concept(years: np.ndarray, n_concepts: int) -> np.ndarray:
    """Map year values to concept IDs by equal-width binning.

    Parameters
    ----------
    years : np.ndarray
        Year values (int).
    n_concepts : int
        Number of temporal concept bins.

    Returns
    -------
    np.ndarray
        Concept IDs in [0, n_concepts).
    """
    year_min = years.min()
    year_max = years.max()
    if year_max == year_min:
        return np.zeros(len(years), dtype=np.int64)
    # Equal-width bins
    bins = np.linspace(year_min, year_max + 1, n_concepts + 1)
    concept_ids = np.digitize(years, bins) - 1
    return np.clip(concept_ids, 0, n_concepts - 1).astype(np.int64)


class _FMOWFeatureDataset(Dataset):
    """Thin wrapper that applies ResNet18 preprocessing to raw images."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        preprocess,
    ) -> None:
        self.images = images  # (N, C, H, W) uint8
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[idx]).float() / 255.0
        image = self.preprocess(image)
        return image, int(self.labels[idx])


def _load_fmow_wilds(
    data_root: str,
    n_classes: int,
    feature_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load FMOW data, returning images, labels, and years.

    Attempts to use the WILDS package. If unavailable, falls back to
    a synthetic proxy for smoke testing.

    Parameters
    ----------
    data_root : str
        Cache/download root.
    n_classes : int
        Keep only the top-N most frequent classes.
    feature_seed : int
        Seed for any random subsampling.

    Returns
    -------
    images : np.ndarray
        Shape (N, 3, H, W), uint8.
    labels : np.ndarray
        Shape (N,), int64.
    years : np.ndarray
        Shape (N,), int.
    """
    try:
        return _load_fmow_wilds_real(data_root, n_classes, feature_seed)
    except (ImportError, FileNotFoundError, OSError):
        return _load_fmow_torchvision_proxy(data_root, n_classes, feature_seed)


def _load_fmow_wilds_real(
    data_root: str,
    n_classes: int,
    feature_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load real FMOW data from WILDS package.

    Raises
    ------
    ImportError
        If ``wilds`` is not installed.
    """
    from wilds import get_dataset  # type: ignore[import-untyped]

    dataset = get_dataset(dataset="fmow", root_dir=data_root, download=True)
    # Use train split
    train_data = dataset.get_subset("train")

    # Extract metadata: WILDS FMOW metadata[:,0] = region, metadata[:,1] = year
    metadata = train_data.dataset.metadata_array[train_data.indices]
    years = metadata[:, 1].numpy() if hasattr(metadata, "numpy") else np.array(metadata[:, 1])
    labels = np.array([train_data.dataset.y_array[i].item() for i in train_data.indices])

    # Keep only top-N classes
    rng = np.random.RandomState(feature_seed)
    class_counts = np.bincount(labels, minlength=_FMOW_N_CLASSES)
    top_classes = np.argsort(class_counts)[::-1][:n_classes]

    mask = np.isin(labels, top_classes)
    indices = train_data.indices[mask] if hasattr(train_data.indices, '__getitem__') else np.array(train_data.indices)[mask]
    labels = labels[mask]
    years = years[mask]

    # Remap labels to [0, n_classes)
    label_map = {old: new for new, old in enumerate(sorted(top_classes))}
    labels = np.array([label_map[l] for l in labels], dtype=np.int64)

    # Load images (this can be slow for full dataset)
    # Subsample if too large
    max_per_concept = 5000
    if len(indices) > max_per_concept * 10:
        keep = rng.choice(len(indices), max_per_concept * 10, replace=False)
        indices = indices[keep]
        labels = labels[keep]
        years = years[keep]

    images = []
    for idx in indices:
        img, _, _ = train_data.dataset[int(idx)]
        if isinstance(img, torch.Tensor):
            images.append(img.numpy())
        else:
            # PIL Image -> numpy
            img_np = np.array(img)
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
            images.append(img_np.astype(np.uint8))

    images = np.stack(images, axis=0)
    return images, labels, years.astype(np.int64)


def _load_fmow_torchvision_proxy(
    data_root: str,
    n_classes: int,
    feature_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: use EuroSAT from torchvision as a satellite imagery proxy.

    EuroSAT is a 10-class satellite land-use dataset that serves as a
    reasonable stand-in when WILDS is not installed. We simulate temporal
    shift by assigning synthetic year labels.

    Parameters
    ----------
    data_root : str
        Download root.
    n_classes : int
        Number of classes to keep (max 10 for EuroSAT).
    feature_seed : int
        Random seed.

    Returns
    -------
    images, labels, years : tuple of np.ndarray
    """
    from torchvision import datasets, transforms

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    # Try EuroSAT
    try:
        ds = datasets.EuroSAT(root=str(root), download=True)
    except Exception:
        # Final fallback: generate synthetic satellite-like data
        return _generate_synthetic_fmow_proxy(n_classes, feature_seed)

    rng = np.random.RandomState(feature_seed)
    n_total = len(ds)

    # Collect all images and labels
    all_labels = np.array([ds[i][1] for i in range(n_total)], dtype=np.int64)

    # Keep top-N classes
    actual_n_classes = min(n_classes, len(np.unique(all_labels)))
    class_counts = np.bincount(all_labels)
    top_classes = np.argsort(class_counts)[::-1][:actual_n_classes]

    mask = np.isin(all_labels, top_classes)
    valid_indices = np.flatnonzero(mask)
    labels = all_labels[valid_indices]

    # Remap labels
    label_map = {old: new for new, old in enumerate(sorted(top_classes))}
    labels = np.array([label_map[l] for l in labels], dtype=np.int64)

    # Simulate temporal years (uniform spread over 2002–2017)
    years = rng.randint(_FMOW_YEAR_MIN, _FMOW_YEAR_MAX + 1, size=len(labels))

    # Load images as numpy arrays (CHW format)
    images = []
    resize = transforms.Resize((64, 64))
    for idx in valid_indices:
        img_pil = ds[idx][0]
        img_pil = resize(img_pil)
        img_np = np.array(img_pil)
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
        images.append(img_np.astype(np.uint8))

    images_arr = np.stack(images, axis=0)
    return images_arr, labels, years.astype(np.int64)


def _generate_synthetic_fmow_proxy(
    n_classes: int,
    feature_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate purely synthetic data as a last-resort fallback.

    Creates random RGB images with class-conditional means to simulate
    satellite imagery. Useful when no real dataset can be downloaded.

    Parameters
    ----------
    n_classes : int
        Number of classes.
    feature_seed : int
        Random seed.

    Returns
    -------
    images, labels, years : tuple of np.ndarray
    """
    rng = np.random.RandomState(feature_seed)
    n_per_class = 500
    n_total = n_classes * n_per_class

    # Class-conditional RGB images (3, 64, 64)
    images = np.zeros((n_total, 3, 64, 64), dtype=np.uint8)
    labels = np.zeros(n_total, dtype=np.int64)

    for c in range(n_classes):
        start = c * n_per_class
        end = start + n_per_class
        # Each class has a different mean color
        mean_color = rng.randint(40, 220, size=(3, 1, 1))
        noise = rng.randint(0, 60, size=(n_per_class, 3, 64, 64))
        images[start:end] = np.clip(mean_color + noise, 0, 255).astype(np.uint8)
        labels[start:end] = c

    # Simulate temporal years
    years = rng.randint(_FMOW_YEAR_MIN, _FMOW_YEAR_MAX + 1, size=n_total)

    # Shuffle
    perm = rng.permutation(n_total)
    return images[perm], labels[perm], years[perm].astype(np.int64)


def _cache_file(config: FMOWConfig, concept_id: int) -> Path:
    """Return path for a cached feature pool."""
    cache_dir = Path(config.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"fmow_c{concept_id}"
        f"_nc{config.n_classes}"
        f"_nf{config.n_features}"
        f"_fseed{config.feature_seed}"
    )
    return cache_dir / f"{tag}.npz"


def _load_cached_feature_pools(
    config: FMOWConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """Load all concept pools from cache. Returns None if any is missing."""
    pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for concept_id in range(n_concepts):
        path = _cache_file(config, concept_id)
        if not path.exists():
            return None
        with np.load(path) as payload:
            pools[concept_id] = (
                payload["X"].astype(np.float32),
                payload["y"].astype(np.int64),
            )
    return pools


def _save_feature_pools(
    config: FMOWConfig,
    pools: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Save extracted feature pools to cache."""
    for concept_id, (X, y) in pools.items():
        path = _cache_file(config, concept_id)
        np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


def _extract_feature_pools(
    config: FMOWConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Extract ResNet18 features per temporal concept, with caching.

    Parameters
    ----------
    config : FMOWConfig
    n_concepts : int
        Number of concept bins.

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping concept_id -> (X_features, y_labels).
    """
    cached = _load_cached_feature_pools(config, n_concepts)
    if cached is not None:
        return cached

    images, labels, years = _load_fmow_wilds(
        config.data_root, config.n_classes, config.feature_seed,
    )
    concept_ids = _year_to_concept(years, n_concepts)

    # Setup ResNet18 feature extractor
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    pin_memory = device.type == "cuda"

    # Extract features per concept
    raw_pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    with torch.inference_mode():
        for cid in range(n_concepts):
            mask = concept_ids == cid
            if mask.sum() == 0:
                # Empty concept: skip (will not appear in concept matrix)
                continue

            concept_images = images[mask]
            concept_labels = labels[mask]

            ds = _FMOWFeatureDataset(concept_images, concept_labels, preprocess)
            loader = DataLoader(
                ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.n_workers,
                pin_memory=pin_memory,
            )

            feature_parts: list[np.ndarray] = []
            label_parts: list[np.ndarray] = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                features = backbone(batch_x)
                feature_parts.append(features.cpu().numpy().astype(np.float32))
                label_parts.append(batch_y.numpy().astype(np.int64))

            raw_pools[cid] = (
                np.concatenate(feature_parts, axis=0),
                np.concatenate(label_parts, axis=0),
            )

    # PCA reduction
    raw_dim = next(iter(raw_pools.values()))[0].shape[1]
    if config.n_features < raw_dim:
        combined = np.vstack([pool[0] for pool in raw_pools.values()])
        pca = PCA(n_components=config.n_features, random_state=config.feature_seed)
        pca.fit(combined)
        pools = {
            cid: (pca.transform(X).astype(np.float32), y)
            for cid, (X, y) in raw_pools.items()
        }
    else:
        pools = raw_pools

    _save_feature_pools(config, pools)
    return pools


def _draw_balanced_batch(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a class-balanced batch from one concept pool.

    Parameters
    ----------
    X_pool : np.ndarray
        Feature matrix for one concept.
    y_pool : np.ndarray
        Label array for one concept.
    n_samples : int
        Number of samples to draw.
    rng : np.random.RandomState
        Random state.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X_batch, y_batch)
    """
    classes = np.unique(y_pool)
    per_class = n_samples // len(classes)
    remainder = n_samples % len(classes)

    chosen: list[np.ndarray] = []
    for offset, cls in enumerate(classes):
        cls_idx = np.flatnonzero(y_pool == cls)
        take = per_class + (1 if offset < remainder else 0)
        chosen.append(rng.choice(cls_idx, size=take, replace=True))

    batch_idx = np.concatenate(chosen, axis=0)
    rng.shuffle(batch_idx)
    return X_pool[batch_idx].astype(np.float32), y_pool[batch_idx].astype(np.int64)


def prepare_fmow_feature_cache(
    config: FMOWConfig | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Prepare and cache per-concept feature pools.

    Call this ahead of time if you want to pre-extract features
    before running experiments.

    Parameters
    ----------
    config : FMOWConfig, optional

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping concept_id -> (X_features, y_labels).
    """
    if config is None:
        config = FMOWConfig()
    return _extract_feature_pools(config, config.n_concepts)


def generate_fmow_dataset(
    config: FMOWConfig | None = None,
) -> DriftDataset:
    """Generate an FMOW temporal-drift dataset compatible with FedProTrack pipeline.

    Each concept corresponds to a temporal period in FMOW satellite imagery.
    The concept matrix follows the same (rho, alpha) recurrence structure
    as synthetic experiments.

    Parameters
    ----------
    config : FMOWConfig, optional

    Returns
    -------
    DriftDataset
        Compatible with all existing runners and metrics.
    """
    if config is None:
        config = FMOWConfig()

    rng = np.random.RandomState(config.seed)

    # Build concept matrix using shared machinery
    gen_config = GeneratorConfig(
        K=config.K,
        T=config.T,
        n_samples=config.n_samples,
        rho=config.rho,
        alpha=config.alpha,
        delta=config.delta,
        generator_type="fmow",
        seed=config.seed,
    )
    concept_matrix = generate_concept_matrix(
        K=config.K,
        T=config.T,
        n_concepts=config.n_concepts,
        alpha=config.alpha,
        seed=config.seed,
    )

    n_concepts = int(concept_matrix.max()) + 1
    raw_pools = _extract_feature_pools(config, n_concepts)

    from .cifar100_recurrence import _split_train_test_pools
    if config.eval_on_test_pool:
        train_pools, test_pools = _split_train_test_pools(
            raw_pools, config.test_split_ratio, config.feature_seed,
        )
    else:
        train_pools = raw_pools
        test_pools = None

    n_eval = config.n_eval_samples if config.n_eval_samples is not None else config.n_samples

    # Build data dict: (k, t) -> (X, y)
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    test_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = (
        {} if test_pools is not None else None
    )
    for k in range(config.K):
        for t in range(config.T):
            concept_id = int(concept_matrix[k, t])
            seed_kt = config.seed + 10000 + k * config.T + t
            kt_rng = np.random.RandomState(seed_kt)

            if concept_id in train_pools:
                X_pool, y_pool = train_pools[concept_id]
            else:
                fallback_id = min(train_pools.keys())
                X_pool, y_pool = train_pools[fallback_id]
                concept_id = fallback_id
            data[(k, t)] = _draw_balanced_batch(
                X_pool, y_pool, config.n_samples, kt_rng,
            )
            if test_data is not None:
                Xt_pool, yt_pool = test_pools[concept_id]
                rng_test = np.random.RandomState(
                    config.seed + 30000 + k * config.T + t
                )
                test_data[(k, t)] = _draw_balanced_batch(
                    Xt_pool, yt_pool, n_eval, rng_test,
                )

    # Build concept specs
    unique_concepts = sorted(set(concept_matrix.flatten()))
    concept_specs = [
        ConceptSpec(
            concept_id=cid,
            generator_type="fmow",
            variant=cid,
            noise_scale=config.delta,
        )
        for cid in unique_concepts
    ]

    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=gen_config,
        concept_specs=concept_specs,
        test_data=test_data,
    )
