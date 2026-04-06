"""CIFAR-10 recurrence benchmark for federated concept-drift experiments.

Builds a recurring-concept dataset from CIFAR-10 by:
1. defining manual concept groupings over the 10 classes
   (CIFAR-10 has no coarse labels, so groups are defined explicitly),
2. applying deterministic concept-specific appearance shifts,
3. extracting pretrained ResNet18 embeddings,
4. reducing them with PCA, and
5. sampling balanced federated batches according to the shared concept matrix.

This keeps the real-data benchmark compatible with the existing
``DriftDataset`` / ``FedProTrackRunner`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as TF

from ..drift_generator.concept_matrix import generate_concept_matrix
from ..drift_generator.configs import GeneratorConfig
from ..drift_generator.data_streams import ConceptSpec
from ..drift_generator.generator import DriftDataset


_N_CLASSES = 10
_STYLE_COUNT = 5

# Default concept groupings: 5 concepts of 2 classes each.
# CIFAR-10 classes: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
#                   5=dog, 6=frog, 7=horse, 8=ship, 9=truck
# Groupings pair semantically related classes:
#   concept 0 (vehicles-air/sea): airplane, ship
#   concept 1 (vehicles-land):    automobile, truck
#   concept 2 (small animals):    bird, frog
#   concept 3 (pets):             cat, dog
#   concept 4 (large animals):    deer, horse
DEFAULT_CONCEPT_GROUPS: dict[int, list[int]] = {
    0: [0, 8],
    1: [1, 9],
    2: [2, 6],
    3: [3, 5],
    4: [4, 7],
}


@dataclass
class CIFAR10RecurrenceConfig:
    """Configuration for the CIFAR-10 recurrence benchmark.

    Parameters
    ----------
    K : int
        Number of federated clients.
    T : int
        Number of time steps.
    n_samples : int
        Samples per client per time step.
    rho : float
        Recurrence frequency; concept pool size = T / rho.
    alpha : float
        Asynchrony level in [0, 1].
    delta : float
        Separability / style strength in (0, 1].
    n_features : int
        PCA dimensions for feature representation.
    samples_per_class : int
        Number of source images sampled per class for feature extraction.
    batch_size : int
        Batch size for ResNet18 feature extraction.
    n_workers : int
        DataLoader worker count.
    concept_groups : dict[int, list[int]]
        Mapping from concept ID to list of CIFAR-10 class indices.
    data_root : str
        Directory for raw CIFAR-10 download cache.
    feature_cache_dir : str
        Directory for extracted feature cache.
    feature_seed : int
        Seed for balanced index selection and PCA.
    seed : int
        Master seed for concept matrix and batch sampling.
    """

    K: int = 6
    T: int = 12
    n_samples: int = 400
    rho: float = 3.0
    alpha: float = 0.75
    delta: float = 0.85
    n_features: int = 128
    samples_per_class: int = 500
    batch_size: int = 256
    n_workers: int = 2
    concept_groups: dict[int, list[int]] = field(
        default_factory=lambda: dict(DEFAULT_CONCEPT_GROUPS)
    )
    data_root: str = ".cifar10_cache"
    feature_cache_dir: str = ".feature_cache"
    feature_seed: int = 2718
    seed: int = 42

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.T < 2:
            raise ValueError(f"T must be >= 2, got {self.T}")
        if self.n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {self.n_samples}")
        if self.samples_per_class < 1:
            raise ValueError(
                f"samples_per_class must be >= 1, got {self.samples_per_class}"
            )
        if self.n_features < 2:
            raise ValueError(f"n_features must be >= 2, got {self.n_features}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not (0.0 < self.delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {self.delta}")
        if len(self.concept_groups) < 2:
            raise ValueError(
                f"concept_groups must have >= 2 entries, got {len(self.concept_groups)}"
            )
        # Validate that all referenced classes are in [0, 9]
        for cid, classes in self.concept_groups.items():
            if not classes:
                raise ValueError(f"concept_groups[{cid}] is empty")
            for cls in classes:
                if cls < 0 or cls >= _N_CLASSES:
                    raise ValueError(
                        f"concept_groups[{cid}] contains invalid class {cls}; "
                        f"must be in [0, {_N_CLASSES - 1}]"
                    )

    def to_generator_config(self) -> GeneratorConfig:
        """Convert to a ``GeneratorConfig`` for concept matrix generation.

        Returns
        -------
        GeneratorConfig
        """
        return GeneratorConfig(
            K=self.K,
            T=self.T,
            n_samples=self.n_samples,
            rho=self.rho,
            alpha=self.alpha,
            delta=self.delta,
            generator_type="cifar10_recurrence",
            seed=self.seed,
        )

    def _groups_hash(self) -> str:
        """Deterministic short hash of concept_groups for cache keying."""
        raw = str(sorted((k, sorted(v)) for k, v in self.concept_groups.items()))
        return hashlib.md5(raw.encode()).hexdigest()[:8]


class _ConceptImageDataset(Dataset):
    """Dataset view for one concept-specific CIFAR-10 appearance style.

    Parameters
    ----------
    images : np.ndarray
        Full image array of shape (N, 3, 32, 32), dtype uint8.
    labels : np.ndarray
        Fine labels array of shape (N,).
    indices : np.ndarray
        Indices into ``images``/``labels`` for this concept.
    concept_id : int
        Concept identifier for style transform.
    delta : float
        Style strength.
    preprocess : callable
        torchvision transform applied after style shift.
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        concept_id: int,
        delta: float,
        preprocess,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = indices
        self.concept_id = concept_id
        self.delta = delta
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = int(self.indices[idx])
        image = torch.from_numpy(self.images[row]).float() / 255.0
        image = _apply_concept_style(image, self.concept_id, self.delta)
        image = self.preprocess(image)
        return image, int(self.labels[row])


def _apply_concept_style(
    image: torch.Tensor,
    concept_id: int,
    delta: float,
) -> torch.Tensor:
    """Apply a deterministic concept-specific appearance transform.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor of shape (3, H, W) in [0, 1].
    concept_id : int
        Concept identifier selecting the style.
    delta : float
        Style strength in (0, 1].

    Returns
    -------
    torch.Tensor
        Styled image clamped to [0, 1].
    """
    style_id = concept_id % _STYLE_COUNT
    strength = float(np.clip(delta, 0.1, 1.0))
    out = image

    if style_id == 0:
        out = image
    elif style_id == 1:
        gray = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
        out = TF.adjust_brightness(gray, 1.0 - 0.12 * strength)
        out = TF.adjust_contrast(out, 1.0 + 0.8 * strength)
    elif style_id == 2:
        tint = torch.tensor(
            [1.0 + 0.25 * strength, 1.0, 1.0 - 0.25 * strength],
            dtype=image.dtype,
        ).view(3, 1, 1)
        out = (image * tint).clamp(0.0, 1.0)
        out = TF.adjust_saturation(out, max(0.2, 1.0 - 0.6 * strength))
        out = TF.adjust_contrast(out, 1.0 + 0.35 * strength)
    elif style_id == 3:
        kernel = 3 if strength < 0.45 else 5
        sigma = 0.4 + 1.2 * strength
        out = TF.gaussian_blur(image, [kernel, kernel], [sigma, sigma])
        out = TF.adjust_brightness(out, 0.95 - 0.2 * strength)
        out = TF.adjust_contrast(out, max(0.3, 1.0 - 0.45 * strength))
    else:
        threshold = 1.0 - 0.35 * strength
        out = torch.where(image > threshold, 1.0 - image, image)
        out = TF.adjust_brightness(out, 1.0 - 0.1 * strength)
        out = TF.adjust_contrast(out, 1.0 + 0.5 * strength)

    return out.clamp(0.0, 1.0)


# ------------------------------------------------------------------
# Data loading helpers
# ------------------------------------------------------------------

def _ensure_cifar10_downloaded(root: str | Path) -> None:
    """Download CIFAR-10 if not present.

    Parameters
    ----------
    root : str or Path
        Cache directory for raw CIFAR-10 data.
    """
    datasets.CIFAR10(root=str(root), train=True, download=True)


def _load_cifar10_arrays(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 training images and labels.

    Parameters
    ----------
    root : str or Path
        Cache directory.

    Returns
    -------
    images : np.ndarray
        Shape (50000, 3, 32, 32), dtype uint8.
    labels : np.ndarray
        Shape (50000,), dtype int64.
    """
    root = Path(root)
    _ensure_cifar10_downloaded(root)
    train_path = root / "cifar-10-batches-py"
    images_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    for i in range(1, 6):
        batch_file = train_path / f"data_batch_{i}"
        with open(batch_file, "rb") as f:
            payload = pickle.load(f, encoding="latin1")
        images_parts.append(payload["data"].reshape(-1, 3, 32, 32).astype(np.uint8))
        labels_parts.append(np.asarray(payload["labels"], dtype=np.int64))
    images = np.concatenate(images_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    return images, labels


def _select_concept_indices(
    labels: np.ndarray,
    concept_groups: dict[int, list[int]],
    samples_per_class: int,
    seed: int,
) -> dict[int, np.ndarray]:
    """Select balanced indices for each concept group.

    Parameters
    ----------
    labels : np.ndarray
        Full label array.
    concept_groups : dict[int, list[int]]
        Concept-to-class mapping.
    samples_per_class : int
        Maximum samples per class.
    seed : int
        Random seed.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from concept ID to index array into ``labels``.
    """
    rng = np.random.RandomState(seed)
    result: dict[int, np.ndarray] = {}
    for concept_id, classes in concept_groups.items():
        chosen: list[np.ndarray] = []
        for cls in classes:
            cls_idx = np.flatnonzero(labels == cls)
            take = min(samples_per_class, len(cls_idx))
            picked = rng.choice(cls_idx, size=take, replace=False)
            chosen.append(np.sort(picked))
        result[concept_id] = np.concatenate(chosen, axis=0)
    return result


# ------------------------------------------------------------------
# Feature cache
# ------------------------------------------------------------------

def _cache_file(config: CIFAR10RecurrenceConfig, concept_id: int) -> Path:
    """Return the cache file path for a concept's extracted features.

    Parameters
    ----------
    config : CIFAR10RecurrenceConfig
    concept_id : int

    Returns
    -------
    Path
    """
    cache_dir = Path(config.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"cifar10_recurrence_c{concept_id}"
        f"_delta{int(round(config.delta * 100))}"
        f"_spc{config.samples_per_class}"
        f"_nf{config.n_features}"
        f"_fseed{config.feature_seed}"
        f"_grp{config._groups_hash()}"
    )
    return cache_dir / f"{tag}.npz"


def _load_cached_feature_pools(
    config: CIFAR10RecurrenceConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """Attempt to load cached feature pools.

    Parameters
    ----------
    config : CIFAR10RecurrenceConfig
    n_concepts : int

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]] or None
        ``None`` if any concept's cache is missing.
    """
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
    config: CIFAR10RecurrenceConfig,
    pools: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Persist feature pools to disk.

    Parameters
    ----------
    config : CIFAR10RecurrenceConfig
    pools : dict[int, tuple[np.ndarray, np.ndarray]]
    """
    for concept_id, (X, y) in pools.items():
        path = _cache_file(config, concept_id)
        np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def _extract_feature_pools(
    config: CIFAR10RecurrenceConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Extract ResNet18 features per concept, using cache when available.

    Parameters
    ----------
    config : CIFAR10RecurrenceConfig
    n_concepts : int

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping from concept ID to (X_features, y_labels).
    """
    cached = _load_cached_feature_pools(config, n_concepts)
    if cached is not None:
        return cached

    images, labels = _load_cifar10_arrays(config.data_root)
    concept_indices = _select_concept_indices(
        labels, config.concept_groups, config.samples_per_class, config.feature_seed
    )

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    raw_pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    pin_memory = device.type == "cuda"

    with torch.inference_mode():
        for concept_id in range(n_concepts):
            if concept_id not in concept_indices:
                # Concept ID generated by the matrix but beyond explicit groups;
                # wrap around to reuse an existing group with a different style.
                source_id = concept_id % len(config.concept_groups)
                indices = concept_indices[source_id]
            else:
                indices = concept_indices[concept_id]

            dataset = _ConceptImageDataset(
                images=images,
                labels=labels,
                indices=indices,
                concept_id=concept_id,
                delta=config.delta,
                preprocess=preprocess,
            )
            loader = DataLoader(
                dataset,
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

            raw_pools[concept_id] = (
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
            concept_id: (pca.transform(X).astype(np.float32), y)
            for concept_id, (X, y) in raw_pools.items()
        }
    else:
        pools = raw_pools

    _save_feature_pools(config, pools)
    return pools


# ------------------------------------------------------------------
# Batch sampling
# ------------------------------------------------------------------

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
        Feature matrix for the concept.
    y_pool : np.ndarray
        Label array for the concept.
    n_samples : int
        Total samples to draw.
    rng : np.random.RandomState
        Seeded RNG.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X_batch, y_batch) with shapes ``(n_samples, D)`` and ``(n_samples,)``.
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


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def prepare_cifar10_recurrence_feature_cache(
    config: CIFAR10RecurrenceConfig | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Prepare and cache per-concept feature pools.

    Parameters
    ----------
    config : CIFAR10RecurrenceConfig or None
        Configuration; uses defaults if ``None``.

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping from concept ID to (features, labels).
    """
    if config is None:
        config = CIFAR10RecurrenceConfig()
    gen_config = config.to_generator_config()
    return _extract_feature_pools(config, gen_config.n_concepts)


def generate_cifar10_recurrence_dataset(
    config: CIFAR10RecurrenceConfig | None = None,
) -> DriftDataset:
    """Generate a CIFAR-10 real-data drift dataset.

    Parameters
    ----------
    config : CIFAR10RecurrenceConfig or None
        Configuration; uses defaults if ``None``.

    Returns
    -------
    DriftDataset
        Complete drift dataset with concept matrix and per-(client, time) data.
    """
    if config is None:
        config = CIFAR10RecurrenceConfig()

    gen_config = config.to_generator_config()
    concept_matrix = generate_concept_matrix(
        K=config.K,
        T=config.T,
        n_concepts=gen_config.n_concepts,
        alpha=config.alpha,
        seed=config.seed,
    )
    n_concepts = int(concept_matrix.max()) + 1

    feature_pools = _extract_feature_pools(config, n_concepts)
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(config.K):
        for t in range(config.T):
            concept_id = int(concept_matrix[k, t])
            rng = np.random.RandomState(config.seed + 10000 + k * config.T + t)
            X_pool, y_pool = feature_pools[concept_id]
            data[(k, t)] = _draw_balanced_batch(
                X_pool, y_pool, config.n_samples, rng
            )

    concept_specs = [
        ConceptSpec(
            concept_id=concept_id,
            generator_type="cifar10_recurrence",
            variant=concept_id,
            noise_scale=config.delta,
        )
        for concept_id in range(n_concepts)
    ]

    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=gen_config,
        concept_specs=concept_specs,
    )
