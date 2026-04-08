"""Fashion-MNIST recurrence benchmark for federated concept-drift experiments.

Builds a recurring-concept dataset from Fashion-MNIST by:
1. defining manual concept groupings over the 10 classes
   (e.g. upper-body, lower-body, footwear, bags, outerwear),
2. converting grayscale images to 3-channel for ResNet18 compatibility,
3. extracting pretrained ResNet18 embeddings,
4. reducing them with PCA, and
5. sampling balanced federated batches according to the shared concept matrix.

Fashion-MNIST classes:
    0 = T-shirt/top, 1 = Trouser, 2 = Pullover, 3 = Dress, 4 = Coat,
    5 = Sandal, 6 = Shirt, 7 = Sneaker, 8 = Bag, 9 = Ankle boot

This keeps the real-data benchmark compatible with the existing
``DriftDataset`` / ``FedProTrackRunner`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
# Groupings pair semantically related garment / accessory types:
#   concept 0 (tops):      T-shirt/top, Shirt
#   concept 1 (outerwear): Pullover, Coat
#   concept 2 (full-body): Trouser, Dress
#   concept 3 (footwear):  Sandal, Sneaker
#   concept 4 (accessories): Bag, Ankle boot
DEFAULT_CONCEPT_GROUPS: dict[int, list[int]] = {
    0: [0, 6],
    1: [2, 4],
    2: [1, 3],
    3: [5, 7],
    4: [8, 9],
}


@dataclass
class FMNISTRecurrenceConfig:
    """Configuration for the Fashion-MNIST recurrence benchmark.

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
        Mapping from concept ID to list of Fashion-MNIST class indices.
    data_root : str
        Directory for raw Fashion-MNIST download cache.
    feature_cache_dir : str
        Directory for extracted feature cache.
    feature_seed : int
        Seed for balanced index selection and PCA.
    seed : int
        Master seed for concept matrix and batch sampling.
    dirichlet_alpha : float or None
        Dirichlet concentration parameter for non-IID label
        distributions.  When ``None`` (default), the existing
        class-balanced sampling is used (backward compatible).
        When set to a positive float, each (client, round) batch
        samples label proportions from
        ``Dir(dirichlet_alpha * ones(n_classes_in_concept))``.
        Lower values (e.g. 0.01) produce highly skewed distributions;
        higher values (e.g. 100.0) approach uniform.
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
    data_root: str = ".fmnist_cache"
    feature_cache_dir: str = ".feature_cache"
    feature_seed: int = 2718
    seed: int = 42
    dirichlet_alpha: float | None = None

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
        for cid, classes in self.concept_groups.items():
            if not classes:
                raise ValueError(f"concept_groups[{cid}] is empty")
            for cls in classes:
                if cls < 0 or cls >= _N_CLASSES:
                    raise ValueError(
                        f"concept_groups[{cid}] contains invalid class {cls}; "
                        f"must be in [0, {_N_CLASSES - 1}]"
                    )
        if self.dirichlet_alpha is not None and self.dirichlet_alpha <= 0.0:
            raise ValueError(
                f"dirichlet_alpha must be > 0 when set, got {self.dirichlet_alpha}"
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
            generator_type="fmnist_recurrence",
            seed=self.seed,
        )

    def _groups_hash(self) -> str:
        """Deterministic short hash of concept_groups for cache keying."""
        raw = str(sorted((k, sorted(v)) for k, v in self.concept_groups.items()))
        return hashlib.md5(raw.encode()).hexdigest()[:8]


class _ConceptImageDataset(Dataset):
    """Dataset view for one concept-specific Fashion-MNIST appearance style.

    Grayscale images are replicated to 3 channels for ResNet18 compatibility.

    Parameters
    ----------
    images : np.ndarray
        Full image array of shape (N, 28, 28), dtype uint8.
    labels : np.ndarray
        Label array of shape (N,).
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
        # Fashion-MNIST is (28, 28) grayscale; expand to (3, 28, 28)
        gray = torch.from_numpy(self.images[row]).float() / 255.0
        image = gray.unsqueeze(0).repeat(3, 1, 1)  # (3, 28, 28)
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
        out = TF.adjust_brightness(image, 1.0 - 0.12 * strength)
        out = TF.adjust_contrast(out, 1.0 + 0.8 * strength)
    elif style_id == 2:
        tint = torch.tensor(
            [1.0 + 0.25 * strength, 1.0, 1.0 - 0.25 * strength],
            dtype=image.dtype,
        ).view(3, 1, 1)
        out = (image * tint).clamp(0.0, 1.0)
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

def _ensure_fmnist_downloaded(root: str | Path) -> None:
    """Download Fashion-MNIST if not present.

    Parameters
    ----------
    root : str or Path
        Cache directory.
    """
    datasets.FashionMNIST(root=str(root), train=True, download=True)


def _load_fmnist_arrays(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST training images and labels.

    Parameters
    ----------
    root : str or Path
        Cache directory.

    Returns
    -------
    images : np.ndarray
        Shape (60000, 28, 28), dtype uint8.
    labels : np.ndarray
        Shape (60000,), dtype int64.
    """
    root = Path(root)
    _ensure_fmnist_downloaded(root)
    ds = datasets.FashionMNIST(root=str(root), train=True, download=False)
    images = ds.data.numpy().astype(np.uint8)  # (60000, 28, 28)
    labels = ds.targets.numpy().astype(np.int64)
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
        Mapping from concept ID to index array.
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

def _cache_file(config: FMNISTRecurrenceConfig, concept_id: int) -> Path:
    """Return the cache file path for a concept's extracted features.

    Parameters
    ----------
    config : FMNISTRecurrenceConfig
    concept_id : int

    Returns
    -------
    Path
    """
    cache_dir = Path(config.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"fmnist_recurrence_c{concept_id}"
        f"_delta{int(round(config.delta * 100))}"
        f"_spc{config.samples_per_class}"
        f"_nf{config.n_features}"
        f"_fseed{config.feature_seed}"
        f"_grp{config._groups_hash()}"
    )
    return cache_dir / f"{tag}.npz"


def _load_cached_feature_pools(
    config: FMNISTRecurrenceConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """Attempt to load cached feature pools.

    Parameters
    ----------
    config : FMNISTRecurrenceConfig
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
    config: FMNISTRecurrenceConfig,
    pools: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Persist feature pools to disk.

    Parameters
    ----------
    config : FMNISTRecurrenceConfig
    pools : dict[int, tuple[np.ndarray, np.ndarray]]
    """
    for concept_id, (X, y) in pools.items():
        path = _cache_file(config, concept_id)
        np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def _extract_feature_pools(
    config: FMNISTRecurrenceConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Extract ResNet18 features per concept, using cache when available.

    Grayscale Fashion-MNIST images are replicated to 3 channels before
    being passed through the pretrained ResNet18 backbone.

    Parameters
    ----------
    config : FMNISTRecurrenceConfig
    n_concepts : int

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping from concept ID to (X_features, y_labels).
    """
    cached = _load_cached_feature_pools(config, n_concepts)
    if cached is not None:
        return cached

    images, labels = _load_fmnist_arrays(config.data_root)
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


def _draw_dirichlet_batch(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_samples: int,
    dirichlet_alpha: float,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a Dirichlet-weighted non-IID batch from one concept pool.

    Parameters
    ----------
    X_pool : np.ndarray
        Feature matrix for the concept.
    y_pool : np.ndarray
        Label array for the concept.
    n_samples : int
        Total samples to draw.
    dirichlet_alpha : float
        Concentration parameter.  Lower values produce more skewed
        label distributions; higher values approach uniform.
    rng : np.random.RandomState
        Seeded RNG.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X_batch, y_batch) with shapes ``(n_samples, D)`` and ``(n_samples,)``.
    """
    classes = np.unique(y_pool)
    n_classes = len(classes)
    proportions = rng.dirichlet(dirichlet_alpha * np.ones(n_classes))

    # Convert proportions to integer counts that sum to n_samples.
    raw_counts = proportions * n_samples
    class_counts = np.floor(raw_counts).astype(int)
    remainder = n_samples - class_counts.sum()
    fracs = raw_counts - class_counts
    top_indices = np.argsort(fracs)[::-1][:remainder]
    class_counts[top_indices] += 1

    chosen: list[np.ndarray] = []
    for i, cls in enumerate(classes):
        cls_idx = np.flatnonzero(y_pool == cls)
        take = class_counts[i]
        if take > 0:
            chosen.append(rng.choice(cls_idx, size=take, replace=True))

    batch_idx = np.concatenate(chosen, axis=0)
    rng.shuffle(batch_idx)
    return X_pool[batch_idx].astype(np.float32), y_pool[batch_idx].astype(np.int64)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def prepare_fmnist_recurrence_feature_cache(
    config: FMNISTRecurrenceConfig | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Prepare and cache per-concept feature pools.

    Parameters
    ----------
    config : FMNISTRecurrenceConfig or None
        Configuration; uses defaults if ``None``.

    Returns
    -------
    dict[int, tuple[np.ndarray, np.ndarray]]
        Mapping from concept ID to (features, labels).
    """
    if config is None:
        config = FMNISTRecurrenceConfig()
    gen_config = config.to_generator_config()
    return _extract_feature_pools(config, gen_config.n_concepts)


def generate_fmnist_recurrence_dataset(
    config: FMNISTRecurrenceConfig | None = None,
) -> DriftDataset:
    """Generate a Fashion-MNIST real-data drift dataset.

    Parameters
    ----------
    config : FMNISTRecurrenceConfig or None
        Configuration; uses defaults if ``None``.

    Returns
    -------
    DriftDataset
        Complete drift dataset with concept matrix and per-(client, time) data.
    """
    if config is None:
        config = FMNISTRecurrenceConfig()

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
            if config.dirichlet_alpha is not None:
                rng = np.random.RandomState(
                    config.seed + 20000 + k * config.T + t
                )
                X_pool, y_pool = feature_pools[concept_id]
                data[(k, t)] = _draw_dirichlet_batch(
                    X_pool, y_pool, config.n_samples,
                    config.dirichlet_alpha, rng,
                )
            else:
                rng = np.random.RandomState(
                    config.seed + 10000 + k * config.T + t
                )
                X_pool, y_pool = feature_pools[concept_id]
                data[(k, t)] = _draw_balanced_batch(
                    X_pool, y_pool, config.n_samples, rng,
                )

    concept_specs = [
        ConceptSpec(
            concept_id=concept_id,
            generator_type="fmnist_recurrence",
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
