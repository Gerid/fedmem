"""CIFAR-100 recurrence benchmark for federated concept-drift experiments.

Builds a recurring-concept dataset from CIFAR-100 by:
1. reading official coarse labels (20 superclasses),
2. applying deterministic concept-specific appearance shifts,
3. extracting pretrained ResNet18 embeddings,
4. reducing them with PCA, and
5. sampling balanced federated batches according to the shared concept matrix.

This keeps the real-data benchmark compatible with the existing
``DriftDataset`` / ``FedProTrackRunner`` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
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

from ..drift_generator.concept_matrix import (
    generate_concept_matrix,
    generate_concept_matrix_low_singleton,
)
from ..drift_generator.configs import GeneratorConfig
from ..drift_generator.data_streams import ConceptSpec
from ..drift_generator.generator import DriftDataset


_N_COARSE_CLASSES = 20
_STYLE_COUNT = 5


@dataclass
class CIFAR100RecurrenceConfig:
    """Configuration for the CIFAR-100 recurrence benchmark.

    Parameters
    ----------
    label_split : str
        How concepts differ in label distribution:
        - ``"none"`` (or ``"shared"``): all concepts share the same 20
          coarse classes (concepts differ only by visual style).
          Original behaviour.
        - ``"disjoint"``: each concept gets a non-overlapping partition
          of the 20 coarse classes.
        - ``"overlap"`` (or ``"overlapping"``): each concept gets
          ``n_classes_per_concept`` coarse classes, with controlled
          overlap between neighbours.
    n_classes_per_concept : int
        Number of coarse classes per concept when
        ``label_split="overlap"``.  Ignored for other modes.
        Must be in ``[2, 20]``.
    """

    K: int = 6
    T: int = 12
    n_samples: int = 400
    rho: float = 3.0
    alpha: float = 0.75
    delta: float = 0.85
    n_features: int = 128
    samples_per_coarse_class: int = 120
    batch_size: int = 256
    n_workers: int = 2
    data_root: str = ".cifar100_cache"
    feature_cache_dir: str = ".feature_cache"
    feature_seed: int = 2718
    seed: int = 42
    label_split: str = "none"
    n_classes_per_concept: int = 10
    min_group_size: int = 1

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.T < 2:
            raise ValueError(f"T must be >= 2, got {self.T}")
        if self.n_samples < 2:
            raise ValueError(
                f"n_samples must be >= 2, got {self.n_samples}"
            )
        if self.samples_per_coarse_class < 1:
            raise ValueError(
                "samples_per_coarse_class must be >= 1, "
                f"got {self.samples_per_coarse_class}"
            )
        if self.n_features < 2:
            raise ValueError(f"n_features must be >= 2, got {self.n_features}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not (0.0 < self.delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {self.delta}")
        # Normalise convenient aliases.
        _aliases = {"shared": "none", "overlapping": "overlap"}
        if self.label_split in _aliases:
            object.__setattr__(self, "label_split", _aliases[self.label_split])
        if self.label_split not in ("none", "disjoint", "overlap"):
            raise ValueError(
                f"label_split must be 'none'/'shared', 'disjoint', or "
                f"'overlap'/'overlapping', got {self.label_split!r}"
            )
        if self.min_group_size < 1:
            raise ValueError(
                f"min_group_size must be >= 1, got {self.min_group_size}"
            )
        if self.label_split == "overlap":
            if not (2 <= self.n_classes_per_concept <= _N_COARSE_CLASSES):
                raise ValueError(
                    f"n_classes_per_concept must be in [2, {_N_COARSE_CLASSES}], "
                    f"got {self.n_classes_per_concept}"
                )

    def to_generator_config(self) -> GeneratorConfig:
        return GeneratorConfig(
            K=self.K,
            T=self.T,
            n_samples=self.n_samples,
            rho=self.rho,
            alpha=self.alpha,
            delta=self.delta,
            generator_type="cifar100_recurrence",
            seed=self.seed,
        )


def _concept_class_subsets(
    n_concepts: int,
    label_split: str,
    n_classes_per_concept: int = 10,
) -> dict[int, np.ndarray]:
    """Return the coarse-class subset for each concept.

    Parameters
    ----------
    n_concepts : int
        Number of distinct concepts.
    label_split : str
        ``"none"``/``"shared"`` (all 20 classes),
        ``"disjoint"`` (non-overlapping partitions), or
        ``"overlap"``/``"overlapping"`` (sliding window with
        controlled overlap).
    n_classes_per_concept : int
        Classes per concept for ``"overlap"`` mode.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from concept_id to sorted array of coarse class indices.
    """
    all_classes = np.arange(_N_COARSE_CLASSES)

    if label_split == "none":
        return {c: all_classes.copy() for c in range(n_concepts)}

    if label_split == "disjoint":
        # Split 20 classes evenly across concepts.  If n_concepts does not
        # divide 20 evenly, the last concept gets the remainder.
        per = _N_COARSE_CLASSES // n_concepts
        subsets: dict[int, np.ndarray] = {}
        for c in range(n_concepts):
            start = c * per
            end = start + per if c < n_concepts - 1 else _N_COARSE_CLASSES
            subsets[c] = all_classes[start:end]
        return subsets

    # "overlap" mode: sliding window around the 20 classes (wrapping).
    stride = _N_COARSE_CLASSES / n_concepts
    subsets_ov: dict[int, np.ndarray] = {}
    for c in range(n_concepts):
        start = int(round(c * stride))
        indices = np.array(
            [(start + j) % _N_COARSE_CLASSES for j in range(n_classes_per_concept)]
        )
        subsets_ov[c] = np.sort(indices)
    return subsets_ov


class _ConceptImageDataset(Dataset):
    """Dataset view for one concept-specific CIFAR appearance style."""

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
    """Apply a deterministic concept-specific appearance transform."""
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


def _ensure_cifar100_downloaded(root: str | Path) -> None:
    """Download CIFAR-100 if the raw archive is not present."""
    datasets.CIFAR100(root=str(root), train=True, download=True)


def _load_cifar100_arrays(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-100 images and official coarse labels from raw pickle."""
    root = Path(root)
    _ensure_cifar100_downloaded(root)
    train_path = root / "cifar-100-python" / "train"
    with open(train_path, "rb") as f:
        payload = pickle.load(f, encoding="latin1")

    images = payload["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    coarse_labels = np.asarray(payload["coarse_labels"], dtype=np.int64)
    return images, coarse_labels


def _select_balanced_indices(
    labels: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    """Choose the same number of source images from each coarse class."""
    rng = np.random.RandomState(seed)
    chosen: list[np.ndarray] = []
    for cls in range(_N_COARSE_CLASSES):
        cls_idx = np.flatnonzero(labels == cls)
        take = min(samples_per_class, len(cls_idx))
        picked = rng.choice(cls_idx, size=take, replace=False)
        chosen.append(np.sort(picked))
    return np.concatenate(chosen, axis=0)


def _cache_file(
    config: CIFAR100RecurrenceConfig,
    concept_id: int,
    n_concepts: int | None = None,
) -> Path:
    cache_dir = Path(config.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_tag = config.label_split
    if config.label_split == "overlap":
        split_tag = f"overlap{config.n_classes_per_concept}"
    # Include n_concepts in tag so disjoint/overlap caches with different
    # concept counts don't collide (class subsets depend on n_concepts).
    nc_tag = f"_nc{n_concepts}" if n_concepts is not None else ""
    tag = (
        f"cifar100_recurrence_c{concept_id}"
        f"_delta{int(round(config.delta * 100))}"
        f"_spc{config.samples_per_coarse_class}"
        f"_nf{config.n_features}"
        f"_fseed{config.feature_seed}"
        f"_ls{split_tag}"
        f"{nc_tag}"
    )
    return cache_dir / f"{tag}.npz"


def _load_cached_feature_pools(
    config: CIFAR100RecurrenceConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for concept_id in range(n_concepts):
        path = _cache_file(config, concept_id, n_concepts=n_concepts)
        if not path.exists():
            return None
        with np.load(path) as payload:
            pools[concept_id] = (
                payload["X"].astype(np.float32),
                payload["y"].astype(np.int64),
            )
    return pools


def _save_feature_pools(
    config: CIFAR100RecurrenceConfig,
    pools: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    n_concepts = len(pools)
    for concept_id, (X, y) in pools.items():
        path = _cache_file(config, concept_id, n_concepts=n_concepts)
        np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


def _extract_feature_pools(
    config: CIFAR100RecurrenceConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    cached = _load_cached_feature_pools(config, n_concepts)
    if cached is not None:
        return cached

    images, coarse_labels = _load_cifar100_arrays(config.data_root)
    subset_indices = _select_balanced_indices(
        coarse_labels, config.samples_per_coarse_class, config.feature_seed
    )

    # Compute per-concept class subsets for label-split modes.
    class_subsets = _concept_class_subsets(
        n_concepts, config.label_split, config.n_classes_per_concept
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
            # Filter indices to only include this concept's classes.
            allowed_classes = set(int(c) for c in class_subsets[concept_id])
            concept_indices = np.array(
                [i for i in subset_indices if int(coarse_labels[i]) in allowed_classes]
            )

            dataset = _ConceptImageDataset(
                images=images,
                labels=coarse_labels,
                indices=concept_indices,
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


def _draw_balanced_batch(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    n_samples: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a class-balanced batch from one concept pool."""
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


def prepare_cifar100_recurrence_feature_cache(
    config: CIFAR100RecurrenceConfig | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Prepare and cache per-concept feature pools."""
    if config is None:
        config = CIFAR100RecurrenceConfig()
    gen_config = config.to_generator_config()
    return _extract_feature_pools(config, gen_config.n_concepts)


def generate_cifar100_recurrence_dataset(
    config: CIFAR100RecurrenceConfig | None = None,
) -> DriftDataset:
    """Generate a CIFAR-100 real-data drift dataset."""
    if config is None:
        config = CIFAR100RecurrenceConfig()

    gen_config = config.to_generator_config()
    concept_matrix = generate_concept_matrix_low_singleton(
        K=config.K,
        T=config.T,
        n_concepts=gen_config.n_concepts,
        alpha=config.alpha,
        seed=config.seed,
        min_group_size=config.min_group_size,
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
            generator_type="cifar100_recurrence",
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
