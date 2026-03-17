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

from ..drift_generator.concept_matrix import generate_concept_matrix
from ..drift_generator.configs import GeneratorConfig
from ..drift_generator.data_streams import ConceptSpec
from ..drift_generator.generator import DriftDataset


_N_COARSE_CLASSES = 20
_STYLE_COUNT = 5


@dataclass
class CIFAR100RecurrenceConfig:
    """Configuration for the CIFAR-100 recurrence benchmark."""

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

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.T < 2:
            raise ValueError(f"T must be >= 2, got {self.T}")
        if self.n_samples < _N_COARSE_CLASSES:
            raise ValueError(
                f"n_samples must be >= {_N_COARSE_CLASSES}, got {self.n_samples}"
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


def _cache_file(config: CIFAR100RecurrenceConfig, concept_id: int) -> Path:
    cache_dir = Path(config.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"cifar100_recurrence_c{concept_id}"
        f"_delta{int(round(config.delta * 100))}"
        f"_spc{config.samples_per_coarse_class}"
        f"_nf{config.n_features}"
        f"_fseed{config.feature_seed}"
    )
    return cache_dir / f"{tag}.npz"


def _load_cached_feature_pools(
    config: CIFAR100RecurrenceConfig,
    n_concepts: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
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
    config: CIFAR100RecurrenceConfig,
    pools: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    for concept_id, (X, y) in pools.items():
        path = _cache_file(config, concept_id)
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
            dataset = _ConceptImageDataset(
                images=images,
                labels=coarse_labels,
                indices=subset_indices,
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
