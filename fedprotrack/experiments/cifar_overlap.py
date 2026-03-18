from __future__ import annotations

"""Shared CIFAR-100 subset benchmark helpers for failure-diagnosis runs."""

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

from ..baselines.comm_tracker import model_bytes
from ..drift_generator.configs import GeneratorConfig
from ..drift_generator.data_streams import ConceptSpec
from ..drift_generator.generator import DriftDataset
from ..models import TorchLinearClassifier
from ..posterior.presets import make_plan_c_config

TOTAL_COARSE_CLASSES = 20


@dataclass
class CIFARSubsetBenchmarkConfig:
    """Configuration for CIFAR subset-based recurrence benchmarks."""

    K: int
    T: int
    n_samples: int
    n_features: int
    seed: int
    data_root: str = ".cifar100_cache"
    switch_prob: float = 0.15
    rho: float = 2.5
    alpha: float = 0.7
    delta: float = 0.85
    generator_type: str = "cifar100_overlap"


def build_overlap_concept_classes(
    overlap_ratio: float,
    *,
    n_concepts: int = 4,
    n_classes_per_concept: int = 5,
) -> dict[int, list[int]]:
    """Build concept-specific coarse-class subsets with controlled overlap."""
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
        )
    if n_concepts * n_classes_per_concept != TOTAL_COARSE_CLASSES:
        raise ValueError(
            "n_concepts * n_classes_per_concept must equal 20 for CIFAR-100"
        )

    n_overlap = int(round(overlap_ratio * n_classes_per_concept))
    base_blocks = [
        list(range(i * n_classes_per_concept, (i + 1) * n_classes_per_concept))
        for i in range(n_concepts)
    ]

    concept_classes: dict[int, list[int]] = {}
    for cid in range(n_concepts):
        own_block = base_blocks[cid]
        next_block = base_blocks[(cid + 1) % n_concepts]
        unique_count = n_classes_per_concept - n_overlap
        concept = own_block[:unique_count] + next_block[:n_overlap]
        concept_classes[cid] = concept
    return concept_classes


def build_phase_concept_matrix(
    K: int,
    T: int,
    *,
    phase_concepts: list[list[int]],
    seed: int,
    switch_prob: float = 0.15,
) -> np.ndarray:
    """Build a phased recurrence concept matrix."""
    if K < 1 or T < 1:
        raise ValueError(f"K and T must be positive, got K={K}, T={T}")
    if not phase_concepts:
        raise ValueError("phase_concepts must be non-empty")

    rng = np.random.RandomState(seed)
    cm = np.zeros((K, T), dtype=np.int32)
    phase_boundaries = np.linspace(0, T, len(phase_concepts) + 1, dtype=int)

    for phase_idx, active in enumerate(phase_concepts):
        t_start = int(phase_boundaries[phase_idx])
        t_end = int(phase_boundaries[phase_idx + 1])
        if not active:
            raise ValueError("Each phase must have at least one active concept")

        for k in range(K):
            concept = int(active[k % len(active)])
            for t in range(t_start, t_end):
                if t > t_start and rng.random() < switch_prob:
                    concept = int(active[rng.randint(len(active))])
                cm[k, t] = concept

    return cm


def _load_cifar100(data_root: str) -> tuple[np.ndarray, np.ndarray]:
    from torchvision import datasets

    datasets.CIFAR100(root=data_root, train=True, download=True)
    path = Path(data_root) / "cifar-100-python" / "train"
    with open(path, "rb") as f:
        payload = pickle.load(f, encoding="latin1")
    images = payload["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    coarse_labels = np.asarray(payload["coarse_labels"], dtype=np.int64)
    return images, coarse_labels


def _extract_features(images: np.ndarray, batch_size: int = 256) -> np.ndarray:
    from torchvision.models import ResNet18_Weights, resnet18

    cache_path = Path(".feature_cache") / "cifar100_raw_resnet18.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    parts: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[i:i + batch_size]).float() / 255.0
            batch = torch.stack([preprocess(img) for img in batch])
            feats = backbone(batch.to(device, non_blocking=True))
            parts.append(feats.cpu().numpy())

    all_feats = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(cache_path, all_feats)
    return all_feats


def build_subset_dataset(
    config: CIFARSubsetBenchmarkConfig,
    *,
    concept_classes: dict[int, list[int]],
    phase_concepts: list[list[int]],
) -> DriftDataset:
    """Build a CIFAR-100 feature dataset from concept-specific label subsets."""
    images, coarse_labels = _load_cifar100(config.data_root)
    raw_features = _extract_features(images)
    pca = PCA(n_components=config.n_features, random_state=2718)
    features = pca.fit_transform(raw_features).astype(np.float32)

    concept_matrix = build_phase_concept_matrix(
        config.K,
        config.T,
        phase_concepts=phase_concepts,
        seed=config.seed,
        switch_prob=config.switch_prob,
    )
    actual_n_concepts = int(concept_matrix.max()) + 1

    concept_pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid in range(actual_n_concepts):
        cls_subset = concept_classes[cid]
        mask = np.isin(coarse_labels, cls_subset)
        X_pool = features[mask]
        y_raw = coarse_labels[mask]
        label_map = {c: i for i, c in enumerate(sorted(set(cls_subset)))}
        y_pool = np.array([label_map[int(y)] for y in y_raw], dtype=np.int64)
        concept_pools[cid] = (X_pool, y_pool)

    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(config.K):
        for t in range(config.T):
            cid = int(concept_matrix[k, t])
            rng = np.random.RandomState(config.seed + 10000 + k * config.T + t)
            X_pool, y_pool = concept_pools[cid]
            classes = np.unique(y_pool)
            per_class = config.n_samples // len(classes)
            remainder = config.n_samples % len(classes)
            chosen = []
            for off, cls in enumerate(classes):
                cls_idx = np.flatnonzero(y_pool == cls)
                take = per_class + (1 if off < remainder else 0)
                chosen.append(rng.choice(cls_idx, size=take, replace=True))
            batch_idx = np.concatenate(chosen)
            rng.shuffle(batch_idx)
            data[(k, t)] = (X_pool[batch_idx], y_pool[batch_idx])

    gen_config = GeneratorConfig(
        K=config.K,
        T=config.T,
        n_samples=config.n_samples,
        rho=config.rho,
        alpha=config.alpha,
        delta=config.delta,
        generator_type=config.generator_type,
        seed=config.seed,
    )
    concept_specs = [
        ConceptSpec(
            concept_id=cid,
            generator_type=config.generator_type,
            variant=cid,
            noise_scale=float(config.delta),
        )
        for cid in range(actual_n_concepts)
    ]
    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=gen_config,
        concept_specs=concept_specs,
    )


def run_local(ds: DriftDataset, epochs: int, lr: float, seed: int) -> tuple[np.ndarray, float]:
    """Run local-only training on a prepared subset dataset."""
    K, T = ds.config.K, ds.config.T
    nf = ds.data[(0, 0)][0].shape[1]
    nc = max(len(np.unique(ds.data[(k, t)][1])) for k in range(K) for t in range(T))
    models = [
        TorchLinearClassifier(
            n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed + k
        )
        for k in range(K)
    ]
    acc = np.zeros((K, T), dtype=np.float64)
    for t in range(T):
        for k in range(K):
            X, y = ds.data[(k, t)]
            mid = len(X) // 2
            acc[k, t] = float(np.mean(models[k].predict(X[:mid]) == y[:mid]))
            if epochs > 1:
                models[k].fit(X[mid:], y[mid:])
            else:
                models[k].partial_fit(X[mid:], y[mid:])
    return acc, 0.0


def run_oracle(
    ds: DriftDataset,
    fed_every: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Run oracle concept-aware aggregation on a prepared subset dataset."""
    K, T = ds.config.K, ds.config.T
    gt = ds.concept_matrix
    nf = ds.data[(0, 0)][0].shape[1]
    nc = max(len(np.unique(ds.data[(k, t)][1])) for k in range(K) for t in range(T))
    models = [
        TorchLinearClassifier(
            n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed + k
        )
        for k in range(K)
    ]
    acc = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0
    concept_models: dict[int, dict[str, np.ndarray]] = {}

    for t in range(T):
        for k in range(K):
            X, y = ds.data[(k, t)]
            mid = len(X) // 2
            cid = int(gt[k, t])
            if t > 0 and gt[k, t] != gt[k, t - 1] and cid in concept_models:
                models[k].set_params(concept_models[cid])
            acc[k, t] = float(np.mean(models[k].predict(X[:mid]) == y[:mid]))
            if epochs > 1:
                models[k].fit(X[mid:], y[mid:])
            else:
                models[k].partial_fit(X[mid:], y[mid:])

        if (t + 1) % fed_every == 0 and t < T - 1:
            plist = [m.get_params() for m in models]
            one_b = model_bytes(plist[0], precision_bits=32)
            concepts_at_t = {k: int(gt[k, t]) for k in range(K)}
            for cid in set(concepts_at_t.values()):
                members = [k for k in range(K) if concepts_at_t[k] == cid]
                if len(members) < 2:
                    continue
                cp = {
                    key: np.mean(np.stack([plist[k][key] for k in members]), axis=0)
                    for key in plist[0]
                }
                for k in members:
                    models[k].set_params(cp)
                total_bytes += len(members) * one_b * 2
                concept_models[cid] = {k: v.copy() for k, v in cp.items()}
    return acc, total_bytes


def run_fedavg(
    ds: DriftDataset,
    fed_every: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Run FedAvg on a prepared subset dataset."""
    K, T = ds.config.K, ds.config.T
    nf = ds.data[(0, 0)][0].shape[1]
    nc = max(len(np.unique(ds.data[(k, t)][1])) for k in range(K) for t in range(T))
    gm = TorchLinearClassifier(n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed)
    cms = [
        TorchLinearClassifier(
            n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed + k
        )
        for k in range(K)
    ]
    acc = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0
    for t in range(T):
        for k in range(K):
            X, y = ds.data[(k, t)]
            mid = len(X) // 2
            acc[k, t] = float(np.mean(gm.predict(X[:mid]) == y[:mid]))
            if t > 0:
                cms[k].set_params(gm.get_params())
            if epochs > 1:
                cms[k].fit(X[mid:], y[mid:])
            else:
                cms[k].partial_fit(X[mid:], y[mid:])
        if (t + 1) % fed_every == 0 and t < T - 1:
            pl = [m.get_params() for m in cms]
            one_b = model_bytes(pl[0], precision_bits=32)
            total_bytes += K * one_b * 2
            gp = {
                key: np.mean(np.stack([p[key] for p in pl]), axis=0)
                for key in pl[0]
            }
            gm.set_params(gp)
    return acc, total_bytes


def run_fpt(
    ds: DriftDataset,
    fed_every: int,
    epochs: int,
    lr: float,
    seed: int,
    *,
    dormant_recall: bool = False,
    loss_novelty_threshold: float = 0.25,
    merge_threshold: float = 0.80,
    max_concepts: int = 10,
    model_type: str = "linear",
    hidden_dim: int = 64,
    adapter_dim: int = 16,
) -> tuple[np.ndarray, float, int, int, int]:
    """Run FedProTrack on a prepared subset dataset."""
    from ..posterior.fedprotrack_runner import FedProTrackRunner
    from ..posterior.two_phase_protocol import TwoPhaseConfig

    gt = ds.concept_matrix
    n_concepts = int(gt.max()) + 1
    runner = FedProTrackRunner(
        config=make_plan_c_config(
            loss_novelty_threshold=loss_novelty_threshold,
            merge_threshold=merge_threshold,
            max_concepts=max(max_concepts, n_concepts + 2),
        ),
        federation_every=fed_every,
        detector_name="ADWIN",
        seed=seed,
        lr=lr,
        n_epochs=epochs,
        soft_aggregation=True,
        blend_alpha=0.0,
        dormant_recall=dormant_recall,
        model_type=model_type,
        hidden_dim=hidden_dim,
        adapter_dim=adapter_dim,
    )
    result = runner.run(ds)
    return (
        result.accuracy_matrix,
        result.total_bytes,
        result.spawned_concepts,
        result.merged_concepts,
        result.active_concepts,
    )
