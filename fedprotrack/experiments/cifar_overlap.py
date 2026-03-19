from __future__ import annotations

"""Shared CIFAR-100 subset benchmark helpers for failure-diagnosis runs."""

from dataclasses import dataclass
import pickle
from pathlib import Path
from collections.abc import Mapping, Sequence

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
RESULT_CONFIG_FIELDS = (
    "model_type",
    "fingerprint_source",
    "expert_update_policy",
    "shared_update_policy",
    "global_shared_aggregation",
    "routed_write_top_k",
    "routed_write_min_secondary_weight",
    "routed_read_top_k",
    "routed_read_temperature",
    "routed_read_only_on_ambiguity",
    "routed_read_min_entropy",
    "routed_read_min_secondary_weight",
    "routed_read_max_primary_gap",
    "max_spawn_clusters_per_round",
    "novelty_hysteresis_rounds",
    "factorized_slot_preserving",
    "factorized_primary_anchor_alpha",
    "factorized_secondary_anchor_alpha",
    "factorized_primary_consolidation_steps",
    "factorized_primary_consolidation_mode",
)
RESULT_ROUTING_FIELDS = (
    "spawned",
    "merged",
    "active",
    "assignment_switch_rate",
    "avg_clients_per_concept",
    "singleton_group_ratio",
    "memory_reuse_rate",
    "routing_consistency",
)
RESULT_MODEL_DIAGNOSTIC_FIELDS = (
    "shared_drift_norm",
    "expert_update_coverage",
    "multi_route_rate",
)


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
    fingerprint_label_pools: dict[int, np.ndarray] = {}
    for cid in range(actual_n_concepts):
        cls_subset = concept_classes[cid]
        mask = np.isin(coarse_labels, cls_subset)
        X_pool = features[mask]
        y_raw = coarse_labels[mask].astype(np.int64, copy=False)
        label_map = {c: i for i, c in enumerate(sorted(set(cls_subset)))}
        y_pool = np.array([label_map[int(y)] for y in y_raw], dtype=np.int64)
        concept_pools[cid] = (X_pool, y_pool)
        fingerprint_label_pools[cid] = y_raw

    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    fingerprint_labels: dict[tuple[int, int], np.ndarray] = {}
    for k in range(config.K):
        for t in range(config.T):
            cid = int(concept_matrix[k, t])
            rng = np.random.RandomState(config.seed + 10000 + k * config.T + t)
            X_pool, y_pool = concept_pools[cid]
            y_fp_pool = fingerprint_label_pools[cid]
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
            fingerprint_labels[(k, t)] = y_fp_pool[batch_idx].copy()

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
    dataset = DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=gen_config,
        concept_specs=concept_specs,
    )
    setattr(dataset, "fingerprint_labels", fingerprint_labels)
    return dataset


def _infer_dataset_n_classes(ds: DriftDataset) -> int:
    labels: set[int] = set()
    for X, y in ds.data.values():
        del X
        labels.update(int(v) for v in np.unique(y))
    return (max(labels) + 1) if labels else 2


def run_local(ds: DriftDataset, epochs: int, lr: float, seed: int) -> tuple[np.ndarray, float]:
    """Run local-only training on a prepared subset dataset."""
    K, T = ds.config.K, ds.config.T
    nf = ds.data[(0, 0)][0].shape[1]
    nc = _infer_dataset_n_classes(ds)
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
    nc = _infer_dataset_n_classes(ds)
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
    nc = _infer_dataset_n_classes(ds)
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
    global_shared_aggregation: bool | None = None,
    routed_local_training: bool = False,
    fingerprint_source: str = "raw_input",
    expert_update_policy: str | None = None,
    shared_update_policy: str = "always",
    routed_write_top_k: int | None = None,
    routed_write_min_secondary_weight: float = 0.0,
    routed_read_top_k: int | None = None,
    routed_read_temperature: float = 1.0,
    routed_read_only_on_ambiguity: bool = False,
    routed_read_min_entropy: float | None = None,
    routed_read_min_secondary_weight: float = 0.0,
    routed_read_max_primary_gap: float | None = None,
    max_spawn_clusters_per_round: int | None = None,
    novelty_hysteresis_rounds: int | None = None,
    factorized_slot_preserving: bool = False,
    factorized_primary_anchor_alpha: float = 0.25,
    factorized_secondary_anchor_alpha: float = 0.75,
    factorized_primary_consolidation_steps: int = 0,
    factorized_primary_consolidation_mode: str = "full",
) -> tuple[np.ndarray, float, int, int, int]:
    """Run FedProTrack on a prepared subset dataset."""
    result = run_fpt_result(
        ds,
        fed_every,
        epochs,
        lr,
        seed,
        dormant_recall=dormant_recall,
        loss_novelty_threshold=loss_novelty_threshold,
        merge_threshold=merge_threshold,
        max_concepts=max_concepts,
        model_type=model_type,
        hidden_dim=hidden_dim,
        adapter_dim=adapter_dim,
        global_shared_aggregation=global_shared_aggregation,
        routed_local_training=routed_local_training,
        fingerprint_source=fingerprint_source,
        expert_update_policy=expert_update_policy,
        shared_update_policy=shared_update_policy,
        routed_write_top_k=routed_write_top_k,
        routed_write_min_secondary_weight=routed_write_min_secondary_weight,
        routed_read_top_k=routed_read_top_k,
        routed_read_temperature=routed_read_temperature,
        routed_read_only_on_ambiguity=routed_read_only_on_ambiguity,
        routed_read_min_entropy=routed_read_min_entropy,
        routed_read_min_secondary_weight=routed_read_min_secondary_weight,
        routed_read_max_primary_gap=routed_read_max_primary_gap,
        max_spawn_clusters_per_round=max_spawn_clusters_per_round,
        novelty_hysteresis_rounds=novelty_hysteresis_rounds,
        factorized_slot_preserving=factorized_slot_preserving,
        factorized_primary_anchor_alpha=factorized_primary_anchor_alpha,
        factorized_secondary_anchor_alpha=factorized_secondary_anchor_alpha,
        factorized_primary_consolidation_steps=factorized_primary_consolidation_steps,
        factorized_primary_consolidation_mode=factorized_primary_consolidation_mode,
    )
    return (
        result.accuracy_matrix,
        result.total_bytes,
        result.spawned_concepts,
        result.merged_concepts,
        result.active_concepts,
    )


def run_fpt_result(
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
    global_shared_aggregation: bool | None = None,
    routed_local_training: bool = False,
    fingerprint_source: str = "raw_input",
    expert_update_policy: str | None = None,
    shared_update_policy: str = "always",
    routed_write_top_k: int | None = None,
    routed_write_min_secondary_weight: float = 0.0,
    routed_read_top_k: int | None = None,
    routed_read_temperature: float = 1.0,
    routed_read_only_on_ambiguity: bool = False,
    routed_read_min_entropy: float | None = None,
    routed_read_min_secondary_weight: float = 0.0,
    routed_read_max_primary_gap: float | None = None,
    max_spawn_clusters_per_round: int | None = None,
    novelty_hysteresis_rounds: int | None = None,
    factorized_slot_preserving: bool = False,
    factorized_primary_anchor_alpha: float = 0.25,
    factorized_secondary_anchor_alpha: float = 0.75,
    factorized_primary_consolidation_steps: int = 0,
    factorized_primary_consolidation_mode: str = "full",
):
    """Run FedProTrack and return the full result object."""
    from ..posterior.fedprotrack_runner import FedProTrackRunner

    gt = ds.concept_matrix
    n_concepts = int(gt.max()) + 1
    config_overrides: dict[str, object] = {}
    if global_shared_aggregation is not None:
        config_overrides["global_shared_aggregation"] = global_shared_aggregation
    if max_spawn_clusters_per_round is not None:
        config_overrides["max_spawn_clusters_per_round"] = int(max_spawn_clusters_per_round)
    if novelty_hysteresis_rounds is not None:
        config_overrides["novelty_hysteresis_rounds"] = int(novelty_hysteresis_rounds)
    runner = FedProTrackRunner(
        config=make_plan_c_config(
            loss_novelty_threshold=loss_novelty_threshold,
            merge_threshold=merge_threshold,
            max_concepts=max(max_concepts, n_concepts + 2),
            **config_overrides,
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
        routed_local_training=routed_local_training,
        fingerprint_source=fingerprint_source,
        expert_update_policy=expert_update_policy,
        shared_update_policy=shared_update_policy,
        routed_write_top_k=routed_write_top_k,
        routed_write_min_secondary_weight=routed_write_min_secondary_weight,
        routed_read_top_k=routed_read_top_k,
        routed_read_temperature=routed_read_temperature,
        routed_read_only_on_ambiguity=routed_read_only_on_ambiguity,
        routed_read_min_entropy=routed_read_min_entropy,
        routed_read_min_secondary_weight=routed_read_min_secondary_weight,
        routed_read_max_primary_gap=routed_read_max_primary_gap,
        factorized_slot_preserving=factorized_slot_preserving,
        factorized_primary_anchor_alpha=factorized_primary_anchor_alpha,
        factorized_secondary_anchor_alpha=factorized_secondary_anchor_alpha,
        factorized_primary_consolidation_steps=factorized_primary_consolidation_steps,
        factorized_primary_consolidation_mode=factorized_primary_consolidation_mode,
    )
    result = runner.run(ds)
    return result


def make_result_metadata(
    *,
    model_type: str | None,
    fingerprint_source: str | None,
    expert_update_policy: str | None,
    shared_update_policy: str | None,
    global_shared_aggregation: bool | None,
    routed_write_top_k: int | None = None,
    routed_write_min_secondary_weight: float | None = None,
    routed_read_top_k: int | None = None,
    routed_read_temperature: float | None = None,
    routed_read_only_on_ambiguity: bool | None = None,
    routed_read_min_entropy: float | None = None,
    routed_read_min_secondary_weight: float | None = None,
    routed_read_max_primary_gap: float | None = None,
    max_spawn_clusters_per_round: int | None = None,
    novelty_hysteresis_rounds: int | None = None,
    spawned: int | None,
    merged: int | None,
    active: int | None,
    factorized_slot_preserving: bool | None = None,
    factorized_primary_anchor_alpha: float | None = None,
    factorized_secondary_anchor_alpha: float | None = None,
    factorized_primary_consolidation_steps: int | None = None,
    factorized_primary_consolidation_mode: str | None = None,
    assignment_switch_rate: float | None = None,
    avg_clients_per_concept: float | None = None,
    singleton_group_ratio: float | None = None,
    memory_reuse_rate: float | None = None,
    routing_consistency: float | None = None,
    shared_drift_norm: float | None = None,
    expert_update_coverage: float | None = None,
    multi_route_rate: float | None = None,
) -> dict[str, object]:
    """Build the config and routing columns shared by recurrence/overlap tables."""
    return {
        "model_type": model_type,
        "fingerprint_source": fingerprint_source,
        "expert_update_policy": expert_update_policy,
        "shared_update_policy": shared_update_policy,
        "global_shared_aggregation": global_shared_aggregation,
        "routed_write_top_k": routed_write_top_k,
        "routed_write_min_secondary_weight": routed_write_min_secondary_weight,
        "routed_read_top_k": routed_read_top_k,
        "routed_read_temperature": routed_read_temperature,
        "routed_read_only_on_ambiguity": routed_read_only_on_ambiguity,
        "routed_read_min_entropy": routed_read_min_entropy,
        "routed_read_min_secondary_weight": routed_read_min_secondary_weight,
        "routed_read_max_primary_gap": routed_read_max_primary_gap,
        "max_spawn_clusters_per_round": max_spawn_clusters_per_round,
        "novelty_hysteresis_rounds": novelty_hysteresis_rounds,
        "factorized_slot_preserving": factorized_slot_preserving,
        "factorized_primary_anchor_alpha": factorized_primary_anchor_alpha,
        "factorized_secondary_anchor_alpha": factorized_secondary_anchor_alpha,
        "factorized_primary_consolidation_steps": factorized_primary_consolidation_steps,
        "factorized_primary_consolidation_mode": factorized_primary_consolidation_mode,
        "spawned": spawned,
        "merged": merged,
        "active": active,
        "assignment_switch_rate": assignment_switch_rate,
        "avg_clients_per_concept": avg_clients_per_concept,
        "singleton_group_ratio": singleton_group_ratio,
        "memory_reuse_rate": memory_reuse_rate,
        "routing_consistency": routing_consistency,
        "shared_drift_norm": shared_drift_norm,
        "expert_update_coverage": expert_update_coverage,
        "multi_route_rate": multi_route_rate,
    }


def _mean_of_rows(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return float(np.mean(values)) if values else None


def _mean_by_seed(rows: Sequence[Mapping[str, object]], key: str) -> dict[int, float]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        seed = row.get("seed")
        value = row.get(key)
        if seed in (None, "") or value in (None, ""):
            continue
        by_seed.setdefault(int(seed), []).append(float(value))
    return {seed: float(np.mean(vals)) for seed, vals in by_seed.items() if vals}


def summarize_root_cause(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *,
    phase_a_metric: str = "concept_re_id_accuracy",
    final_metric: str = "final",
    downstream_metrics: Sequence[str] | None = ("final", "phase3", "recovery_next_fed"),
    bytes_metric: str = "bytes",
    overlap_field: str = "overlap_ratio",
    overlap_compare: tuple[float, float] | None = None,
) -> list[str]:
    """Summarize the dominant failure mode in at most five lines."""
    if not baseline_rows or not candidate_rows:
        return [
            "Root cause: insufficient rows for comparison.",
            "Need both baseline and candidate rows to diagnose the run.",
        ]
    if downstream_metrics is None:
        downstream_metrics = (final_metric,)

    b_phase_a = _mean_of_rows(baseline_rows, phase_a_metric)
    c_phase_a = _mean_of_rows(candidate_rows, phase_a_metric)
    b_downstream = {m: _mean_of_rows(baseline_rows, m) for m in downstream_metrics}
    c_downstream = {m: _mean_of_rows(candidate_rows, m) for m in downstream_metrics}
    b_bytes = _mean_of_rows(baseline_rows, bytes_metric)
    c_bytes = _mean_of_rows(candidate_rows, bytes_metric)
    b_switch = _mean_of_rows(baseline_rows, "assignment_switch_rate")
    c_switch = _mean_of_rows(candidate_rows, "assignment_switch_rate")
    b_clients = _mean_of_rows(baseline_rows, "avg_clients_per_concept")
    c_clients = _mean_of_rows(candidate_rows, "avg_clients_per_concept")
    b_singleton = _mean_of_rows(baseline_rows, "singleton_group_ratio")
    c_singleton = _mean_of_rows(candidate_rows, "singleton_group_ratio")
    b_reuse = _mean_of_rows(baseline_rows, "memory_reuse_rate")
    c_reuse = _mean_of_rows(candidate_rows, "memory_reuse_rate")
    b_routing = _mean_of_rows(baseline_rows, "routing_consistency")
    c_routing = _mean_of_rows(candidate_rows, "routing_consistency")
    b_shared_drift = _mean_of_rows(baseline_rows, "shared_drift_norm")
    c_shared_drift = _mean_of_rows(candidate_rows, "shared_drift_norm")
    b_update_coverage = _mean_of_rows(baseline_rows, "expert_update_coverage")
    c_update_coverage = _mean_of_rows(candidate_rows, "expert_update_coverage")
    b_multi_route = _mean_of_rows(baseline_rows, "multi_route_rate")
    c_multi_route = _mean_of_rows(candidate_rows, "multi_route_rate")

    baseline_final_by_seed = _mean_by_seed(baseline_rows, final_metric)
    candidate_final_by_seed = _mean_by_seed(candidate_rows, final_metric)
    common_seeds = sorted(set(baseline_final_by_seed) & set(candidate_final_by_seed))
    seed_deltas = [
        candidate_final_by_seed[seed] - baseline_final_by_seed[seed]
        for seed in common_seeds
    ]
    candidate_spawn_by_seed = _mean_by_seed(candidate_rows, "spawned")
    candidate_active_by_seed = _mean_by_seed(candidate_rows, "active")
    candidate_recovery_by_seed = _mean_by_seed(
        candidate_rows,
        "recovery_next_fed" if "recovery_next_fed" in downstream_metrics else final_metric,
    )

    bytes_ratio = None
    if b_bytes not in (None, 0.0) and c_bytes is not None:
        bytes_ratio = c_bytes / b_bytes

    overlap_gap = None
    if overlap_compare is not None:
        paired_gaps: list[float] = []
        for overlap_value in overlap_compare:
            baseline_overlap_rows = [
                row for row in baseline_rows
                if row.get(overlap_field) not in (None, "")
                and float(row[overlap_field]) == float(overlap_value)
            ]
            candidate_overlap_rows = [
                row for row in candidate_rows
                if row.get(overlap_field) not in (None, "")
                and float(row[overlap_field]) == float(overlap_value)
            ]
            if not baseline_overlap_rows or not candidate_overlap_rows:
                continue
            baseline_overlap_mean = _mean_of_rows(baseline_overlap_rows, final_metric)
            candidate_overlap_mean = _mean_of_rows(candidate_overlap_rows, final_metric)
            if baseline_overlap_mean is not None and candidate_overlap_mean is not None:
                paired_gaps.append(candidate_overlap_mean - baseline_overlap_mean)
        if paired_gaps:
            overlap_gap = min(paired_gaps)

    phase_a_improved = (
        b_phase_a is not None and c_phase_a is not None and c_phase_a > b_phase_a + 1e-6
    )
    downstream_pairs = [
        (metric, b_downstream[metric], c_downstream[metric])
        for metric in downstream_metrics
        if b_downstream[metric] is not None and c_downstream[metric] is not None
    ]
    downstream_improved = bool(downstream_pairs) and all(
        candidate_value > baseline_value + 1e-6
        for _, baseline_value, candidate_value in downstream_pairs
    )
    final_delta = None
    if b_downstream.get(final_metric) is not None and c_downstream.get(final_metric) is not None:
        final_delta = c_downstream[final_metric] - b_downstream[final_metric]

    overlap_switch_worse = (
        b_switch is not None and c_switch is not None and c_switch > b_switch + 1e-6
    )
    overlap_singleton_worse = (
        b_singleton is not None
        and c_singleton is not None
        and c_singleton > b_singleton + 1e-6
    )

    routing_instability = False
    if seed_deltas and max(seed_deltas) > 0.0 and float(np.mean(seed_deltas)) <= 0.0:
        spawn_span = (
            max(candidate_spawn_by_seed.values()) - min(candidate_spawn_by_seed.values())
            if candidate_spawn_by_seed
            else 0.0
        )
        active_span = (
            max(candidate_active_by_seed.values()) - min(candidate_active_by_seed.values())
            if candidate_active_by_seed
            else 0.0
        )
        recovery_span = (
            max(candidate_recovery_by_seed.values()) - min(candidate_recovery_by_seed.values())
            if candidate_recovery_by_seed
            else 0.0
        )
        routing_instability = bool(spawn_span > 0.0 or active_span > 0.0 or recovery_span > 0.01)

    read_write_mismatch = False
    if phase_a_improved and not downstream_improved:
        drift_not_better = (
            c_shared_drift is not None
            and (b_shared_drift is None or c_shared_drift > b_shared_drift + 1e-6)
        )
        coverage_not_better = (
            c_update_coverage is not None
            and (b_update_coverage is None or c_update_coverage <= b_update_coverage + 1e-6)
        )
        read_write_mismatch = drift_not_better or coverage_not_better

    if overlap_gap is not None and overlap_gap < -0.015:
        if overlap_switch_worse or overlap_singleton_worse:
            primary = "mechanism overfits recurrence"
        else:
            primary = "overlap veto"
    elif bytes_ratio is not None and bytes_ratio > 2.5:
        primary = "cost-inefficient improvement"
    elif routing_instability:
        primary = "routing instability"
    elif (not phase_a_improved) and (final_delta is None or final_delta <= 1e-6):
        primary = "问题仍在 Phase A 表征"
    elif read_write_mismatch:
        primary = "read/write mismatch + shared drift"
    elif phase_a_improved and not downstream_improved:
        primary = "Phase A improved, but downstream did not track"
    elif seed_deltas and max(seed_deltas) > 0.0 and float(np.mean(seed_deltas)) <= 0.0:
        primary = "single-seed gain did not survive multi-seed averaging"
    else:
        primary = "no hard regression detected"

    lines: list[str] = []
    lines.append(f"Root cause: {primary}.")

    if primary == "问题仍在 Phase A 表征":
        evidence_parts: list[str] = []
        if b_phase_a is not None and c_phase_a is not None:
            evidence_parts.append(f"{phase_a_metric}={b_phase_a:.4f}->{c_phase_a:.4f}")
        if b_singleton is not None and c_singleton is not None:
            evidence_parts.append(f"singleton={b_singleton:.4f}->{c_singleton:.4f}")
        if b_switch is not None and c_switch is not None:
            evidence_parts.append(f"switch={b_switch:.4f}->{c_switch:.4f}")
        if b_clients is not None and c_clients is not None:
            evidence_parts.append(f"clients/concept={b_clients:.4f}->{c_clients:.4f}")
        if evidence_parts:
            lines.append("Evidence: " + "; ".join(evidence_parts) + ".")
        lines.append("Strongest signal: Phase A clustering metrics stayed flat while downstream metrics did not improve.")
        lines.append("Next experiment: keep local training fixed and run the `model_embed` contrast.")
    elif primary in {"read/write mismatch + shared drift", "Phase A improved, but downstream did not track"}:
        evidence_parts = []
        if b_phase_a is not None and c_phase_a is not None:
            evidence_parts.append(f"{phase_a_metric}={b_phase_a:.4f}->{c_phase_a:.4f}")
        if b_shared_drift is not None and c_shared_drift is not None:
            evidence_parts.append(f"shared_drift={b_shared_drift:.4f}->{c_shared_drift:.4f}")
        elif c_shared_drift is not None:
            evidence_parts.append(f"shared_drift={c_shared_drift:.4f}")
        if b_update_coverage is not None and c_update_coverage is not None:
            evidence_parts.append(f"expert_cov={b_update_coverage:.4f}->{c_update_coverage:.4f}")
        elif c_update_coverage is not None:
            evidence_parts.append(f"expert_cov={c_update_coverage:.4f}")
        if b_multi_route is not None and c_multi_route is not None:
            evidence_parts.append(f"multi_route={b_multi_route:.4f}->{c_multi_route:.4f}")
        elif c_multi_route is not None:
            evidence_parts.append(f"multi_route={c_multi_route:.4f}")
        if b_reuse is not None and c_reuse is not None:
            evidence_parts.append(f"memory_reuse={b_reuse:.4f}->{c_reuse:.4f}")
        if evidence_parts:
            lines.append("Evidence: " + "; ".join(evidence_parts) + ".")
        lines.append("Strongest signal: Phase A moved, but local writes or shared updates still fail to preserve the gain.")
        lines.append("Next experiment: prioritize `posterior_weighted + freeze_on_multiroute` before any threshold sweep.")
    elif primary in {"overlap veto", "mechanism overfits recurrence"}:
        evidence_parts = []
        if overlap_gap is not None:
            evidence_parts.append(f"{final_metric}_gap={overlap_gap:.4f}")
        if b_switch is not None and c_switch is not None:
            evidence_parts.append(f"switch={b_switch:.4f}->{c_switch:.4f}")
        if b_singleton is not None and c_singleton is not None:
            evidence_parts.append(f"singleton={b_singleton:.4f}->{c_singleton:.4f}")
        if b_routing is not None and c_routing is not None:
            evidence_parts.append(f"routing={b_routing:.4f}->{c_routing:.4f}")
        if evidence_parts:
            lines.append("Evidence: " + "; ".join(evidence_parts) + ".")
        lines.append("Strongest signal: the recurrence gain does not survive overlap and should not be promoted as a default.")
        lines.append("Next experiment: veto this mechanism globally and escalate to Stage 2 if recurrence still matters.")
    elif primary in {"routing instability", "single-seed gain did not survive multi-seed averaging"}:
        evidence_parts = []
        if seed_deltas:
            evidence_parts.append(f"seed_deltas_{final_metric}={min(seed_deltas):.4f}..{max(seed_deltas):.4f}")
        if candidate_spawn_by_seed:
            values = list(candidate_spawn_by_seed.values())
            evidence_parts.append(f"spawned_span={max(values) - min(values):.4f}")
        if candidate_active_by_seed:
            values = list(candidate_active_by_seed.values())
            evidence_parts.append(f"active_span={max(values) - min(values):.4f}")
        if candidate_recovery_by_seed:
            values = list(candidate_recovery_by_seed.values())
            evidence_parts.append(f"recovery_span={max(values) - min(values):.4f}")
        if evidence_parts:
            lines.append("Evidence: " + "; ".join(evidence_parts) + ".")
        lines.append("Strongest signal: the candidate is high-variance across seeds, so the apparent gain is not stable.")
        lines.append("Next experiment: stop threshold sweeps and move to the next architecture gate.")
    else:
        evidence_parts = []
        if b_phase_a is not None and c_phase_a is not None:
            evidence_parts.append(f"{phase_a_metric}={b_phase_a:.4f}->{c_phase_a:.4f}")
        if downstream_pairs:
            evidence_parts.append(
                ", ".join(
                    f"{metric}={baseline_value:.4f}->{candidate_value:.4f}"
                    for metric, baseline_value, candidate_value in downstream_pairs
                )
            )
        if evidence_parts:
            lines.append("Evidence: " + "; ".join(evidence_parts) + ".")
        if primary == "cost-inefficient improvement":
            lines.append("Strongest signal: the candidate buys accuracy with too much extra communication.")
            lines.append("Next experiment: reject this variant unless the bytes increase falls under the budget cap.")
        else:
            lines.append("Strongest signal: no single failure mode dominates the comparison.")
            lines.append("Next experiment: keep the current winner and only run the narrow sweep if the gate still passes.")

    if bytes_ratio is not None and primary != "cost-inefficient improvement" and len(lines) < 5:
        lines.append(f"Bytes ratio candidate/baseline={bytes_ratio:.2f}x.")
    if len(lines) < 5:
        lines.append("Gate: stop promoting variants that fail the selected root-cause check.")

    return lines[:5]
