from __future__ import annotations

"""Debug adapter in federated loop: trace accuracy per step."""

import numpy as np

from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

SEED = 42
K = 4
T = 10
N_SAMPLES = 400
N_FEATURES = 64
FEDERATION_EVERY = 2


def _build_cfg(dataset) -> TwoPhaseConfig:
    return TwoPhaseConfig(
        omega=2.0, kappa=0.7, novelty_threshold=0.25,
        loss_novelty_threshold=0.15, sticky_dampening=1.5,
        sticky_posterior_gate=0.35, merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, int(dataset.concept_matrix.max()) + 3),
        merge_every=2, shrink_every=6,
    )


def _common_kwargs() -> dict:
    return {
        "auto_scale": False, "update_ot_weight": 0.0, "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0, "labelwise_proto_dim": 4,
        "prototype_alignment_early_rounds": 0, "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0, "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0, "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3, "prototype_subgroup_similarity_gate": 0.8,
    }


cache_cfg = CIFAR100RecurrenceConfig(
    K=K, T=T, n_samples=N_SAMPLES, seed=SEED,
    n_features=N_FEATURES, samples_per_coarse_class=30,
)
prepare_cifar100_recurrence_feature_cache(cache_cfg)
dataset_cfg = CIFAR100RecurrenceConfig(
    K=K, T=T, n_samples=N_SAMPLES, rho=2.0, alpha=0.75, delta=0.9,
    n_features=N_FEATURES, samples_per_coarse_class=30,
    batch_size=128, n_workers=0, seed=SEED,
)
dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

configs = [
    ("linear lr=0.05 ep=5", "linear", 0.05, 5),
    ("linear lr=0.05 ep=20", "linear", 0.05, 20),
    ("adapter lr=0.05 ep=5", "feature_adapter", 0.05, 5),
    ("adapter lr=0.05 ep=20", "feature_adapter", 0.05, 20),
    ("adapter lr=0.01 ep=20", "feature_adapter", 0.01, 20),
]

for name, mtype, lr, n_epochs in configs:
    cfg = _build_cfg(dataset)
    result = FedProTrackRunner(
        config=cfg, federation_every=FEDERATION_EVERY,
        detector_name="ADWIN", seed=SEED,
        lr=lr, n_epochs=n_epochs,
        soft_aggregation=True, blend_alpha=0.0,
        model_type=mtype, hidden_dim=64, adapter_dim=16,
        similarity_calibration=False, model_signature_weight=0.0,
        model_signature_dim=8, prototype_alignment_mix=0.0,
        **_common_kwargs(),
    ).run(dataset)
    # Per-step mean accuracy
    mean_per_step = result.accuracy_matrix.mean(axis=0)
    final_acc = result.final_accuracy
    mean_acc = result.mean_accuracy
    print(f"{name:30s}  final={final_acc:.4f}  mean={mean_acc:.4f}  "
          f"bytes={result.total_bytes:.0f}")
    print(f"  per-step: {' '.join(f'{a:.3f}' for a in mean_per_step)}")
