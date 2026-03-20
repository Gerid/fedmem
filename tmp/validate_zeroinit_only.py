from __future__ import annotations

"""Quick check: adapter with old config (ep=5, fed=2) after zero-init fix.
Just 2 seeds to quantify zero-init improvement alone."""

import numpy as np

from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

K = 4
T = 20
N_SAMPLES = 400
N_FEATURES = 64


def _build_cfg(dataset):
    return TwoPhaseConfig(
        omega=2.0, kappa=0.7, novelty_threshold=0.25,
        loss_novelty_threshold=0.15, sticky_dampening=1.5,
        sticky_posterior_gate=0.35, merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, int(dataset.concept_matrix.max()) + 3),
        merge_every=2, shrink_every=6,
    )


def _common_kwargs():
    return {
        "auto_scale": False, "update_ot_weight": 0.0, "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0, "labelwise_proto_dim": 4,
        "prototype_alignment_early_rounds": 0, "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0, "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0, "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3, "prototype_subgroup_similarity_gate": 0.8,
    }


cache_cfg = CIFAR100RecurrenceConfig(
    K=K, T=T, n_samples=N_SAMPLES, seed=42,
    n_features=N_FEATURES, samples_per_coarse_class=30,
)
prepare_cifar100_recurrence_feature_cache(cache_cfg)

for seed in [42, 43]:
    dataset_cfg = CIFAR100RecurrenceConfig(
        K=K, T=T, n_samples=N_SAMPLES, rho=2.0, alpha=0.75, delta=0.9,
        n_features=N_FEATURES, samples_per_coarse_class=30,
        batch_size=128, n_workers=0, seed=seed,
    )
    dataset = generate_cifar100_recurrence_dataset(dataset_cfg)
    cfg = _build_cfg(dataset)
    result = FedProTrackRunner(
        config=cfg, federation_every=2, detector_name="ADWIN", seed=seed,
        lr=0.05, n_epochs=5, soft_aggregation=True, blend_alpha=0.0,
        model_type="feature_adapter", hidden_dim=64, adapter_dim=16,
        similarity_calibration=False, model_signature_weight=0.0,
        model_signature_dim=8, prototype_alignment_mix=0.0,
        **_common_kwargs(),
    ).run(dataset)
    log = result.to_experiment_log()
    metrics = compute_all_metrics(log, identity_capable=True)
    print(f"seed={seed}  adapter ep=5 fed=2 (zero-init): "
          f"final_acc={metrics.final_accuracy:.4f}  "
          f"re-ID={metrics.concept_re_id_accuracy:.4f}")
