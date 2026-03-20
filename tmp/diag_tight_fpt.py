from __future__ import annotations

"""Quick diagnostic: FPT with tight concept cap vs default."""

import numpy as np

from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

cfg = CIFAR100RecurrenceConfig(
    K=10, T=30, n_samples=800, rho=7.5, alpha=0.5, delta=0.85,
    n_features=128, samples_per_coarse_class=120, batch_size=256,
    n_workers=0, seed=42, label_split="disjoint", min_group_size=2,
)
prepare_cifar100_recurrence_feature_cache(cfg)
dataset = generate_cifar100_recurrence_dataset(cfg)

configs = {
    "original (max6, merge0.8)": TwoPhaseConfig(
        omega=2.0, kappa=0.7,
        novelty_threshold=0.25,
        loss_novelty_threshold=0.02,
        sticky_dampening=1.0,
        sticky_posterior_gate=0.35,
        merge_threshold=0.80,
        min_count=5.0,
        max_concepts=6,
        merge_every=2,
        shrink_every=6,
    ),
    "tight (max4, merge0.5)": TwoPhaseConfig(
        omega=2.0, kappa=0.7,
        novelty_threshold=0.5,
        loss_novelty_threshold=0.05,
        sticky_dampening=1.0,
        sticky_posterior_gate=0.35,
        merge_threshold=0.5,
        min_count=5.0,
        max_concepts=4,
        merge_every=1,
        shrink_every=3,
    ),
    "very tight (max4, novelty0.8)": TwoPhaseConfig(
        omega=2.0, kappa=0.7,
        novelty_threshold=0.8,
        loss_novelty_threshold=0.10,
        sticky_dampening=1.0,
        sticky_posterior_gate=0.35,
        merge_threshold=0.4,
        min_count=5.0,
        max_concepts=4,
        merge_every=1,
        shrink_every=2,
    ),
}

for label, tpc in configs.items():
    runner = FedProTrackRunner(
        config=tpc,
        federation_every=2,
        detector_name="ADWIN",
        seed=42, lr=0.05, n_epochs=5,
        soft_aggregation=True, blend_alpha=0.0,
        similarity_calibration=True,
    )
    result = runner.run(dataset)
    acc = result.accuracy_matrix[:, -1].mean()
    auc = result.accuracy_matrix.mean()
    spawned = getattr(result, "spawned_concepts", 0)
    active = getattr(result, "active_concepts", 0)
    n_pred = int(result.predicted_concept_matrix.max()) + 1
    print(
        f"{label:35s}  acc={acc:.3f}  auc={auc:.3f}  "
        f"spawned={spawned}  active={active}  pred_concepts={n_pred}"
    )
