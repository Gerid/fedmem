from __future__ import annotations

"""Diagnostic: sweep blend_alpha to test global-model fallback effect."""

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

print(f"{'blend_alpha':<15} {'final_acc':<12} {'auc':<12} {'re_id':<12}")
print("-" * 55)

for blend in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    runner = FedProTrackRunner(
        config=TwoPhaseConfig(
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
        federation_every=2,
        detector_name="ADWIN",
        seed=42, lr=0.05, n_epochs=5,
        soft_aggregation=True,
        blend_alpha=blend,
        similarity_calibration=True,
    )
    result = runner.run(dataset)
    acc = result.accuracy_matrix[:, -1].mean()
    auc = result.accuracy_matrix.mean()

    from fedprotrack.metrics.concept_metrics import concept_re_id_accuracy
    reid, _, _ = concept_re_id_accuracy(dataset.concept_matrix, result.predicted_concept_matrix)

    print(f"{blend:<15.1f} {acc:<12.4f} {auc:<12.4f} {reid:<12.4f}")
