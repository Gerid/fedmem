from __future__ import annotations

"""Diagnose FedProTrack re-ID failure on disjoint label split.

Runs FedProTrack on disjoint dataset and dumps per-round diagnostics:
spawned/merged/pruned counts, active concepts, predicted vs true concept
matrices, and per-round posterior snapshots.
"""

import json
import sys
from pathlib import Path

import numpy as np

from fedprotrack.metrics.concept_metrics import concept_re_id_accuracy
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def main() -> None:
    seed = 42
    for label_split in ["none", "disjoint"]:
        print(f"\n{'='*70}")
        print(f"label_split={label_split}, seed={seed}")
        print(f"{'='*70}")

        cfg = CIFAR100RecurrenceConfig(
            K=4, T=12, n_samples=200, rho=3.0, alpha=0.75, delta=0.9,
            n_features=64, samples_per_coarse_class=30, batch_size=128,
            n_workers=0, data_root=".cifar100_cache",
            feature_cache_dir=".feature_cache", feature_seed=2718,
            seed=seed, label_split=label_split,
        )
        prepare_cifar100_recurrence_feature_cache(cfg)
        ds = generate_cifar100_recurrence_dataset(cfg)

        n_true_concepts = int(ds.concept_matrix.max()) + 1
        print(f"True concept matrix:\n{ds.concept_matrix}")
        print(f"n_true_concepts={n_true_concepts}")

        # Print label distribution per concept
        for c_id in range(n_true_concepts):
            cells = list(zip(*np.where(ds.concept_matrix == c_id)))
            if cells:
                k, t = cells[0]
                _, y = ds.data[(k, t)]
                unique_labels = np.unique(y)
                print(f"  Concept {c_id}: {len(unique_labels)} classes: {unique_labels}")

        result = FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0, kappa=0.7, novelty_threshold=0.25,
                loss_novelty_threshold=0.15, sticky_dampening=1.5,
                sticky_posterior_gate=0.35, merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, n_true_concepts + 3),
                merge_every=2, shrink_every=6,
            ),
            federation_every=2,
            detector_name="ADWIN",
            seed=seed,
            lr=0.05,
            n_epochs=5,
            soft_aggregation=True,
            blend_alpha=0.0,
        ).run(ds)

        print(f"\nPredicted concept matrix:\n{result.predicted_concept_matrix}")
        print(f"spawned={result.spawned_concepts}, merged={result.merged_concepts}, "
              f"pruned={result.pruned_concepts}, active={result.active_concepts}")

        # Per-client re-ID
        re_id, per_client, per_step = concept_re_id_accuracy(
            ds.concept_matrix, result.predicted_concept_matrix
        )
        print(f"Overall re-ID: {re_id:.3f}")
        print(f"Per-client re-ID: {np.round(per_client, 3)}")
        print(f"Per-step re-ID: {np.round(per_step, 3)}")

        # Soft assignments analysis
        if result.soft_assignments is not None:
            sa = result.soft_assignments  # (K, T, C)
            print(f"\nSoft assignments shape: {sa.shape}")
            # For each federation step, show the MAP concept and its probability
            for t in range(ds.config.T):
                if sa[:, t, :].sum() < 0.01:
                    continue  # non-federation step
                print(f"  t={t}:")
                for k in range(ds.config.K):
                    p = sa[k, t, :]
                    if p.sum() < 0.01:
                        continue
                    map_c = int(np.argmax(p))
                    map_p = p[map_c]
                    true_c = int(ds.concept_matrix[k, t])
                    pred_c = int(result.predicted_concept_matrix[k, t])
                    status = "OK" if pred_c == true_c else "WRONG"  # Note: uses Hungarian alignment
                    print(f"    k={k}: true={true_c}, pred={pred_c}, "
                          f"MAP_concept={map_c} (p={map_p:.3f}), "
                          f"full_dist={np.round(p[:6], 3)}")

        # Phase A diagnostics
        if result.phase_a_round_diagnostics:
            print(f"\nPhase A round diagnostics ({len(result.phase_a_round_diagnostics)} rounds):")
            for rd in result.phase_a_round_diagnostics:
                t = rd.get("t", "?")
                spawned = rd.get("spawned", 0)
                merged = rd.get("merged", 0)
                pruned = rd.get("pruned", 0)
                n_active = rd.get("n_active", "?")
                print(f"  t={t}: spawned={spawned}, merged={merged}, "
                      f"pruned={pruned}, active={n_active}")

        print(f"\nAccuracy matrix:\n{np.round(result.accuracy_matrix, 3)}")
        print(f"Mean accuracy: {result.mean_accuracy:.3f}")
        print(f"Final accuracy: {result.final_accuracy:.3f}")


if __name__ == "__main__":
    main()
