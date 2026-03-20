from __future__ import annotations

"""Sweep novelty/merge thresholds to find settings that improve re-ID on disjoint."""

import sys
import numpy as np

from fedprotrack.metrics.concept_metrics import concept_re_id_accuracy
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def run_one(label_split, seed, loss_novelty, merge_thresh, novelty_thresh, sticky_damp, max_concepts=None):
    cfg = CIFAR100RecurrenceConfig(
        K=4, T=12, n_samples=200, rho=3.0, alpha=0.75, delta=0.9,
        n_features=64, samples_per_coarse_class=30, batch_size=128,
        n_workers=0, data_root=".cifar100_cache",
        feature_cache_dir=".feature_cache", feature_seed=2718,
        seed=seed, label_split=label_split,
    )
    prepare_cifar100_recurrence_feature_cache(cfg)
    ds = generate_cifar100_recurrence_dataset(cfg)
    n_true = int(ds.concept_matrix.max()) + 1
    mc = max_concepts if max_concepts is not None else max(6, n_true + 3)

    result = FedProTrackRunner(
        config=TwoPhaseConfig(
            omega=2.0, kappa=0.7,
            novelty_threshold=novelty_thresh,
            loss_novelty_threshold=loss_novelty,
            sticky_dampening=sticky_damp,
            sticky_posterior_gate=0.35,
            merge_threshold=merge_thresh,
            min_count=5.0,
            max_concepts=mc,
            merge_every=2, shrink_every=6,
        ),
        federation_every=2,
        detector_name="ADWIN",
        seed=seed,
        lr=0.05, n_epochs=5,
        soft_aggregation=True, blend_alpha=0.0,
    ).run(ds)

    re_id, _, _ = concept_re_id_accuracy(
        ds.concept_matrix, result.predicted_concept_matrix
    )
    return {
        "re_id": re_id,
        "acc": result.final_accuracy,
        "spawned": result.spawned_concepts,
        "merged": result.merged_concepts,
        "active": result.active_concepts,
    }


def main():
    seeds = [42, 43, 44]
    label_split = "disjoint"

    configs = [
        # (loss_novelty, merge_thresh, novelty_thresh, sticky_damp, max_concepts, label)
        (0.15, 0.85, 0.25, 1.5, None, "baseline(max=7)"),
        (0.15, 0.85, 0.25, 1.5, 4, "max_concepts=4"),
        (0.15, 0.85, 0.25, 1.5, 5, "max_concepts=5"),
        (0.15, 0.85, 0.25, 1.5, 6, "max_concepts=6"),
        (0.15, 0.50, 0.25, 1.5, None, "aggressive_merge"),
        (0.15, 0.50, 0.25, 1.5, 5, "agg_merge+max5"),
        (0.15, 0.50, 0.25, 1.5, 4, "agg_merge+max4"),
        (0.35, 0.50, 0.25, 2.5, 5, "combo_agg+max5"),
        (0.35, 0.50, 0.25, 2.5, 4, "combo_agg+max4"),
        (0.15, 0.85, 0.25, 1.5, 3, "max_concepts=3"),
    ]

    print(f"{'Label':<25} {'ReID':>6} {'Acc':>6} {'Spawn':>6} {'Merge':>6} {'Active':>6}")
    print("-" * 65)

    for loss_nov, merge_t, nov_t, sticky, mc, label in configs:
        reids, accs, spawns, merges, actives = [], [], [], [], []
        for seed in seeds:
            r = run_one(label_split, seed, loss_nov, merge_t, nov_t, sticky, mc)
            reids.append(r["re_id"])
            accs.append(r["acc"])
            spawns.append(r["spawned"])
            merges.append(r["merged"])
            actives.append(r["active"])
        print(f"{label:<25} {np.mean(reids):>6.3f} {np.mean(accs):>6.3f} "
              f"{np.mean(spawns):>6.1f} {np.mean(merges):>6.1f} {np.mean(actives):>6.1f}")


if __name__ == "__main__":
    main()
