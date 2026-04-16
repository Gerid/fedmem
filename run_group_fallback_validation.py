from __future__ import annotations

"""Validation script for min_agg_group_size fallback on CIFAR-100 disjoint.

Runs FedProTrack with min_agg_group_size = 1, 3, 5 on CIFAR-100 disjoint
(K=10, T=20, 1 seed) and saves results to tmp/group_fallback_validation/.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# GPU_THRESHOLD=0 removed: linear models stay on CPU (faster for <8192 params)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fedprotrack.metrics import compute_all_metrics
from fedprotrack.posterior import FedProTrackRunner, TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


SEED = 42
K = 10
T = 20
N_EPOCHS = 5
LR = 0.1
RESULTS_DIR = Path("tmp/group_fallback_validation")


def _run_variant(
    name: str,
    dataset,
    ground_truth: np.ndarray,
    *,
    min_agg_group_size: int = 1,
) -> dict:
    """Run one FPT variant and return metrics dict."""
    n_concepts = int(ground_truth.max()) + 1
    config = TwoPhaseConfig(
        max_concepts=max(6, n_concepts + 2),
        shrink_every=6,
        loss_novelty_threshold=0.02,
        sticky_dampening=1.0,
        min_agg_group_size=min_agg_group_size,
    )
    runner = FedProTrackRunner(
        config=config,
        seed=SEED,
        soft_aggregation=True,
        lr=LR,
        n_epochs=N_EPOCHS,
        min_agg_group_size=min_agg_group_size,
    )

    print(f"\n{'='*60}")
    print(f"Running: {name} (min_agg_group_size={min_agg_group_size})")
    print(f"{'='*60}")
    t0 = time.time()
    result = runner.run(dataset)
    elapsed = time.time() - t0

    log = result.to_experiment_log()
    metrics = compute_all_metrics(log)

    final_acc = float(np.mean(result.accuracy_matrix[:, -1]))
    mean_acc = float(np.mean(result.accuracy_matrix))
    re_id = (
        metrics.concept_re_id_accuracy
        if metrics.concept_re_id_accuracy is not None
        else float("nan")
    )
    total_bytes = result.total_bytes

    print(f"  Final acc: {final_acc:.4f}")
    print(f"  Mean acc:  {mean_acc:.4f}")
    print(f"  Re-ID:     {re_id:.4f}")
    print(f"  Bytes:     {total_bytes:.0f}")
    print(f"  Time:      {elapsed:.1f}s")

    return {
        "name": name,
        "final_acc": final_acc,
        "mean_acc": mean_acc,
        "re_id": re_id,
        "total_bytes": total_bytes,
        "elapsed": elapsed,
        "min_agg_group_size": min_agg_group_size,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building CIFAR-100 disjoint dataset (K={K}, T={T})...")
    ds_cfg = CIFAR100RecurrenceConfig(
        K=K,
        T=T,
        n_samples=400,
        rho=3.0,
        alpha=0.75,
        delta=0.85,
        n_features=128,
        label_split="disjoint",
        seed=SEED,
    )
    prepare_cifar100_recurrence_feature_cache(ds_cfg)
    dataset = generate_cifar100_recurrence_dataset(ds_cfg)
    ground_truth = dataset.concept_matrix

    variants = [
        ("FPT-base", 1),
        ("FPT+fallback3", 3),
        ("FPT+fallback5", 5),
    ]

    results = []
    for name, mags in variants:
        result = _run_variant(
            name, dataset, ground_truth, min_agg_group_size=mags
        )
        results.append(result)

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: min_agg_group_size fallback validation")
    print(f"Dataset: CIFAR-100 disjoint, K={K}, T={T}, seed={SEED}")
    print("=" * 70)
    print(
        f"{'Variant':<20} {'min_gs':>6} {'Final Acc':>10} "
        f"{'Mean Acc':>10} {'Re-ID':>10} {'Time':>8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<20} {r['min_agg_group_size']:>6} "
            f"{r['final_acc']:>10.4f} {r['mean_acc']:>10.4f} "
            f"{r['re_id']:>10.4f} {r['elapsed']:>7.1f}s"
        )

    # Deltas vs baseline
    base = results[0]
    print("\nDeltas vs FPT-base:")
    for r in results[1:]:
        delta_acc = r["final_acc"] - base["final_acc"]
        delta_mean = r["mean_acc"] - base["mean_acc"]
        print(
            f"  {r['name']}: final_acc {delta_acc:+.4f}, "
            f"mean_acc {delta_mean:+.4f}"
        )

    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
