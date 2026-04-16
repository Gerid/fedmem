from __future__ import annotations

"""Validation script for min-group-size fallback and cross-time EMA features.

Runs FPT with four configurations on CIFAR-100 disjoint (K=10, T=30, 1 seed)
and compares final accuracy and re-ID scores:
  1. FPT-base (defaults: min_agg_group_size=1, cross_time_ema=1.0)
  2. FPT+fallback (min_agg_group_size=3)
  3. FPT+crosstime (cross_time_ema=0.5)
  4. FPT+both (min_agg_group_size=3, cross_time_ema=0.5)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# GPU_THRESHOLD=0 removed: linear models stay on CPU (faster for <8192 params)
os.environ.setdefault("PYTHONUNBUFFERED", "1")

from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior import FedProTrackRunner, TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


SEED = 42
K = 10
T = 30
N_EPOCHS = 5
LR = 0.1
RESULTS_DIR = Path("tmp/fallback_validation")


def _make_config(
    n_concepts: int,
    *,
    min_agg_group_size: int = 1,
    cross_time_ema: float = 1.0,
) -> tuple[TwoPhaseConfig, dict]:
    """Build TwoPhaseConfig and extra runner kwargs."""
    config = TwoPhaseConfig(
        max_concepts=max(6, n_concepts + 2),
        shrink_every=6,
        loss_novelty_threshold=0.02,
        sticky_dampening=1.0,
        min_agg_group_size=min_agg_group_size,
        cross_time_ema=cross_time_ema,
    )
    runner_kwargs = dict(
        config=config,
        seed=SEED,
        soft_aggregation=True,
        lr=LR,
        n_epochs=N_EPOCHS,
        min_agg_group_size=min_agg_group_size,
        cross_time_ema=cross_time_ema,
    )
    return config, runner_kwargs


def _run_variant(
    name: str,
    dataset,
    ground_truth: np.ndarray,
    *,
    min_agg_group_size: int = 1,
    cross_time_ema: float = 1.0,
) -> dict:
    """Run one FPT variant and return metrics dict."""
    n_concepts = int(ground_truth.max()) + 1
    _, runner_kwargs = _make_config(
        n_concepts,
        min_agg_group_size=min_agg_group_size,
        cross_time_ema=cross_time_ema,
    )
    runner = FedProTrackRunner(**runner_kwargs)

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"  min_agg_group_size={min_agg_group_size}, cross_time_ema={cross_time_ema}")
    print(f"{'='*60}")
    t0 = time.time()
    result = runner.run(dataset)
    elapsed = time.time() - t0

    log = result.to_experiment_log()
    metrics = compute_all_metrics(log)

    final_acc = float(np.mean(result.accuracy_matrix[:, -1]))
    mean_acc = float(np.mean(result.accuracy_matrix))
    re_id = metrics.concept_re_id_accuracy if metrics.concept_re_id_accuracy is not None else float("nan")
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
        "cross_time_ema": cross_time_ema,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build dataset
    print("Building CIFAR-100 disjoint dataset (K=10, T=30)...")
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

    # Define variants
    variants = [
        ("FPT-base", {"min_agg_group_size": 1, "cross_time_ema": 1.0}),
        ("FPT+fallback", {"min_agg_group_size": 3, "cross_time_ema": 1.0}),
        ("FPT+crosstime", {"min_agg_group_size": 1, "cross_time_ema": 0.5}),
        ("FPT+both", {"min_agg_group_size": 3, "cross_time_ema": 0.5}),
    ]

    results = []
    for name, kwargs in variants:
        result = _run_variant(name, dataset, ground_truth, **kwargs)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<20} {'Final Acc':>10} {'Mean Acc':>10} {'Re-ID':>10} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['name']:<20} {r['final_acc']:>10.4f} {r['mean_acc']:>10.4f} "
            f"{r['re_id']:>10.4f} {r['elapsed']:>7.1f}s"
        )

    # Save
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
