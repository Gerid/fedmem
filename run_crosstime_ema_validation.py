from __future__ import annotations

"""Validate cross-time EMA concept model aggregation on CIFAR-100 disjoint.

Compares FedProTrack with different cross_time_ema values:
  - 1.0 = no blending (current behavior, 100% current-round aggregate)
  - 0.7 = 70% current + 30% stored historical
  - 0.5 = 50% current + 50% stored historical
  - 0.3 = 30% current + 70% stored historical

Convention: cross_time_ema is the weight on the CURRENT round aggregate.
  new_model = cross_time_ema * current_aggregate + (1 - cross_time_ema) * stored_model

Results saved to tmp/crosstime_validation/.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Force GPU preference
os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fedprotrack.metrics import compute_all_metrics
from fedprotrack.posterior import FedProTrackRunner, TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

RESULTS_DIR = Path(__file__).parent / "tmp" / "crosstime_validation"

# Dataset config: K=10, T=20, disjoint labels, 4 concepts
# rho controls number of concepts (low rho -> more concepts)
DATASET_CFG = {
    "K": 10,
    "T": 20,
    "n_samples": 400,
    "rho": 3.0,
    "alpha": 0.75,
    "delta": 0.85,
    "n_features": 128,
    "samples_per_coarse_class": 120,
    "label_split": "disjoint",
    "feature_seed": 2718,
}

# Cross-time EMA values to sweep
# Convention: value = weight on current-round aggregate
# 1.0 = no blending (baseline), lower = more historical weight
EMA_VALUES = [1.0, 0.7, 0.5, 0.3]

# Also run with "none" label split to test shared-label scenario
DATASET_CFG_NONE = {
    "K": 10,
    "T": 20,
    "n_samples": 400,
    "rho": 3.0,
    "alpha": 0.75,
    "delta": 0.85,
    "n_features": 128,
    "samples_per_coarse_class": 120,
    "label_split": "none",
    "feature_seed": 2718,
}

SEED = 42


def run_one(
    ema_value: float, seed: int, dataset_cfg: dict | None = None
) -> dict:
    """Run FedProTrack with a specific cross_time_ema value."""
    ds_dict = dataset_cfg if dataset_cfg is not None else DATASET_CFG
    cfg = CIFAR100RecurrenceConfig(**{**ds_dict, "seed": seed})
    dataset = generate_cifar100_recurrence_dataset(cfg)
    ground_truth = dataset.concept_matrix
    n_concepts = int(ground_truth.max()) + 1

    two_phase_cfg = TwoPhaseConfig(
        max_concepts=max(6, n_concepts + 2),
        shrink_every=6,
    )

    runner = FedProTrackRunner(
        config=two_phase_cfg,
        seed=seed,
        federation_every=1,
        detector_name="NoDrift",
        lr=0.01,
        n_epochs=3,
        cross_time_ema=ema_value,
    )

    t0 = time.time()
    result = runner.run(dataset)
    elapsed = time.time() - t0

    log = result.to_experiment_log()
    metrics = compute_all_metrics(log, identity_capable=True)

    return {
        "cross_time_ema": ema_value,
        "seed": seed,
        "final_accuracy": float(metrics.final_accuracy),
        "accuracy_auc": float(metrics.accuracy_auc),
        "concept_re_id_accuracy": float(metrics.concept_re_id_accuracy),
        "wrong_memory_reuse_rate": float(metrics.wrong_memory_reuse_rate),
        "assignment_entropy": float(metrics.assignment_entropy),
        "total_bytes": float(result.total_bytes),
        "wall_clock_s": round(elapsed, 1),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure feature cache exists
    print("Checking / building CIFAR-100 feature cache...")
    cache_cfg = CIFAR100RecurrenceConfig(**{**DATASET_CFG, "seed": SEED})
    prepare_cifar100_recurrence_feature_cache(cache_cfg)
    print("Cache ready.\n")

    all_results: list[dict] = []
    for ema in EMA_VALUES:
        label = f"cross_time_ema={ema}"
        print(f"--- Running {label} (seed={SEED}) ---")
        try:
            row = run_one(ema, SEED)
            all_results.append(row)
            print(
                f"  final_acc={row['final_accuracy']:.4f}  "
                f"auc={row['accuracy_auc']:.4f}  "
                f"re_id={row['concept_re_id_accuracy']:.4f}  "
                f"wrong_mem={row['wrong_memory_reuse_rate']:.4f}  "
                f"time={row['wall_clock_s']}s"
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Save raw results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("Cross-Time EMA Validation Summary (CIFAR-100 disjoint, K=10, T=20)")
    print("=" * 80)
    print(
        f"{'EMA':>6s}  {'Final Acc':>10s}  {'AUC':>8s}  {'Re-ID':>8s}  "
        f"{'Wrong Mem':>10s}  {'Entropy':>8s}  {'Bytes':>10s}  {'Time':>6s}"
    )
    print("-" * 80)
    for row in all_results:
        print(
            f"{row['cross_time_ema']:>6.1f}  "
            f"{row['final_accuracy']:>10.4f}  "
            f"{row['accuracy_auc']:>8.4f}  "
            f"{row['concept_re_id_accuracy']:>8.4f}  "
            f"{row['wrong_memory_reuse_rate']:>10.4f}  "
            f"{row['assignment_entropy']:>8.4f}  "
            f"{row['total_bytes']:>10.0f}  "
            f"{row['wall_clock_s']:>6.1f}"
        )
    print("=" * 80)
    print(
        "\nNote: cross_time_ema = weight on current-round aggregate.\n"
        "  1.0 = no blending (baseline)\n"
        "  0.5 = 50/50 blend with stored historical model\n"
        "  0.3 = 30% current + 70% historical\n"
    )

    # Identify best
    if all_results:
        best = max(all_results, key=lambda r: r["final_accuracy"])
        baseline = next(
            (r for r in all_results if r["cross_time_ema"] == 1.0), all_results[0]
        )
        delta = best["final_accuracy"] - baseline["final_accuracy"]
        print(
            f"Best EMA (disjoint) = {best['cross_time_ema']:.1f}: "
            f"final_acc = {best['final_accuracy']:.4f} "
            f"(delta vs baseline = {delta:+.4f})"
        )

    # --- Repeat with "none" (shared labels) ---
    print("\n\n" + "=" * 80)
    print("Now testing with label_split='none' (shared labels)")
    print("=" * 80)

    none_results: list[dict] = []
    cache_cfg_none = CIFAR100RecurrenceConfig(**{**DATASET_CFG_NONE, "seed": SEED})
    print("Checking / building 'none' label split cache...")
    prepare_cifar100_recurrence_feature_cache(cache_cfg_none)
    print("Cache ready.\n")

    for ema in EMA_VALUES:
        label = f"cross_time_ema={ema}"
        print(f"--- Running {label} (seed={SEED}, label_split=none) ---")
        try:
            row = run_one(ema, SEED, dataset_cfg=DATASET_CFG_NONE)
            none_results.append(row)
            print(
                f"  final_acc={row['final_accuracy']:.4f}  "
                f"auc={row['accuracy_auc']:.4f}  "
                f"re_id={row['concept_re_id_accuracy']:.4f}  "
                f"wrong_mem={row['wrong_memory_reuse_rate']:.4f}  "
                f"time={row['wall_clock_s']}s"
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Save none results
    none_path = RESULTS_DIR / "results_none.json"
    with open(none_path, "w", encoding="utf-8") as f:
        json.dump(none_results, f, indent=2)

    print("\n" + "=" * 80)
    print("Cross-Time EMA Validation Summary (CIFAR-100 none/shared, K=10, T=20)")
    print("=" * 80)
    print(
        f"{'EMA':>6s}  {'Final Acc':>10s}  {'AUC':>8s}  {'Re-ID':>8s}  "
        f"{'Wrong Mem':>10s}  {'Entropy':>8s}  {'Bytes':>10s}  {'Time':>6s}"
    )
    print("-" * 80)
    for row in none_results:
        print(
            f"{row['cross_time_ema']:>6.1f}  "
            f"{row['final_accuracy']:>10.4f}  "
            f"{row['accuracy_auc']:>8.4f}  "
            f"{row['concept_re_id_accuracy']:>8.4f}  "
            f"{row['wrong_memory_reuse_rate']:>10.4f}  "
            f"{row['assignment_entropy']:>8.4f}  "
            f"{row['total_bytes']:>10.0f}  "
            f"{row['wall_clock_s']:>6.1f}"
        )
    print("=" * 80)

    if none_results:
        best_none = max(none_results, key=lambda r: r["final_accuracy"])
        baseline_none = next(
            (r for r in none_results if r["cross_time_ema"] == 1.0),
            none_results[0],
        )
        delta_none = best_none["final_accuracy"] - baseline_none["final_accuracy"]
        print(
            f"Best EMA (none) = {best_none['cross_time_ema']:.1f}: "
            f"final_acc = {best_none['final_accuracy']:.4f} "
            f"(delta vs baseline = {delta_none:+.4f})"
        )

    # Save combined summary
    combined = {
        "disjoint": all_results,
        "none": none_results,
    }
    combined_path = RESULTS_DIR / "results_combined.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")


if __name__ == "__main__":
    main()
