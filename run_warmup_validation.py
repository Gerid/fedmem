from __future__ import annotations

"""Validate FedAvg warmup rounds for FedProTrack on CIFAR-100.

Compares FPT with warmup_rounds=0 vs 5 vs 10 on a small CIFAR-100
recurrence scenario (K=10, T=20, 1 seed, label_split='disjoint').
"""

import json
import time
from pathlib import Path

import numpy as np

from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


RESULTS_DIR = Path(__file__).parent / "tmp" / "warmup_validation"
SEED = 42
WARMUP_VALUES = [0, 5, 10]

DATASET_CFG = dict(
    K=10,
    T=20,
    n_samples=200,
    rho=0.6,
    alpha=0.8,
    delta=0.3,
    n_features=20,
    samples_per_coarse_class=100,
    batch_size=64,
    n_workers=0,
    seed=SEED,
)


def _make_log(
    method_name: str,
    result: object,
    ground_truth: np.ndarray,
) -> ExperimentLog:
    total_bytes = getattr(result, "total_bytes", None)
    if total_bytes is not None and float(total_bytes) <= 0.0:
        total_bytes = None
    return ExperimentLog(
        ground_truth=ground_truth,
        predicted=np.asarray(
            getattr(result, "predicted_concept_matrix"), dtype=np.int32
        ),
        accuracy_curve=np.asarray(
            getattr(result, "accuracy_matrix"), dtype=np.float64
        ),
        total_bytes=total_bytes,
        method_name=method_name,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    ds_cfg = CIFAR100RecurrenceConfig(**DATASET_CFG)
    print("Preparing CIFAR-100 feature cache...", flush=True)
    prepare_cifar100_recurrence_feature_cache(ds_cfg)
    dataset = generate_cifar100_recurrence_dataset(ds_cfg)

    results = {}

    for warmup in WARMUP_VALUES:
        label = f"FPT-warmup{warmup}"
        print(f"\n{'='*60}", flush=True)
        print(f"Running {label} ...", flush=True)
        t0 = time.time()

        runner = FedProTrackRunner(
            config=TwoPhaseConfig(),
            federation_every=2,
            detector_name="ADWIN",
            seed=SEED,
            soft_aggregation=True,
            warmup_rounds=warmup,
            lr=0.1,
            n_epochs=1,
        )
        result = runner.run(dataset)
        elapsed = time.time() - t0

        log = _make_log(label, result, dataset.concept_matrix)
        metrics = compute_all_metrics(log)

        row = {
            "warmup_rounds": warmup,
            "final_acc": float(result.final_accuracy),
            "mean_acc": float(result.mean_accuracy),
            "total_bytes": float(result.total_bytes),
            "phase_a_bytes": float(result.phase_a_bytes),
            "phase_b_bytes": float(result.phase_b_bytes),
            "spawned": result.spawned_concepts,
            "active": result.active_concepts,
            "re_id": float(metrics.concept_re_id_accuracy or 0.0),
            "elapsed_s": round(elapsed, 1),
        }
        results[label] = row
        print(f"  final_acc={row['final_acc']:.4f}  mean_acc={row['mean_acc']:.4f}"
              f"  re_id={row['re_id']:.4f}  bytes={row['total_bytes']:.0f}"
              f"  spawned={row['spawned']}  active={row['active']}"
              f"  time={row['elapsed_s']}s",
              flush=True)

    # Save results
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary table
    print("\n" + "=" * 70, flush=True)
    print(f"{'Warmup':>10} | {'Final Acc':>10} | {'Mean Acc':>10} | "
          f"{'Re-ID':>8} | {'Bytes':>12} | {'Spawned':>8}", flush=True)
    print("-" * 70, flush=True)
    for label, row in results.items():
        print(f"{row['warmup_rounds']:>10} | {row['final_acc']:>10.4f} | "
              f"{row['mean_acc']:>10.4f} | {row['re_id']:>8.4f} | "
              f"{row['total_bytes']:>12.0f} | {row['spawned']:>8}", flush=True)


if __name__ == "__main__":
    main()
