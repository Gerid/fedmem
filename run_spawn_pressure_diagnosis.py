from __future__ import annotations

"""Diagnose and compare FedProTrack over-spawning with spawn_pressure_damping.

Runs CIFAR-100 recurrence with:
  - baseline FPT (no pressure damping)
  - FPT + spawn_pressure_damping=2.0
  - FPT + spawn_pressure_damping=4.0
  - FPT + max_concepts=5 + spawn_pressure_damping=3.0
  - FPT + novelty_hysteresis_rounds=2 + spawn_pressure_damping=3.0

Prints spawned concepts, final accuracy, and re-ID for each variant.
"""

import json
import time
from pathlib import Path

import numpy as np

from fedprotrack.experiment.baselines import run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import identity_metrics_valid
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def _make_log(result: object, ground_truth: np.ndarray, name: str) -> ExperimentLog:
    total_bytes = getattr(result, "total_bytes", None)
    if total_bytes is not None and float(total_bytes) <= 0.0:
        total_bytes = None
    if hasattr(result, "to_experiment_log"):
        try:
            return result.to_experiment_log(ground_truth)
        except TypeError:
            return result.to_experiment_log()
    return ExperimentLog(
        ground_truth=ground_truth,
        predicted=np.asarray(getattr(result, "predicted_concept_matrix"), dtype=np.int32),
        accuracy_curve=np.asarray(getattr(result, "accuracy_matrix"), dtype=np.float64),
        total_bytes=total_bytes,
        method_name=name,
    )


def _run_variant(
    name: str,
    dataset: object,
    *,
    spawn_pressure_damping: float = 0.0,
    max_concepts: int = 8,
    novelty_hysteresis_rounds: int = 1,
    loss_novelty_threshold: float = 0.15,
    merge_threshold: float = 0.85,
) -> dict:
    """Run a single FPT variant and return metrics dict."""
    print(f"\n--- Running: {name} ---")
    print(f"  spawn_pressure_damping={spawn_pressure_damping}")
    print(f"  max_concepts={max_concepts}")
    print(f"  novelty_hysteresis_rounds={novelty_hysteresis_rounds}")
    print(f"  loss_novelty_threshold={loss_novelty_threshold}")
    print(f"  merge_threshold={merge_threshold}")

    t0 = time.time()
    runner = FedProTrackRunner(
        config=TwoPhaseConfig(
            omega=2.0,
            kappa=0.7,
            novelty_threshold=0.25,
            loss_novelty_threshold=loss_novelty_threshold,
            sticky_dampening=1.5,
            sticky_posterior_gate=0.35,
            merge_threshold=merge_threshold,
            min_count=5.0,
            max_concepts=max_concepts,
            merge_every=2,
            shrink_every=6,
            spawn_pressure_damping=spawn_pressure_damping,
            novelty_hysteresis_rounds=novelty_hysteresis_rounds,
        ),
        federation_every=2,
        detector_name="ADWIN",
        seed=int(dataset.config.seed),
        lr=0.05,
        n_epochs=5,
        soft_aggregation=True,
        blend_alpha=0.0,
    )
    result = runner.run(dataset)
    elapsed = time.time() - t0

    gt = dataset.concept_matrix
    log = _make_log(result, gt, name)
    metrics = compute_all_metrics(log)

    spawned = getattr(result, "spawned_concepts", "?")
    active = getattr(result, "active_concepts", "?")
    merged = getattr(result, "merged_concepts", "?")

    true_concepts = len(np.unique(gt))

    print(f"  Time: {elapsed:.1f}s")
    print(f"  True concepts: {true_concepts}")
    print(f"  Spawned: {spawned}, Merged: {merged}, Active: {active}")
    final_acc = getattr(metrics, 'final_accuracy', None) or 0.0
    mean_acc = getattr(metrics, 'mean_accuracy', None) or 0.0
    re_id = getattr(metrics, "concept_re_id_accuracy", None)
    print(f"  Final accuracy: {final_acc:.4f}")
    print(f"  Mean accuracy:  {mean_acc:.4f}")
    print(f"  Re-ID accuracy: {re_id:.4f}" if re_id is not None else "  Re-ID accuracy: N/A")

    return {
        "name": name,
        "spawned": int(spawned) if isinstance(spawned, (int, np.integer)) else spawned,
        "merged": int(merged) if isinstance(merged, (int, np.integer)) else merged,
        "active": int(active) if isinstance(active, (int, np.integer)) else active,
        "true_concepts": int(true_concepts),
        "final_accuracy": float(final_acc),
        "mean_accuracy": float(mean_acc),
        "re_id": float(re_id) if re_id is not None else None,
        "elapsed": round(elapsed, 1),
    }


def main() -> None:
    results_dir = Path("E:/fedprotrack/.claude/worktrees/elegant-poitras/tmp/spawn_pressure_diagnosis")
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    K, T = 6, 10
    n_features = 64

    print("=" * 60)
    print("CIFAR-100 Spawn Pressure Diagnosis")
    print(f"K={K}, T={T}, n_features={n_features}, seed={seed}")
    print("=" * 60)

    # Build feature cache and dataset
    print("\nPreparing CIFAR-100 feature cache...")

    cifar_cfg = CIFAR100RecurrenceConfig(
        K=K,
        T=T,
        n_samples=200,
        rho=2.0,
        alpha=0.75,
        delta=0.9,
        seed=seed,
        n_features=n_features,
        samples_per_coarse_class=30,
        label_split="disjoint",
    )
    dataset = generate_cifar100_recurrence_dataset(cifar_cfg)
    gt = dataset.concept_matrix
    true_k = len(np.unique(gt))
    print(f"Dataset ready. True concepts: {true_k}")
    print(f"Concept matrix:\n{gt}")

    # Also run Oracle for reference
    print("\n--- Running: Oracle ---")
    from fedprotrack.drift_generator import GeneratorConfig
    gen_cfg = GeneratorConfig(
        K=K, T=T, n_samples=200,
        generator_type="cifar100_recurrence",
        seed=seed,
    )
    exp_cfg = ExperimentConfig(generator_config=gen_cfg)
    t0 = time.time()
    oracle_result = run_oracle_baseline(exp_cfg, dataset=dataset)
    oracle_elapsed = time.time() - t0
    oracle_log = _make_log(oracle_result, gt, "Oracle")
    oracle_metrics = compute_all_metrics(oracle_log)
    print(f"  Time: {oracle_elapsed:.1f}s")
    print(f"  Final accuracy: {getattr(oracle_metrics, 'final_accuracy', 0):.4f}")
    print(f"  Mean accuracy:  {getattr(oracle_metrics, 'mean_accuracy', 0):.4f}")

    # Define variants
    variants = [
        ("FPT-baseline", dict(
            spawn_pressure_damping=0.0,
            max_concepts=8,
        )),
        ("FPT-pressure2", dict(
            spawn_pressure_damping=2.0,
            max_concepts=8,
        )),
        ("FPT-pressure4", dict(
            spawn_pressure_damping=4.0,
            max_concepts=8,
        )),
        ("FPT-pressure8", dict(
            spawn_pressure_damping=8.0,
            max_concepts=8,
        )),
        ("FPT-cap5-pressure3", dict(
            spawn_pressure_damping=3.0,
            max_concepts=5,
        )),
        ("FPT-hysteresis2-pressure3", dict(
            spawn_pressure_damping=3.0,
            max_concepts=8,
            novelty_hysteresis_rounds=2,
        )),
        ("FPT-cap5-hysteresis2-pressure4", dict(
            spawn_pressure_damping=4.0,
            max_concepts=5,
            novelty_hysteresis_rounds=2,
        )),
        ("FPT-aggressive-merge", dict(
            spawn_pressure_damping=0.0,
            max_concepts=8,
            merge_threshold=0.70,
        )),
        ("FPT-aggressive-merge-pressure3", dict(
            spawn_pressure_damping=3.0,
            max_concepts=8,
            merge_threshold=0.70,
        )),
    ]

    all_results = []
    for name, kwargs in variants:
        try:
            row = _run_variant(name, dataset, **kwargs)
            all_results.append(row)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Oracle final_acc={getattr(oracle_metrics, 'final_accuracy', 0) or 0:.4f}")
    print(f"True concepts: {true_k}")
    print()
    header = f"{'Variant':<40} {'Spawned':>7} {'Active':>7} {'FinalAcc':>8} {'MeanAcc':>8} {'ReID':>6}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        re_id_str = f"{r['re_id']:.3f}" if r['re_id'] is not None else "N/A"
        print(
            f"{r['name']:<40} {r['spawned']:>7} {r['active']:>7} "
            f"{r['final_accuracy']:>8.4f} {r['mean_accuracy']:>8.4f} {re_id_str:>6}"
        )

    # Save results
    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
