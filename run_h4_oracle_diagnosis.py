from __future__ import annotations

"""H4: Oracle n_samples diagnosis experiment for CIFAR-100.

Tests whether Oracle baseline's poor performance (0.15 final_acc) is due to
data scarcity at n_samples=400. Sweeps n_samples=[200,400,800,1200,2000] to
find the crossover where concept specialization becomes beneficial.
"""

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import canonical_method_name, identity_metrics_valid
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

K = 4
T = 20
FEDERATION_EVERY = 2
N_FEATURES = 64
SAMPLES_PER_COARSE_CLASS = 30
FPT_LR = 0.05
FPT_EPOCHS = 5

N_SAMPLES_VALUES = [200, 400, 800, 1200, 2000]
SEEDS = [42, 43, 44, 45, 46]
METHODS = ["Oracle", "FPT-linear", "CFL", "FedAvg"]


def _make_log(method_name: str, result: object, ground_truth: np.ndarray) -> ExperimentLog:
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
        method_name=method_name,
    )


def run_single(method: str, dataset, exp_cfg: ExperimentConfig) -> object:
    seed = int(dataset.config.seed)

    if method == "Oracle":
        return run_oracle_baseline(exp_cfg, dataset=dataset)

    if method == "FedAvg":
        return run_fedavg_baseline(exp_cfg, dataset=dataset)

    if method == "CFL":
        return run_cfl_full(dataset, federation_every=FEDERATION_EVERY)

    if method == "FPT-linear":
        return FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, int(dataset.concept_matrix.max()) + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=FEDERATION_EVERY,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=FPT_EPOCHS,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="linear",
            auto_scale=False,
            similarity_calibration=False,
            model_signature_weight=0.0,
            model_signature_dim=8,
            update_ot_weight=0.0,
            update_ot_dim=4,
            labelwise_proto_weight=0.0,
            labelwise_proto_dim=4,
            prototype_alignment_mix=0.0,
            prototype_alignment_early_rounds=0,
            prototype_alignment_early_mix=0.0,
            prototype_prealign_early_rounds=0,
            prototype_prealign_early_mix=0.0,
            prototype_subgroup_early_rounds=0,
            prototype_subgroup_early_mix=0.0,
            prototype_subgroup_min_clients=3,
            prototype_subgroup_similarity_gate=0.8,
        ).run(dataset)

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    results_dir = Path("tmp/cifar100_h4_oracle")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_failures: list[dict] = []

    total_combos = len(N_SAMPLES_VALUES) * len(SEEDS) * len(METHODS)
    done = 0

    # Prepare cache once
    print("Preparing CIFAR-100 feature cache...", flush=True)
    cache_cfg = CIFAR100RecurrenceConfig(
        K=K, T=T, n_samples=200, seed=42,
        n_features=N_FEATURES, samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS,
    )
    prepare_cifar100_recurrence_feature_cache(cache_cfg)

    for n_samples in N_SAMPLES_VALUES:
        for seed in SEEDS:
            dataset_cfg = CIFAR100RecurrenceConfig(
                K=K,
                T=T,
                n_samples=n_samples,
                rho=2.0,
                alpha=0.75,
                delta=0.9,
                n_features=N_FEATURES,
                samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS,
                batch_size=128,
                n_workers=0,
                seed=seed,
            )
            dataset = generate_cifar100_recurrence_dataset(dataset_cfg)
            exp_cfg = ExperimentConfig(
                generator_config=dataset.config,
                federation_every=FEDERATION_EVERY,
            )

            for method in METHODS:
                done += 1
                print(f"[{done}/{total_combos}] n_samples={n_samples} seed={seed} {method}...", flush=True)
                t0 = time.time()
                try:
                    result = run_single(method, dataset, exp_cfg)
                    log = _make_log(method, result, dataset.concept_matrix)
                    canon = canonical_method_name(
                        "FedProTrack" if method.startswith("FPT") else method
                    )
                    metrics = compute_all_metrics(
                        log, identity_capable=identity_metrics_valid(canon)
                    )
                    row = {
                        "method": method,
                        "n_samples": n_samples,
                        "seed": seed,
                        "final_accuracy": metrics.final_accuracy,
                        "accuracy_auc": metrics.accuracy_auc,
                        "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                        "assignment_entropy": metrics.assignment_entropy,
                        "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
                        "total_bytes": float(getattr(result, "total_bytes", 0.0) or 0.0),
                        "wall_clock_s": time.time() - t0,
                    }
                    all_rows.append(row)
                    acc_str = f"acc={row['final_accuracy']:.4f}"
                    reid_str = (
                        f"re-ID={row['concept_re_id_accuracy']:.4f}"
                        if row["concept_re_id_accuracy"] is not None
                        else "re-ID=N/A"
                    )
                    print(
                        f"  -> {acc_str}  {reid_str}  bytes={row['total_bytes']:.0f}  ({time.time()-t0:.1f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    all_failures.append({
                        "method": method,
                        "n_samples": n_samples,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "wall_clock_s": time.time() - t0,
                    })
                    print(f"  -> FAILED: {exc}", flush=True)

    # Save results
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "failures": all_failures}, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "method", "n_samples", "seed", "final_accuracy", "accuracy_auc",
        "concept_re_id_accuracy", "assignment_entropy",
        "wrong_memory_reuse_rate", "total_bytes", "wall_clock_s",
    ]
    with open(results_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary table
    print("\n=== H4 Oracle Diagnosis Summary (mean over 5 seeds) ===", flush=True)
    print(
        f"{'n_samples':>9s}  {'Method':12s}  {'FinalAcc':>8s}  {'Re-ID':>8s}  {'AUC':>8s}  {'Bytes':>8s}",
        flush=True,
    )
    print("-" * 65, flush=True)
    for n_samples in N_SAMPLES_VALUES:
        for method in METHODS:
            subset = [r for r in all_rows if r["n_samples"] == n_samples and r["method"] == method]
            if not subset:
                continue
            mean_acc = np.mean([r["final_accuracy"] for r in subset])
            reid_vals = [r["concept_re_id_accuracy"] for r in subset if r["concept_re_id_accuracy"] is not None]
            mean_reid = np.mean(reid_vals) if reid_vals else float("nan")
            mean_auc = np.mean([r["accuracy_auc"] for r in subset])
            mean_bytes = np.mean([r["total_bytes"] for r in subset])
            reid_str = f"{mean_reid:.4f}" if not np.isnan(mean_reid) else "N/A"
            print(
                f"{n_samples:9d}  {method:12s}  {mean_acc:8.4f}  {reid_str:>8s}  {mean_auc:8.4f}  {mean_bytes:8.0f}",
                flush=True,
            )

    print(f"\nFailures: {len(all_failures)}", flush=True)
    print(f"Results saved to: {results_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
