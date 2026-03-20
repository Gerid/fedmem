from __future__ import annotations

"""H1: Temporal horizon T-sweep experiment for CIFAR-100.

Tests whether FedProTrack re-ID advantage emerges at T>=20.
Runs FedProTrack-base, FedProTrack-hybrid-proto, CFL, IFCA across
T=[6,10,15,20,30,40], K=4, n_samples=200, 5 seeds.
"""

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full, run_ifca_full
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


T_VALUES = [6, 10, 15, 20, 30, 40]
SEEDS = [42, 43, 44, 45, 46]
K = 4
N_SAMPLES = 200
FEDERATION_EVERY = 2
FPT_LR = 0.05
FPT_EPOCHS = 5


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


def _build_fpt_runner(dataset, *, mode: str) -> FedProTrackRunner:
    fpt_kwargs = {
        "auto_scale": False,
        "similarity_calibration": mode == "hybrid-proto",
        "model_signature_weight": 0.55 if mode == "hybrid-proto" else 0.0,
        "model_signature_dim": 8,
        "update_ot_weight": 0.0,
        "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0,
        "labelwise_proto_dim": 4,
        "prototype_alignment_mix": 0.25 if mode == "hybrid-proto" else 0.0,
        "prototype_alignment_early_rounds": 0,
        "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0,
        "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0,
        "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }
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
        seed=int(dataset.config.seed),
        lr=FPT_LR,
        n_epochs=FPT_EPOCHS,
        soft_aggregation=True,
        blend_alpha=0.0,
        **fpt_kwargs,
    )


def run_single(method: str, dataset, exp_cfg: ExperimentConfig) -> object:
    if method == "FedProTrack-base":
        return _build_fpt_runner(dataset, mode="base").run(dataset)
    elif method == "FedProTrack-hybrid-proto":
        return _build_fpt_runner(dataset, mode="hybrid-proto").run(dataset)
    elif method == "CFL":
        return run_cfl_full(dataset, federation_every=FEDERATION_EVERY)
    elif method == "IFCA":
        return run_ifca_full(dataset, federation_every=FEDERATION_EVERY, n_clusters=4)
    else:
        raise ValueError(f"Unknown method: {method}")


def main() -> None:
    results_dir = Path("tmp/cifar100_h1_tsweep")
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = ["FedProTrack-base", "FedProTrack-hybrid-proto", "CFL", "IFCA"]
    all_rows: list[dict] = []
    all_failures: list[dict] = []

    total_combos = len(T_VALUES) * len(SEEDS) * len(methods)
    done = 0

    # Prepare cache once (uses default T — cache is shared across T values)
    print("Preparing CIFAR-100 feature cache...", flush=True)
    cache_cfg = CIFAR100RecurrenceConfig(K=K, T=6, n_samples=N_SAMPLES, seed=42)
    prepare_cifar100_recurrence_feature_cache(cache_cfg)

    for T in T_VALUES:
        for seed in SEEDS:
            dataset_cfg = CIFAR100RecurrenceConfig(
                K=K,
                T=T,
                n_samples=N_SAMPLES,
                rho=2.0,
                alpha=0.75,
                delta=0.9,
                n_features=64,
                samples_per_coarse_class=30,
                batch_size=128,
                n_workers=0,
                seed=seed,
            )
            dataset = generate_cifar100_recurrence_dataset(dataset_cfg)
            exp_cfg = ExperimentConfig(
                generator_config=dataset.config,
                federation_every=FEDERATION_EVERY,
            )

            for method in methods:
                done += 1
                print(f"[{done}/{total_combos}] T={T} seed={seed} {method}...", flush=True)
                t0 = time.time()
                try:
                    result = run_single(method, dataset, exp_cfg)
                    log = _make_log(method, result, dataset.concept_matrix)
                    canon = canonical_method_name(
                        "FedProTrack" if method.startswith("FedProTrack") else method
                    )
                    metrics = compute_all_metrics(
                        log, identity_capable=identity_metrics_valid(canon)
                    )
                    row = {
                        "T": T,
                        "seed": seed,
                        "method": method,
                        "final_accuracy": metrics.final_accuracy,
                        "accuracy_auc": metrics.accuracy_auc,
                        "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                        "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
                        "assignment_entropy": metrics.assignment_entropy,
                        "total_bytes": float(getattr(result, "total_bytes", 0.0) or 0.0),
                        "wall_clock_s": time.time() - t0,
                    }
                    all_rows.append(row)
                    acc_str = f"acc={row['final_accuracy']:.4f}"
                    reid_str = f"re-ID={row['concept_re_id_accuracy']:.4f}" if row['concept_re_id_accuracy'] is not None else "re-ID=N/A"
                    print(f"  -> {acc_str}  {reid_str}  bytes={row['total_bytes']:.0f}  ({time.time()-t0:.1f}s)", flush=True)
                except Exception as exc:
                    all_failures.append({
                        "T": T,
                        "seed": seed,
                        "method": method,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "wall_clock_s": time.time() - t0,
                    })
                    print(f"  -> FAILED: {exc}", flush=True)

    # Save results
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "failures": all_failures}, f, indent=2, ensure_ascii=False)

    with open(results_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["T", "seed", "method", "final_accuracy", "accuracy_auc",
                      "concept_re_id_accuracy", "wrong_memory_reuse_rate",
                      "assignment_entropy", "total_bytes", "wall_clock_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary table
    print("\n=== H1 T-Sweep Summary (mean over 5 seeds) ===", flush=True)
    print(f"{'T':>4s}  {'Method':30s}  {'FinalAcc':>8s}  {'Re-ID':>8s}  {'AUC':>8s}  {'Bytes':>8s}", flush=True)
    print("-" * 75, flush=True)
    for T in T_VALUES:
        for method in methods:
            subset = [r for r in all_rows if r["T"] == T and r["method"] == method]
            if not subset:
                continue
            mean_acc = np.mean([r["final_accuracy"] for r in subset])
            reid_vals = [r["concept_re_id_accuracy"] for r in subset if r["concept_re_id_accuracy"] is not None]
            mean_reid = np.mean(reid_vals) if reid_vals else float("nan")
            mean_auc = np.mean([r["accuracy_auc"] for r in subset])
            mean_bytes = np.mean([r["total_bytes"] for r in subset])
            reid_str = f"{mean_reid:.4f}" if not np.isnan(mean_reid) else "N/A"
            print(f"{T:4d}  {method:30s}  {mean_acc:8.4f}  {reid_str:>8s}  {mean_auc:8.4f}  {mean_bytes:8.0f}", flush=True)

    print(f"\nFailures: {len(all_failures)}", flush=True)
    print(f"Results saved to: {results_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
