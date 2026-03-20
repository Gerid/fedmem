from __future__ import annotations

"""H-adapter: Does feature_adapter model close the accuracy gap for FedProTrack?

Compares FPT-linear-base, FPT-adapter-base, FPT-adapter-hybrid-proto against
CFL and IFCA on CIFAR-100 with K=4, T=20, 5 seeds.
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

SEEDS = [42, 43, 44, 45, 46]
K = 4
T = 20
N_SAMPLES = 400
N_FEATURES = 64
SAMPLES_PER_COARSE_CLASS = 30
FEDERATION_EVERY = 2
FPT_LR = 0.05
FPT_EPOCHS = 5

METHODS = [
    "FPT-linear-base",
    "FPT-adapter-base",
    "FPT-adapter-hybrid-proto",
    "CFL",
    "IFCA",
]


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


def _build_fpt_config(dataset) -> TwoPhaseConfig:
    return TwoPhaseConfig(
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
    )


def _common_fpt_kwargs() -> dict:
    return {
        "auto_scale": False,
        "update_ot_weight": 0.0,
        "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0,
        "labelwise_proto_dim": 4,
        "prototype_alignment_early_rounds": 0,
        "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0,
        "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0,
        "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }


def run_single(method: str, dataset) -> object:
    seed = int(dataset.config.seed)
    cfg = _build_fpt_config(dataset)
    common = _common_fpt_kwargs()

    if method == "FPT-linear-base":
        return FedProTrackRunner(
            config=cfg,
            federation_every=FEDERATION_EVERY,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=FPT_EPOCHS,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="linear",
            similarity_calibration=False,
            model_signature_weight=0.0,
            model_signature_dim=8,
            prototype_alignment_mix=0.0,
            **common,
        ).run(dataset)

    if method == "FPT-adapter-base":
        return FedProTrackRunner(
            config=cfg,
            federation_every=FEDERATION_EVERY,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=FPT_EPOCHS,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="feature_adapter",
            hidden_dim=64,
            adapter_dim=16,
            similarity_calibration=False,
            model_signature_weight=0.0,
            model_signature_dim=8,
            prototype_alignment_mix=0.0,
            **common,
        ).run(dataset)

    if method == "FPT-adapter-hybrid-proto":
        return FedProTrackRunner(
            config=cfg,
            federation_every=FEDERATION_EVERY,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=FPT_EPOCHS,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="feature_adapter",
            hidden_dim=64,
            adapter_dim=16,
            similarity_calibration=True,
            model_signature_weight=0.55,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            **common,
        ).run(dataset)

    if method == "CFL":
        return run_cfl_full(dataset, federation_every=FEDERATION_EVERY)

    if method == "IFCA":
        return run_ifca_full(dataset, federation_every=FEDERATION_EVERY, n_clusters=4)

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    results_dir = Path("tmp/cifar100_adapter")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_failures: list[dict] = []

    total_combos = len(SEEDS) * len(METHODS)
    done = 0

    print("Preparing CIFAR-100 feature cache...", flush=True)
    cache_cfg = CIFAR100RecurrenceConfig(K=K, T=T, n_samples=N_SAMPLES, seed=42,
                                          n_features=N_FEATURES,
                                          samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS)
    prepare_cifar100_recurrence_feature_cache(cache_cfg)

    for seed in SEEDS:
        dataset_cfg = CIFAR100RecurrenceConfig(
            K=K,
            T=T,
            n_samples=N_SAMPLES,
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
            print(f"[{done}/{total_combos}] seed={seed} {method}...", flush=True)
            t0 = time.time()
            try:
                result = run_single(method, dataset)
                log = _make_log(method, result, dataset.concept_matrix)
                canon = canonical_method_name(
                    "FedProTrack" if method.startswith("FPT") else method
                )
                metrics = compute_all_metrics(
                    log, identity_capable=identity_metrics_valid(canon)
                )
                row = {
                    "method": method,
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
        "method", "seed", "final_accuracy", "accuracy_auc",
        "concept_re_id_accuracy", "assignment_entropy",
        "wrong_memory_reuse_rate", "total_bytes", "wall_clock_s",
    ]
    with open(results_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Print summary table
    print("\n=== H-Adapter Summary (mean +/- std over seeds) ===", flush=True)
    print(
        f"{'Method':30s}  {'FinalAcc':>10s}  {'Re-ID':>10s}  {'AUC':>10s}  {'Bytes':>8s}",
        flush=True,
    )
    print("-" * 75, flush=True)
    for method in METHODS:
        subset = [r for r in all_rows if r["method"] == method]
        if not subset:
            continue
        accs = [r["final_accuracy"] for r in subset]
        reids = [r["concept_re_id_accuracy"] for r in subset if r["concept_re_id_accuracy"] is not None]
        aucs = [r["accuracy_auc"] for r in subset]
        bytes_vals = [r["total_bytes"] for r in subset]

        acc_str = f"{np.mean(accs):.4f}+/-{np.std(accs):.4f}"
        reid_str = f"{np.mean(reids):.4f}+/-{np.std(reids):.4f}" if reids else "N/A"
        auc_str = f"{np.mean(aucs):.4f}+/-{np.std(aucs):.4f}"
        bytes_str = f"{np.mean(bytes_vals):.0f}"
        print(f"{method:30s}  {acc_str:>10s}  {reid_str:>10s}  {auc_str:>10s}  {bytes_str:>8s}", flush=True)

    print(f"\nFailures: {len(all_failures)}", flush=True)
    print(f"Results saved to: {results_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
