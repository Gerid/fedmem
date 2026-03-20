from __future__ import annotations

"""H2c+H2d: Posterior dynamics fix for feature_adapter model.

Tests whether relaxing Gibbs thresholds (calibrated for 5-epoch linear)
restores concept tracking for the adapter model with 30 epochs of local
training. The hypothesis: posterior entropy collapse (not fingerprint
distortion) is the root cause of adapter re-ID drop.

6 methods x 5 seeds, K=4, T=20, n_samples=400.
"""

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
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
FPT_LR = 0.05

METHODS = [
    "adapter-baseline",
    "adapter-relaxed",
    "adapter-very-relaxed",
    "adapter-fewer-epochs",
    "linear",
    "CFL",
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


def _common_fpt_kwargs() -> dict:
    return {
        "auto_scale": False,
        "similarity_calibration": False,
        "model_signature_weight": 0.0,
        "model_signature_dim": 8,
        "update_ot_weight": 0.0,
        "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0,
        "labelwise_proto_dim": 4,
        "prototype_alignment_mix": 0.0,
        "prototype_alignment_early_rounds": 0,
        "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0,
        "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0,
        "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }


def _baseline_config(dataset) -> TwoPhaseConfig:
    """Original thresholds calibrated for 5-epoch linear training."""
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


def _relaxed_config(dataset) -> TwoPhaseConfig:
    """Relaxed thresholds for stronger local training (30 epochs)."""
    return TwoPhaseConfig(
        omega=2.0,
        kappa=0.7,
        novelty_threshold=0.40,
        loss_novelty_threshold=0.25,
        sticky_dampening=0.8,
        sticky_posterior_gate=0.20,
        merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, int(dataset.concept_matrix.max()) + 3),
        merge_every=2,
        shrink_every=6,
    )


def _very_relaxed_config(dataset) -> TwoPhaseConfig:
    """Even more relaxed thresholds."""
    return TwoPhaseConfig(
        omega=2.0,
        kappa=0.7,
        novelty_threshold=0.50,
        loss_novelty_threshold=0.35,
        sticky_dampening=0.5,
        sticky_posterior_gate=0.15,
        merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, int(dataset.concept_matrix.max()) + 3),
        merge_every=2,
        shrink_every=6,
    )


def run_single(method: str, dataset) -> object:
    seed = int(dataset.config.seed)
    common = _common_fpt_kwargs()

    if method == "adapter-baseline":
        return FedProTrackRunner(
            config=_baseline_config(dataset),
            federation_every=5,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=30,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="feature_adapter",
            hidden_dim=64,
            adapter_dim=16,
            **common,
        ).run(dataset)

    if method == "adapter-relaxed":
        return FedProTrackRunner(
            config=_relaxed_config(dataset),
            federation_every=5,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=30,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="feature_adapter",
            hidden_dim=64,
            adapter_dim=16,
            **common,
        ).run(dataset)

    if method == "adapter-very-relaxed":
        return FedProTrackRunner(
            config=_very_relaxed_config(dataset),
            federation_every=5,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=30,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="feature_adapter",
            hidden_dim=64,
            adapter_dim=16,
            **common,
        ).run(dataset)

    if method == "adapter-fewer-epochs":
        return FedProTrackRunner(
            config=_baseline_config(dataset),
            federation_every=5,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=10,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="feature_adapter",
            hidden_dim=64,
            adapter_dim=16,
            **common,
        ).run(dataset)

    if method == "linear":
        return FedProTrackRunner(
            config=_baseline_config(dataset),
            federation_every=2,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=5,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="linear",
            **common,
        ).run(dataset)

    if method == "CFL":
        return run_cfl_full(dataset, federation_every=2)

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    results_dir = Path("tmp/cifar100_h2cd")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_failures: list[dict] = []

    total_combos = len(SEEDS) * len(METHODS)
    done = 0

    print("Preparing CIFAR-100 feature cache...", flush=True)
    cache_cfg = CIFAR100RecurrenceConfig(
        K=K, T=T, n_samples=N_SAMPLES, seed=42,
        n_features=N_FEATURES,
        samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS,
    )
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

        for method in METHODS:
            done += 1
            print(f"[{done}/{total_combos}] seed={seed} {method}...", flush=True)
            t0 = time.time()
            try:
                result = run_single(method, dataset)
                log = _make_log(method, result, dataset.concept_matrix)
                canon = canonical_method_name(
                    "FedProTrack" if method != "CFL" else "CFL"
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
                ent_str = (
                    f"entropy={row['assignment_entropy']:.4f}"
                    if row["assignment_entropy"] is not None
                    else "entropy=N/A"
                )
                print(
                    f"  -> {acc_str}  {reid_str}  {ent_str}  bytes={row['total_bytes']:.0f}  ({time.time()-t0:.1f}s)",
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
    print("\n=== H2c+H2d Posterior Fix Summary (mean +/- std over seeds) ===", flush=True)
    print(
        f"{'Method':25s}  {'FinalAcc':>14s}  {'Re-ID':>14s}  {'Entropy':>14s}  {'AUC':>14s}  {'Bytes':>8s}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for method in METHODS:
        subset = [r for r in all_rows if r["method"] == method]
        if not subset:
            continue
        accs = [r["final_accuracy"] for r in subset]
        reids = [r["concept_re_id_accuracy"] for r in subset if r["concept_re_id_accuracy"] is not None]
        ents = [r["assignment_entropy"] for r in subset if r["assignment_entropy"] is not None]
        aucs = [r["accuracy_auc"] for r in subset]
        bytes_vals = [r["total_bytes"] for r in subset]

        acc_str = f"{np.mean(accs):.4f}+/-{np.std(accs):.4f}"
        reid_str = f"{np.mean(reids):.4f}+/-{np.std(reids):.4f}" if reids else "N/A"
        ent_str = f"{np.mean(ents):.4f}+/-{np.std(ents):.4f}" if ents else "N/A"
        auc_str = f"{np.mean(aucs):.4f}+/-{np.std(aucs):.4f}"
        bytes_str = f"{np.mean(bytes_vals):.0f}"
        print(
            f"{method:25s}  {acc_str:>14s}  {reid_str:>14s}  {ent_str:>14s}  {auc_str:>14s}  {bytes_str:>8s}",
            flush=True,
        )

    print(f"\nFailures: {len(all_failures)}", flush=True)
    print(f"Results saved to: {results_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
