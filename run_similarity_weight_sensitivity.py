from __future__ import annotations

"""Similarity weight sensitivity analysis for FedProTrack rebuttal (W5).

Tests how sensitive FPT is to the (alpha_1, alpha_2, alpha_3) weights used in
the composite fingerprint similarity Eq. (8):
    sim(f, f') = alpha_1 * feat_sim + alpha_2 * label_sim + alpha_3 * cc_sim

Five weight configurations are swept across 3 seeds, alongside FedAvg and
Oracle baselines for reference.

Outputs:
  - results_weight_sensitivity/weight_sensitivity.json      (per-seed per-config)
  - results_weight_sensitivity/weight_sensitivity_summary.json (aggregated)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior import FedProTrackRunner
from fedprotrack.posterior.presets import make_legacy_config
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

# ---------------------------------------------------------------------------
# Weight configurations to test
# ---------------------------------------------------------------------------
WEIGHT_CONFIGS: dict[str, tuple[float, float, float]] = {
    "Current":      (0.25, 0.30, 0.45),
    "Uniform":      (0.33, 0.33, 0.34),
    "Inverted":     (0.45, 0.30, 0.25),
    "CC-dominant":  (0.10, 0.10, 0.80),
    "No-feat":      (0.00, 0.50, 0.50),
}

SEEDS: list[int] = [42, 43, 44]


def _make_dataset_config(seed: int) -> CIFAR100RecurrenceConfig:
    """Standard CIFAR-100 recurrence config matching main table settings."""
    return CIFAR100RecurrenceConfig(
        K=20,
        T=100,
        n_samples=200,
        rho=25,
        alpha=0.5,
        delta=0.5,
        n_features=128,
        batch_size=64,
        n_workers=0,
        data_root=".cifar100_cache",
        feature_cache_dir=".feature_cache",
        feature_seed=2718,
        samples_per_coarse_class=120,
        seed=seed,
    )


def _make_experiment_log(
    method_name: str,
    accuracy_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    ground_truth: np.ndarray,
    total_bytes: float,
) -> ExperimentLog:
    return ExperimentLog(
        ground_truth=ground_truth,
        predicted=predicted_matrix,
        accuracy_curve=accuracy_matrix,
        total_bytes=total_bytes if total_bytes > 0 else None,
        method_name=method_name,
    )


def run_fpt_with_weights(
    config_name: str,
    alpha_feat: float,
    alpha_label: float,
    alpha_cc: float,
    seed: int,
) -> dict:
    """Run FedProTrack with a specific set of similarity weights."""
    ds_cfg = _make_dataset_config(seed)
    dataset = generate_cifar100_recurrence_dataset(ds_cfg)
    ground_truth = dataset.concept_matrix
    n_concepts = int(ground_truth.max()) + 1

    from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
    config = TwoPhaseConfig(
        omega=2.0,
        kappa=0.7,
        novelty_threshold=0.25,
        loss_novelty_threshold=0.15,
        sticky_dampening=1.5,
        sticky_posterior_gate=0.35,
        merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, n_concepts + 3),
        merge_every=2,
        shrink_every=6,
        fingerprint_sim_weights=(alpha_feat, alpha_label, alpha_cc),
    )

    runner = FedProTrackRunner(
        config=config,
        seed=seed,
        federation_every=1,
        detector_name="ADWIN",
        lr=0.01,
        n_epochs=5,
        soft_aggregation=True,
        blend_alpha=0.0,
    )

    t0 = time.time()
    result = runner.run(dataset)
    elapsed = time.time() - t0

    log = result.to_experiment_log()
    metrics = compute_all_metrics(log, identity_capable=True)

    # Clustering error eta = 1 - concept_re_id_accuracy (when available)
    reid = metrics.concept_re_id_accuracy
    eta = (1.0 - reid) if reid is not None else None

    return {
        "method": f"FPT-{config_name}",
        "config_name": config_name,
        "seed": seed,
        "alpha_feat": alpha_feat,
        "alpha_label": alpha_label,
        "alpha_cc": alpha_cc,
        "final_accuracy": metrics.final_accuracy,
        "accuracy_auc": metrics.accuracy_auc,
        "concept_re_id_accuracy": reid,
        "clustering_error_eta": eta,
        "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
        "assignment_entropy": metrics.assignment_entropy,
        "assignment_switch_rate": metrics.assignment_switch_rate,
        "avg_clients_per_concept": metrics.avg_clients_per_concept,
        "singleton_group_ratio": metrics.singleton_group_ratio,
        "memory_reuse_rate": metrics.memory_reuse_rate,
        "routing_consistency": metrics.routing_consistency,
        "total_bytes": result.total_bytes,
        "wall_clock_s": elapsed,
        "mean_accuracy_curve": result.accuracy_matrix.mean(axis=0).tolist(),
    }


def run_baseline(method: str, seed: int) -> dict:
    """Run FedAvg or Oracle baseline for reference."""
    ds_cfg = _make_dataset_config(seed)
    dataset = generate_cifar100_recurrence_dataset(ds_cfg)
    ground_truth = dataset.concept_matrix

    t0 = time.time()

    if method == "FedAvg":
        exp_cfg = ExperimentConfig(generator_config=dataset.config)
        result = run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=0.01, n_epochs=5, seed=seed,
        )
        log = _make_experiment_log(
            "FedAvg",
            result.accuracy_matrix,
            result.predicted_concept_matrix,
            ground_truth,
            total_bytes=0.0,
        )
        metrics = compute_all_metrics(log, identity_capable=False)
        accuracy_matrix = result.accuracy_matrix
        total_bytes = 0.0
    elif method == "Oracle":
        exp_cfg = ExperimentConfig(generator_config=dataset.config)
        result = run_oracle_baseline(
            exp_cfg, dataset=dataset, lr=0.01, n_epochs=5, seed=seed,
        )
        log = _make_experiment_log(
            "Oracle",
            result.accuracy_matrix,
            result.predicted_concept_matrix,
            ground_truth,
            total_bytes=0.0,
        )
        metrics = compute_all_metrics(log, identity_capable=True)
        accuracy_matrix = result.accuracy_matrix
        total_bytes = 0.0
    else:
        raise ValueError(f"Unknown baseline: {method}")

    elapsed = time.time() - t0
    reid = metrics.concept_re_id_accuracy
    eta = (1.0 - reid) if reid is not None else None

    return {
        "method": method,
        "config_name": method,
        "seed": seed,
        "alpha_feat": None,
        "alpha_label": None,
        "alpha_cc": None,
        "final_accuracy": metrics.final_accuracy,
        "accuracy_auc": metrics.accuracy_auc,
        "concept_re_id_accuracy": reid,
        "clustering_error_eta": eta,
        "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
        "assignment_entropy": metrics.assignment_entropy,
        "assignment_switch_rate": metrics.assignment_switch_rate,
        "avg_clients_per_concept": metrics.avg_clients_per_concept,
        "singleton_group_ratio": metrics.singleton_group_ratio,
        "memory_reuse_rate": metrics.memory_reuse_rate,
        "routing_consistency": metrics.routing_consistency,
        "total_bytes": total_bytes,
        "wall_clock_s": elapsed,
        "mean_accuracy_curve": accuracy_matrix.mean(axis=0).tolist(),
    }


def aggregate_results(
    all_rows: list[dict],
) -> dict[str, dict[str, float | None]]:
    """Aggregate per-seed rows into mean +/- std summaries keyed by method."""
    methods = sorted({row["method"] for row in all_rows})
    summary: dict[str, dict[str, float | None]] = {}

    for method in methods:
        rows = [r for r in all_rows if r["method"] == method]
        n = len(rows)

        entry: dict[str, float | None] = {
            "n_runs": n,
            "alpha_feat": rows[0]["alpha_feat"],
            "alpha_label": rows[0]["alpha_label"],
            "alpha_cc": rows[0]["alpha_cc"],
        }

        for key in (
            "final_accuracy",
            "accuracy_auc",
            "concept_re_id_accuracy",
            "clustering_error_eta",
            "wrong_memory_reuse_rate",
            "assignment_entropy",
            "assignment_switch_rate",
            "avg_clients_per_concept",
            "singleton_group_ratio",
            "memory_reuse_rate",
            "routing_consistency",
        ):
            vals = [float(r[key]) for r in rows if r[key] is not None]
            if vals:
                entry[f"mean_{key}"] = float(np.mean(vals))
                entry[f"std_{key}"] = float(np.std(vals))
            else:
                entry[f"mean_{key}"] = None
                entry[f"std_{key}"] = None

        # Mean accuracy curve
        curves = np.array(
            [r["mean_accuracy_curve"] for r in rows], dtype=np.float64,
        )
        entry["mean_accuracy_curve"] = curves.mean(axis=0).tolist()

        entry["mean_total_bytes"] = float(
            np.mean([r["total_bytes"] for r in rows])
        )
        entry["mean_wall_clock_s"] = float(
            np.mean([r["wall_clock_s"] for r in rows])
        )

        summary[method] = entry

    return summary


def _fmt(val: float | None, width: int = 7, decimals: int = 3) -> str:
    """Format a numeric value for table display."""
    if val is None:
        return "--".center(width)
    return f"{val:.{decimals}f}".rjust(width)


def print_comparison_table(summary: dict[str, dict]) -> None:
    """Print a clear ASCII comparison table."""
    header = (
        f"{'Method':<20s} "
        f"{'alpha':>14s}  "
        f"{'Acc':>7s}  "
        f"{'AUC':>7s}  "
        f"{'Re-ID':>7s}  "
        f"{'eta':>7s}  "
        f"{'WMRR':>7s}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print("  SIMILARITY WEIGHT SENSITIVITY — CIFAR-100 RECURRENCE")
    print(f"  K=20, T=100, rho=25, seeds={SEEDS}")
    print(sep)
    print(header)
    print(sep)

    # Print baselines first, then FPT variants
    baseline_keys = [k for k in summary if not k.startswith("FPT-")]
    fpt_keys = [k for k in summary if k.startswith("FPT-")]

    for key in baseline_keys + fpt_keys:
        entry = summary[key]
        alpha_str = (
            f"({entry['alpha_feat']:.2f},{entry['alpha_label']:.2f},{entry['alpha_cc']:.2f})"
            if entry.get("alpha_feat") is not None
            else "--"
        )
        line = (
            f"{key:<20s} "
            f"{alpha_str:>14s}  "
            f"{_fmt(entry.get('mean_final_accuracy'))}  "
            f"{_fmt(entry.get('mean_accuracy_auc'))}  "
            f"{_fmt(entry.get('mean_concept_re_id_accuracy'))}  "
            f"{_fmt(entry.get('mean_clustering_error_eta'))}  "
            f"{_fmt(entry.get('mean_wrong_memory_reuse_rate'))}"
        )
        print(line)

    print(sep)

    # Compute and display range across FPT variants
    fpt_accs = [
        summary[k]["mean_final_accuracy"]
        for k in fpt_keys
        if summary[k].get("mean_final_accuracy") is not None
    ]
    if len(fpt_accs) >= 2:
        acc_range = max(fpt_accs) - min(fpt_accs)
        print(f"\n  FPT accuracy range across weight configs: {acc_range:.4f}")
        print(f"  FPT accuracy min: {min(fpt_accs):.4f}  max: {max(fpt_accs):.4f}")

    fpt_reids = [
        summary[k]["mean_concept_re_id_accuracy"]
        for k in fpt_keys
        if summary[k].get("mean_concept_re_id_accuracy") is not None
    ]
    if len(fpt_reids) >= 2:
        reid_range = max(fpt_reids) - min(fpt_reids)
        print(f"  FPT re-ID range across weight configs: {reid_range:.4f}")
        print(f"  FPT re-ID min: {min(fpt_reids):.4f}  max: {max(fpt_reids):.4f}")

    print()


def main() -> None:
    out_dir = Path("results_weight_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm the feature cache once
    print("Warming CIFAR-100 feature cache...", flush=True)
    warm_cfg = _make_dataset_config(SEEDS[0])
    prepare_cifar100_recurrence_feature_cache(warm_cfg)
    print("Feature cache ready.\n", flush=True)

    all_rows: list[dict] = []
    total_tasks = len(WEIGHT_CONFIGS) * len(SEEDS) + 2 * len(SEEDS)
    task_idx = 0

    # --- Run baselines ---
    for method in ("FedAvg", "Oracle"):
        for seed in SEEDS:
            task_idx += 1
            print(
                f"[{task_idx}/{total_tasks}] {method} seed={seed}",
                flush=True,
            )
            row = run_baseline(method, seed)
            all_rows.append(row)
            print(
                f"  -> final_acc={row['final_accuracy']:.4f}  "
                f"auc={row['accuracy_auc']:.4f}  "
                f"reid={_fmt(row['concept_re_id_accuracy'])}  "
                f"eta={_fmt(row['clustering_error_eta'])}",
                flush=True,
            )

    # --- Run FPT weight configurations ---
    for config_name, (a1, a2, a3) in WEIGHT_CONFIGS.items():
        for seed in SEEDS:
            task_idx += 1
            print(
                f"[{task_idx}/{total_tasks}] FPT-{config_name} "
                f"(alpha={a1:.2f},{a2:.2f},{a3:.2f}) seed={seed}",
                flush=True,
            )
            row = run_fpt_with_weights(config_name, a1, a2, a3, seed)
            all_rows.append(row)
            print(
                f"  -> final_acc={row['final_accuracy']:.4f}  "
                f"auc={row['accuracy_auc']:.4f}  "
                f"reid={_fmt(row['concept_re_id_accuracy'])}  "
                f"eta={_fmt(row['clustering_error_eta'])}",
                flush=True,
            )

    # --- Aggregate ---
    summary = aggregate_results(all_rows)

    # --- Save outputs ---
    with open(out_dir / "weight_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, default=str)

    with open(
        out_dir / "weight_sensitivity_summary.json", "w", encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2, default=str)

    # --- Print comparison table ---
    print_comparison_table(summary)

    total_time = sum(r["wall_clock_s"] for r in all_rows)
    print(
        f"All {total_tasks} runs completed in {total_time:.1f}s total. "
        f"Results saved to {out_dir}/",
        flush=True,
    )


if __name__ == "__main__":
    main()
