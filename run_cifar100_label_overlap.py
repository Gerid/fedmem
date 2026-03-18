from __future__ import annotations

"""Parameterized CIFAR-100 label-overlap sweep for failure diagnosis."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fedprotrack.baselines.runners import run_ifca_full
from fedprotrack.experiments.cifar_overlap import (
    CIFARSubsetBenchmarkConfig,
    build_overlap_concept_classes,
    build_subset_dataset,
    run_fedavg,
    run_local,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior import make_plan_c_config

os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")


def _make_log(
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


def _write_raw_csv(rows: list[dict], path: Path) -> None:
    header = [
        "method",
        "seed",
        "overlap_ratio",
        "final_accuracy",
        "accuracy_auc",
        "concept_re_id_accuracy",
        "assignment_entropy",
        "assignment_switch_rate",
        "avg_clients_per_concept",
        "singleton_group_ratio",
        "memory_reuse_rate",
        "routing_consistency",
        "wrong_memory_reuse_rate",
        "total_bytes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def _aggregate_by_overlap(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    summary: dict[str, dict[str, dict[str, float]]] = {}
    methods = sorted({str(row["method"]) for row in rows})
    overlap_values = sorted({float(row["overlap_ratio"]) for row in rows})
    metrics = [
        "final_accuracy",
        "concept_re_id_accuracy",
        "assignment_entropy",
        "assignment_switch_rate",
        "singleton_group_ratio",
        "routing_consistency",
    ]

    for method in methods:
        by_overlap: dict[str, dict[str, float]] = {}
        for overlap in overlap_values:
            subset = [
                row for row in rows
                if str(row["method"]) == method and float(row["overlap_ratio"]) == overlap
            ]
            if not subset:
                continue
            entry = {"n_runs": float(len(subset))}
            for metric in metrics:
                vals = [
                    float(row[metric]) for row in subset
                    if row.get(metric) is not None and row.get(metric) != ""
                ]
                if vals:
                    entry[f"mean_{metric}"] = float(np.mean(vals))
                    entry[f"std_{metric}"] = float(np.std(vals))
            by_overlap[f"{overlap:.2f}"] = entry
        summary[method] = by_overlap
    return summary


def _plot_metric_vs_overlap(rows: list[dict], output_path: Path) -> None:
    methods = ["FedProTrack", "IFCA", "FedAvg", "LocalOnly"]
    overlap_values = sorted({float(row["overlap_ratio"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    metrics = [
        ("final_accuracy", "Final Acc"),
        ("concept_re_id_accuracy", "Re-ID Acc"),
        ("assignment_switch_rate", "Switch Rate"),
        ("singleton_group_ratio", "Singleton Ratio"),
    ]
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for method in methods:
            means = []
            for overlap in overlap_values:
                vals = [
                    float(row[metric]) for row in rows
                    if str(row["method"]) == method
                    and float(row["overlap_ratio"]) == overlap
                    and row.get(metric) not in ("", None)
                ]
                means.append(float(np.mean(vals)) if vals else np.nan)
            ax.plot(overlap_values, means, marker="o", label=method)
        ax.set_xlabel("Overlap Ratio")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_cifar100_label_overlap")
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--federation-every", type=int, default=2)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    phase_concepts = [[0, 1], [2, 3], [0, 1]]

    all_rows: list[dict] = []
    for overlap_ratio in args.overlaps:
        concept_classes = build_overlap_concept_classes(overlap_ratio)
        for seed in args.seeds:
            ds = build_subset_dataset(
                CIFARSubsetBenchmarkConfig(
                    K=args.K,
                    T=args.T,
                    n_samples=args.n_samples,
                    n_features=args.n_features,
                    seed=seed,
                    generator_type="cifar100_label_overlap",
                ),
                concept_classes=concept_classes,
                phase_concepts=phase_concepts,
            )
            gt = ds.concept_matrix

            local_acc, local_bytes = run_local(ds, args.epochs, args.lr, seed)
            local_log = _make_log(
                "LocalOnly",
                local_acc,
                np.zeros_like(gt, dtype=np.int32),
                gt,
                local_bytes,
            )
            local_metrics = compute_all_metrics(local_log, identity_capable=False)

            fedavg_acc, fedavg_bytes = run_fedavg(
                ds, args.federation_every, args.epochs, args.lr, seed
            )
            fedavg_log = _make_log(
                "FedAvg",
                fedavg_acc,
                np.zeros_like(gt, dtype=np.int32),
                gt,
                fedavg_bytes,
            )
            fedavg_metrics = compute_all_metrics(fedavg_log, identity_capable=False)

            ifca_res = run_ifca_full(
                ds,
                federation_every=args.federation_every,
                n_clusters=4,
                lr=args.lr,
                n_epochs=args.epochs,
            )
            ifca_metrics = compute_all_metrics(
                ifca_res.to_experiment_log(gt),
                identity_capable=True,
            )

            from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner

            runner = FedProTrackRunner(
                config=make_plan_c_config(
                    loss_novelty_threshold=0.25,
                    merge_threshold=0.80,
                    max_concepts=4,
                ),
                federation_every=args.federation_every,
                detector_name="ADWIN",
                seed=seed,
                lr=args.lr,
                n_epochs=args.epochs,
                soft_aggregation=True,
                blend_alpha=0.0,
                model_type="feature_adapter",
                hidden_dim=64,
                adapter_dim=16,
            )
            fpt_result = runner.run(ds)
            fpt_metrics = compute_all_metrics(
                fpt_result.to_experiment_log(),
                identity_capable=True,
            )

            metric_map = {
                "LocalOnly": (local_metrics, local_bytes),
                "FedAvg": (fedavg_metrics, fedavg_bytes),
                "IFCA": (ifca_metrics, ifca_res.total_bytes),
                "FedProTrack": (fpt_metrics, fpt_result.total_bytes),
            }
            for method, (metrics, total_bytes) in metric_map.items():
                all_rows.append({
                    "method": method,
                    "seed": seed,
                    "overlap_ratio": overlap_ratio,
                    "final_accuracy": metrics.final_accuracy,
                    "accuracy_auc": metrics.accuracy_auc,
                    "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                    "assignment_entropy": metrics.assignment_entropy,
                    "assignment_switch_rate": metrics.assignment_switch_rate,
                    "avg_clients_per_concept": metrics.avg_clients_per_concept,
                    "singleton_group_ratio": metrics.singleton_group_ratio,
                    "memory_reuse_rate": metrics.memory_reuse_rate,
                    "routing_consistency": metrics.routing_consistency,
                    "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
                    "total_bytes": total_bytes,
                })

            print(
                f"overlap={overlap_ratio:.2f} seed={seed} "
                f"FPT-final={fpt_metrics.final_accuracy:.4f} "
                f"IFCA-final={ifca_metrics.final_accuracy:.4f}"
            )

    _write_raw_csv(all_rows, results_dir / "raw_results.csv")
    summary = _aggregate_by_overlap(all_rows)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _plot_metric_vs_overlap(all_rows, results_dir / "diagnostics_vs_overlap.png")
    print(f"Saved overlap sweep outputs to {results_dir}")


if __name__ == "__main__":
    main()
