"""Run a focused CIFAR-100 comparison that highlights FedProTrack.

The benchmark uses official CIFAR-100 coarse labels (20 superclasses) and
recurring concept shifts induced by deterministic appearance transforms.
Per-concept feature pools are extracted once with pretrained ResNet18,
cached on disk, and then reused by all workers.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fedprotrack.experiment.baselines import run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.baselines.runners import run_fedproto_full, run_ifca_full
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior import (
    FEDPROTRACK_VARIANTS,
    FedProTrackRunner,
    make_variant_bundle,
)
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


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


def _build_dataset_config(base_cfg: dict, seed: int) -> CIFAR100RecurrenceConfig:
    cfg = dict(base_cfg)
    cfg["seed"] = seed
    return CIFAR100RecurrenceConfig(**cfg)


def _run_task(task: dict) -> dict:
    os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

    method = task["method"]
    seed = task["seed"]
    ds_cfg = _build_dataset_config(task["dataset_config"], seed)
    dataset = generate_cifar100_recurrence_dataset(ds_cfg)
    ground_truth = dataset.concept_matrix

    t0 = time.time()
    if method == task["fedprotrack_method_name"]:
        n_concepts = int(ground_truth.max()) + 1
        _, config, runner_kwargs = make_variant_bundle(
            task["fedprotrack_variant"],
            config_overrides={
                "max_concepts": max(6, n_concepts + 2),
                "shrink_every": 6,
            },
            runner_overrides={
                "federation_every": task["federation_every"],
                "detector_name": task["detector_name"],
                "lr": task["fpt_lr"],
                "n_epochs": task["fpt_epochs"],
            },
        )
        runner = FedProTrackRunner(config=config, seed=seed, **runner_kwargs)
        result = runner.run(dataset)
        log = result.to_experiment_log()
        metrics = compute_all_metrics(log, identity_capable=True)
        accuracy_matrix = result.accuracy_matrix
        total_bytes = result.total_bytes
    elif method == "IFCA":
        result = run_ifca_full(
            dataset,
            federation_every=task["federation_every"],
            n_clusters=task["ifca_clusters"],
        )
        log = result.to_experiment_log(ground_truth)
        metrics = compute_all_metrics(log, identity_capable=True)
        accuracy_matrix = result.accuracy_matrix
        total_bytes = result.total_bytes
    elif method == "FedProto":
        result = run_fedproto_full(
            dataset,
            federation_every=task["federation_every"],
        )
        log = result.to_experiment_log(ground_truth)
        metrics = compute_all_metrics(log, identity_capable=False)
        accuracy_matrix = result.accuracy_matrix
        total_bytes = result.total_bytes
    elif method == "FedAvg":
        exp_cfg = ExperimentConfig(generator_config=dataset.config)
        result = run_fedavg_baseline(exp_cfg, dataset=dataset)
        log = _make_log(
            "FedAvg",
            result.accuracy_matrix,
            result.predicted_concept_matrix,
            ground_truth,
            total_bytes=0.0,
        )
        metrics = compute_all_metrics(log, identity_capable=False)
        accuracy_matrix = result.accuracy_matrix
        total_bytes = 0.0
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0
    return {
        "method": method,
        "seed": seed,
        "final_accuracy": metrics.final_accuracy,
        "accuracy_auc": metrics.accuracy_auc,
        "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
        "assignment_switch_rate": metrics.assignment_switch_rate,
        "avg_clients_per_concept": metrics.avg_clients_per_concept,
        "singleton_group_ratio": metrics.singleton_group_ratio,
        "memory_reuse_rate": metrics.memory_reuse_rate,
        "routing_consistency": metrics.routing_consistency,
        "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
        "assignment_entropy": metrics.assignment_entropy,
        "total_bytes": total_bytes,
        "wall_clock_s": elapsed,
        "mean_accuracy_curve": accuracy_matrix.mean(axis=0).tolist(),
    }


def _write_raw_csv(rows: list[dict], path: Path) -> None:
    header = [
        "method",
        "seed",
        "final_accuracy",
        "accuracy_auc",
        "concept_re_id_accuracy",
        "assignment_switch_rate",
        "avg_clients_per_concept",
        "singleton_group_ratio",
        "memory_reuse_rate",
        "routing_consistency",
        "wrong_memory_reuse_rate",
        "assignment_entropy",
        "total_bytes",
        "wall_clock_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in header})


def _aggregate(rows: list[dict]) -> dict[str, dict[str, float | list[float]]]:
    summary: dict[str, dict[str, float | list[float]]] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        curves = np.array(
            [row["mean_accuracy_curve"] for row in method_rows], dtype=np.float64
        )
        entry: dict[str, float | list[float]] = {
            "n_runs": len(method_rows),
            "mean_final_accuracy": float(
                np.mean([row["final_accuracy"] for row in method_rows])
            ),
            "std_final_accuracy": float(
                np.std([row["final_accuracy"] for row in method_rows])
            ),
            "mean_accuracy_auc": float(
                np.mean([row["accuracy_auc"] for row in method_rows])
            ),
            "std_accuracy_auc": float(
                np.std([row["accuracy_auc"] for row in method_rows])
            ),
            "mean_total_bytes": float(
                np.mean([row["total_bytes"] for row in method_rows])
            ),
            "mean_wall_clock_s": float(
                np.mean([row["wall_clock_s"] for row in method_rows])
            ),
            "mean_accuracy_curve": curves.mean(axis=0).tolist(),
        }

        reid_vals = [
            float(row["concept_re_id_accuracy"])
            for row in method_rows
            if row["concept_re_id_accuracy"] is not None
        ]
        if reid_vals:
            entry["mean_concept_re_id_accuracy"] = float(np.mean(reid_vals))
            entry["std_concept_re_id_accuracy"] = float(np.std(reid_vals))

        wrong_vals = [
            float(row["wrong_memory_reuse_rate"])
            for row in method_rows
            if row["wrong_memory_reuse_rate"] is not None
        ]
        if wrong_vals:
            entry["mean_wrong_memory_reuse_rate"] = float(np.mean(wrong_vals))

        ent_vals = [
            float(row["assignment_entropy"])
            for row in method_rows
            if row["assignment_entropy"] is not None
        ]
        if ent_vals:
            entry["mean_assignment_entropy"] = float(np.mean(ent_vals))

        for source_key in (
            "assignment_switch_rate",
            "avg_clients_per_concept",
            "singleton_group_ratio",
            "memory_reuse_rate",
            "routing_consistency",
        ):
            vals = [
                float(row[source_key])
                for row in method_rows
                if row[source_key] is not None
            ]
            if vals:
                entry[f"mean_{source_key}"] = float(np.mean(vals))

        summary[method] = entry
    return summary


def _plot_accuracy_curves(summary: dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, entry in summary.items():
        curve = np.asarray(entry["mean_accuracy_curve"], dtype=np.float64)
        ax.plot(range(len(curve)), curve, marker="o", linewidth=2, label=method)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("CIFAR-100 Recurrence Accuracy Trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_bars(summary: dict[str, dict], output_path: Path) -> None:
    methods = list(summary.keys())
    metrics = [
        ("mean_final_accuracy", "Final Accuracy"),
        ("mean_accuracy_auc", "Accuracy AUC"),
        ("mean_concept_re_id_accuracy", "Concept Re-ID Accuracy"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, metrics):
        vals = [summary[m].get(key, np.nan) for m in methods]
        ax.bar(
            range(len(methods)),
            vals,
            color=plt.cm.tab10(np.linspace(0, 1, len(methods))),
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused CIFAR-100 comparison")
    parser.add_argument("--results-dir", default="results_cifar100")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--methods", default="FedProTrack,IFCA,FedProto,FedAvg")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.85)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--samples-per-coarse-class", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--federation-every", type=int, default=1)
    parser.add_argument("--ifca-clusters", type=int, default=4)
    parser.add_argument("--detector-name", default="ADWIN")
    parser.add_argument("--fpt-lr", type=float, default=0.1)
    parser.add_argument("--fpt-epochs", type=int, default=10)
    parser.add_argument(
        "--fedprotrack-variant",
        choices=FEDPROTRACK_VARIANTS,
        default="legacy",
        help="FedProTrack preset to run when methods include FedProTrack.",
    )
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.K = 4
        args.T = 8
        args.n_samples = 240
        args.rho = 2.0
        args.alpha = 0.75
        args.delta = 0.9
        args.n_features = 96
        args.samples_per_coarse_class = 60
        args.batch_size = 192

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    fedprotrack_method_name, _, _ = make_variant_bundle(args.fedprotrack_variant)
    methods = [
        fedprotrack_method_name if part.strip().startswith("FedProTrack") else part.strip()
        for part in args.methods.split(",")
        if part.strip()
    ]

    dataset_config = {
        "K": args.K,
        "T": args.T,
        "n_samples": args.n_samples,
        "rho": args.rho,
        "alpha": args.alpha,
        "delta": args.delta,
        "n_features": args.n_features,
        "samples_per_coarse_class": args.samples_per_coarse_class,
        "batch_size": args.batch_size,
        "n_workers": args.n_workers,
        "data_root": args.data_root,
        "feature_cache_dir": args.feature_cache_dir,
        "feature_seed": args.feature_seed,
    }

    print("Warming CIFAR-100 feature cache...", flush=True)
    warm_cfg = _build_dataset_config(dataset_config, seeds[0])
    prepare_cifar100_recurrence_feature_cache(warm_cfg)

    tasks = [
        {
            "method": method,
            "seed": seed,
            "dataset_config": dataset_config,
            "federation_every": args.federation_every,
            "ifca_clusters": args.ifca_clusters,
            "detector_name": args.detector_name,
            "fpt_lr": args.fpt_lr,
            "fpt_epochs": args.fpt_epochs,
            "fedprotrack_variant": args.fedprotrack_variant,
            "fedprotrack_method_name": fedprotrack_method_name,
        }
        for seed in seeds
        for method in methods
    ]

    print(
        f"Running {len(tasks)} tasks on {args.max_workers} worker(s): "
        f"methods={methods}, seeds={seeds}, "
        f"fedprotrack_variant={args.fedprotrack_variant}",
        flush=True,
    )

    raw_rows: list[dict] = []
    t0 = time.time()
    if args.max_workers <= 1:
        for task in tasks:
            row = _run_task(task)
            raw_rows.append(row)
            print(
                f"  {row['method']} seed={row['seed']} "
                f"final_acc={row['final_accuracy']:.4f} "
                f"auc={row['accuracy_auc']:.4f} "
                f"reid={row['concept_re_id_accuracy'] if row['concept_re_id_accuracy'] is not None else '--'} "
                f"switch={row['assignment_switch_rate'] if row['assignment_switch_rate'] is not None else '--'} "
                f"singleton={row['singleton_group_ratio'] if row['singleton_group_ratio'] is not None else '--'}",
                flush=True,
            )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.max_workers,
            mp_context=ctx,
        ) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                raw_rows.append(row)
                print(
                    f"  {row['method']} seed={row['seed']} "
                    f"final_acc={row['final_accuracy']:.4f} "
                    f"auc={row['accuracy_auc']:.4f} "
                    f"reid={row['concept_re_id_accuracy'] if row['concept_re_id_accuracy'] is not None else '--'} "
                    f"switch={row['assignment_switch_rate'] if row['assignment_switch_rate'] is not None else '--'} "
                    f"singleton={row['singleton_group_ratio'] if row['singleton_group_ratio'] is not None else '--'}",
                    flush=True,
                )

    elapsed = time.time() - t0
    summary = _aggregate(raw_rows)

    _write_raw_csv(raw_rows, results_dir / "raw_results.csv")
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_config": dataset_config,
                "seeds": seeds,
                "methods": methods,
                "fedprotrack_variant": args.fedprotrack_variant,
                "federation_every": args.federation_every,
                "ifca_clusters": args.ifca_clusters,
                "detector_name": args.detector_name,
                "fpt_lr": args.fpt_lr,
                "fpt_epochs": args.fpt_epochs,
            },
            f,
            indent=2,
        )

    _plot_accuracy_curves(summary, results_dir / "accuracy_curves.png")
    _plot_metric_bars(summary, results_dir / "comparison_metrics.png")

    print("\nSummary", flush=True)
    for method, entry in summary.items():
        line = (
            f"  {method:12s} final={entry['mean_final_accuracy']:.4f} "
            f"auc={entry['mean_accuracy_auc']:.4f}"
        )
        if "mean_concept_re_id_accuracy" in entry:
            line += f" reid={entry['mean_concept_re_id_accuracy']:.4f}"
        if "mean_assignment_switch_rate" in entry:
            line += f" switch={entry['mean_assignment_switch_rate']:.4f}"
        if "mean_singleton_group_ratio" in entry:
            line += f" singleton={entry['mean_singleton_group_ratio']:.4f}"
        print(line, flush=True)

    print(
        f"\nCompleted in {elapsed:.1f}s. Results saved to {results_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
