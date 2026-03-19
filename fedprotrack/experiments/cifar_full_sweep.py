from __future__ import annotations

"""Full CIFAR-100 recurrence sweep with tuned linear-split FedProTrack."""

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

from ..baselines.runners import (
    MethodResult,
    run_apfl_full,
    run_atp_full,
    run_cfl_full,
    run_compressed_fedavg_full,
    run_fedavg_full,
    run_fedccfa_full,
    run_feddrift_full,
    run_fedem_full,
    run_fedproto_full,
    run_fedrc_full,
    run_fesem_full,
    run_flash_full,
    run_flux_full,
    run_flux_prior_full,
    run_ifca_full,
    run_local_only_full,
    run_oracle_full,
    run_pfedme_full,
    run_tracked_summary_full,
)
from ..metrics import compute_all_metrics
from ..metrics.experiment_log import MetricsResult
from ..posterior import FedProTrackRunner, make_plan_c_config
from ..real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)
from .method_registry import (
    FOCUSED_CIFAR_SWEEP_METHODS,
    FULL_CIFAR_SWEEP_METHODS,
    identity_metrics_valid,
)
from .tables import export_summary_csv, generate_main_table


def available_cifar_sweep_methods() -> tuple[str, ...]:
    """Return all supported method names for the recurrence sweep."""
    return (
        "FedProTrack-linear-split",
        "FedProTrack-feature-adapter",
        "FedAvg",
        "CompressedFedAvg",
        "FedProto",
        "pFedMe",
        "APFL",
        "FedEM-3",
        "IFCA-3",
        "IFCA-8",
        "CFL",
        "FeSEM-3",
        "FedRC-3",
        "FedCCFA",
        "FedDrift",
        "TrackedSummary",
        "Flash",
        "ATP",
        "FLUX",
        "FLUX-prior",
        "LocalOnly",
        "Oracle",
    )


def resolve_cifar_sweep_methods(methods: str | list[str]) -> list[str]:
    """Resolve CLI method selectors into an ordered method list."""
    if isinstance(methods, str):
        selector = methods.strip()
        lowered = selector.lower()
        if lowered == "all":
            return list(FULL_CIFAR_SWEEP_METHODS)
        if lowered == "focused":
            return list(FOCUSED_CIFAR_SWEEP_METHODS)
        requested = [part.strip() for part in selector.split(",") if part.strip()]
    else:
        requested = [str(part).strip() for part in methods if str(part).strip()]

    known = set(available_cifar_sweep_methods())
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(
            f"Unknown CIFAR sweep methods: {unknown}. "
            f"Choose from {sorted(known)} or selectors all/focused."
        )
    return requested


def _build_dataset_config(base_cfg: dict, seed: int) -> CIFAR100RecurrenceConfig:
    cfg = dict(base_cfg)
    cfg["seed"] = seed
    return CIFAR100RecurrenceConfig(**cfg)


def _metrics_to_result(metrics_dict: dict) -> MetricsResult:
    """Rebuild a lightweight MetricsResult for table/export helpers."""
    return MetricsResult(
        concept_re_id_accuracy=metrics_dict.get("concept_re_id_accuracy"),
        assignment_entropy=metrics_dict.get("assignment_entropy"),
        assignment_switch_rate=metrics_dict.get("assignment_switch_rate"),
        avg_clients_per_concept=metrics_dict.get("avg_clients_per_concept"),
        singleton_group_ratio=metrics_dict.get("singleton_group_ratio"),
        memory_reuse_rate=metrics_dict.get("memory_reuse_rate"),
        routing_consistency=metrics_dict.get("routing_consistency"),
        wrong_memory_reuse_rate=metrics_dict.get("wrong_memory_reuse_rate"),
        worst_window_dip=metrics_dict.get("worst_window_dip"),
        worst_window_recovery=metrics_dict.get("worst_window_recovery"),
        budget_normalized_score=metrics_dict.get("budget_normalized_score"),
        per_client_re_id=None,
        per_timestep_re_id=None,
        final_accuracy=metrics_dict.get("final_accuracy"),
        accuracy_auc=metrics_dict.get("accuracy_auc"),
    )


def _result_to_log_and_stats(
    method_name: str,
    result,
    ground_truth: np.ndarray,
) -> tuple[MetricsResult, np.ndarray, float, dict[str, float | int | None]]:
    if hasattr(result, "to_experiment_log"):
        if isinstance(result, MethodResult):
            log = result.to_experiment_log(ground_truth)
        else:
            log = result.to_experiment_log()
        accuracy_matrix = np.asarray(result.accuracy_matrix, dtype=np.float64)
        total_bytes = float(result.total_bytes)
    else:
        raise TypeError(f"Unsupported result type for {method_name}: {type(result)!r}")

    metrics = compute_all_metrics(
        log,
        identity_capable=identity_metrics_valid(method_name),
    )
    extra_stats = {
        "phase_a_bytes": float(getattr(result, "phase_a_bytes", 0.0))
        if hasattr(result, "phase_a_bytes") else None,
        "phase_b_bytes": float(getattr(result, "phase_b_bytes", 0.0))
        if hasattr(result, "phase_b_bytes") else None,
        "spawned_concepts": int(getattr(result, "spawned_concepts", 0))
        if hasattr(result, "spawned_concepts") else None,
        "merged_concepts": int(getattr(result, "merged_concepts", 0))
        if hasattr(result, "merged_concepts") else None,
        "pruned_concepts": int(getattr(result, "pruned_concepts", 0))
        if hasattr(result, "pruned_concepts") else None,
        "active_concepts": int(getattr(result, "active_concepts", 0))
        if hasattr(result, "active_concepts") else None,
    }
    return metrics, accuracy_matrix, total_bytes, extra_stats


def _run_one_method(
    method_name: str,
    dataset,
    ground_truth: np.ndarray,
    task: dict,
) -> dict:
    t0 = time.time()
    seed = int(task["seed"])
    fed_every = int(task["federation_every"])

    if method_name == "FedProTrack-linear-split":
        n_concepts = int(ground_truth.max()) + 1
        result = FedProTrackRunner(
            config=make_plan_c_config(
                loss_novelty_threshold=float(task["fpt_linear_loss_novelty_threshold"]),
                merge_threshold=float(task["fpt_linear_merge_threshold"]),
                max_concepts=max(
                    int(task["fpt_linear_max_concepts"]),
                    n_concepts,
                ),
            ),
            federation_every=fed_every,
            detector_name=str(task["detector_name"]),
            seed=seed,
            lr=float(task["fpt_linear_lr"]),
            n_epochs=int(task["fpt_linear_epochs"]),
            soft_aggregation=True,
            blend_alpha=0.0,
            dormant_recall=bool(task["fpt_linear_dormant_recall"]),
            model_type="linear",
        ).run(dataset)
    elif method_name == "FedProTrack-feature-adapter":
        n_concepts = int(ground_truth.max()) + 1
        result = FedProTrackRunner(
            config=make_plan_c_config(
                loss_novelty_threshold=float(task["fpt_adapter_loss_novelty_threshold"]),
                merge_threshold=float(task["fpt_adapter_merge_threshold"]),
                max_concepts=max(
                    int(task["fpt_adapter_max_concepts"]),
                    n_concepts,
                ),
                global_shared_aggregation=bool(task["fpt_adapter_global_shared_aggregation"]),
            ),
            federation_every=fed_every,
            detector_name=str(task["detector_name"]),
            seed=seed,
            lr=float(task["fpt_adapter_lr"]),
            n_epochs=int(task["fpt_adapter_epochs"]),
            soft_aggregation=True,
            blend_alpha=0.0,
            dormant_recall=bool(task["fpt_adapter_dormant_recall"]),
            model_type="feature_adapter",
            hidden_dim=int(task["fpt_adapter_hidden_dim"]),
            adapter_dim=int(task["fpt_adapter_dim"]),
            routed_local_training=bool(task["fpt_adapter_routed_local_training"]),
        ).run(dataset)
    elif method_name == "FedAvg":
        result = run_fedavg_full(
            dataset,
            federation_every=fed_every,
            lr=float(task["baseline_lr"]),
            n_epochs=int(task["baseline_epochs"]),
            seed=seed,
        )
    elif method_name == "CompressedFedAvg":
        result = run_compressed_fedavg_full(
            dataset,
            federation_every=fed_every,
            topk_fraction=float(task["compressed_topk_fraction"]),
        )
    elif method_name == "FedProto":
        result = run_fedproto_full(dataset, federation_every=fed_every)
    elif method_name == "pFedMe":
        result = run_pfedme_full(
            dataset,
            federation_every=fed_every,
            local_epochs=int(task["pfedme_local_epochs"]),
            K_steps=int(task["pfedme_k_steps"]),
            lamda=float(task["pfedme_lamda"]),
            personal_learning_rate=float(task["pfedme_personal_learning_rate"]),
        )
    elif method_name == "APFL":
        result = run_apfl_full(
            dataset,
            federation_every=fed_every,
            alpha=float(task["apfl_alpha"]),
            alpha_lr=float(task["apfl_alpha_lr"]),
            local_steps=int(task["apfl_local_steps"]),
        )
    elif method_name == "FedEM-3":
        result = run_fedem_full(
            dataset,
            federation_every=fed_every,
            n_components=3,
            local_epochs=int(task["fedem_local_epochs"]),
        )
    elif method_name == "IFCA-3":
        result = run_ifca_full(
            dataset,
            federation_every=fed_every,
            n_clusters=3,
            lr=float(task["ifca_lr"]),
            n_epochs=int(task["ifca_epochs"]),
        )
    elif method_name == "IFCA-8":
        result = run_ifca_full(
            dataset,
            federation_every=fed_every,
            n_clusters=8,
            lr=float(task["ifca_lr"]),
            n_epochs=int(task["ifca_epochs"]),
        )
    elif method_name == "CFL":
        result = run_cfl_full(
            dataset,
            federation_every=fed_every,
            eps_1=float(task["cfl_eps_1"]),
            eps_2=float(task["cfl_eps_2"]),
            warmup_rounds=int(task["cfl_warmup_rounds"]),
            max_clusters=int(task["cfl_max_clusters"]),
        )
    elif method_name == "FeSEM-3":
        result = run_fesem_full(
            dataset,
            federation_every=fed_every,
            n_clusters=3,
        )
    elif method_name == "FedRC-3":
        result = run_fedrc_full(
            dataset,
            federation_every=fed_every,
            n_clusters=3,
        )
    elif method_name == "FedCCFA":
        result = run_fedccfa_full(
            dataset,
            federation_every=fed_every,
            cluster_eps=float(task["fedccfa_cluster_eps"]),
            reid_similarity_threshold=float(task["fedccfa_reid_similarity_threshold"]),
            prototype_mix=float(task["fedccfa_prototype_mix"]),
        )
    elif method_name == "FedDrift":
        result = run_feddrift_full(
            dataset,
            federation_every=fed_every,
            similarity_threshold=float(task["feddrift_similarity_threshold"]),
        )
    elif method_name == "TrackedSummary":
        result = run_tracked_summary_full(
            dataset,
            federation_every=fed_every,
            similarity_threshold=float(task["tracked_summary_similarity_threshold"]),
        )
    elif method_name == "Flash":
        result = run_flash_full(
            dataset,
            federation_every=fed_every,
            distill_alpha=float(task["flash_distill_alpha"]),
        )
    elif method_name == "ATP":
        result = run_atp_full(
            dataset,
            federation_every=fed_every,
            base_lr=float(task["atp_base_lr"]),
            meta_lr=float(task["atp_meta_lr"]),
        )
    elif method_name == "FLUX":
        result = run_flux_full(dataset, federation_every=fed_every)
    elif method_name == "FLUX-prior":
        result = run_flux_prior_full(
            dataset,
            federation_every=fed_every,
            n_clusters=int(task["flux_prior_clusters"]),
        )
    elif method_name == "LocalOnly":
        result = run_local_only_full(
            dataset,
            lr=float(task["baseline_lr"]),
            n_epochs=int(task["baseline_epochs"]),
            seed=seed,
        )
    elif method_name == "Oracle":
        result = run_oracle_full(
            dataset,
            federation_every=fed_every,
            lr=float(task["baseline_lr"]),
            n_epochs=int(task["baseline_epochs"]),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

    metrics, accuracy_matrix, total_bytes, extra_stats = _result_to_log_and_stats(
        method_name,
        result,
        ground_truth,
    )
    elapsed = time.time() - t0

    row = {
        "method": method_name,
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
        "worst_window_dip": metrics.worst_window_dip,
        "worst_window_recovery": metrics.worst_window_recovery,
        "budget_normalized_score": metrics.budget_normalized_score,
        "total_bytes": total_bytes,
        "wall_clock_s": elapsed,
        "mean_accuracy_curve": accuracy_matrix.mean(axis=0).tolist(),
    }
    row.update(extra_stats)
    return row


def _run_seed_bundle(task: dict) -> list[dict]:
    os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

    seed = int(task["seed"])
    ds_cfg = _build_dataset_config(task["dataset_config"], seed)
    dataset = generate_cifar100_recurrence_dataset(ds_cfg)
    ground_truth = dataset.concept_matrix

    return [
        _run_one_method(method_name, dataset, ground_truth, task)
        for method_name in task["methods"]
    ]


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
        "worst_window_dip",
        "worst_window_recovery",
        "budget_normalized_score",
        "total_bytes",
        "wall_clock_s",
        "phase_a_bytes",
        "phase_b_bytes",
        "spawned_concepts",
        "merged_concepts",
        "pruned_concepts",
        "active_concepts",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def _aggregate(
    rows: list[dict],
    methods: list[str],
) -> dict[str, dict[str, float | list[float] | int]]:
    summary: dict[str, dict[str, float | list[float] | int]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        curves = np.array(
            [row["mean_accuracy_curve"] for row in method_rows],
            dtype=np.float64,
        )
        entry: dict[str, float | list[float] | int] = {
            "n_runs": len(method_rows),
            "mean_final_accuracy": float(np.mean([row["final_accuracy"] for row in method_rows])),
            "std_final_accuracy": float(np.std([row["final_accuracy"] for row in method_rows])),
            "mean_accuracy_auc": float(np.mean([row["accuracy_auc"] for row in method_rows])),
            "std_accuracy_auc": float(np.std([row["accuracy_auc"] for row in method_rows])),
            "mean_total_bytes": float(np.mean([row["total_bytes"] for row in method_rows])),
            "mean_wall_clock_s": float(np.mean([row["wall_clock_s"] for row in method_rows])),
            "mean_accuracy_curve": curves.mean(axis=0).tolist(),
        }

        for key in (
            "concept_re_id_accuracy",
            "assignment_switch_rate",
            "avg_clients_per_concept",
            "singleton_group_ratio",
            "memory_reuse_rate",
            "routing_consistency",
            "wrong_memory_reuse_rate",
            "assignment_entropy",
            "worst_window_dip",
            "worst_window_recovery",
            "budget_normalized_score",
            "phase_a_bytes",
            "phase_b_bytes",
            "spawned_concepts",
            "merged_concepts",
            "pruned_concepts",
            "active_concepts",
        ):
            values = [row[key] for row in method_rows if row.get(key) is not None]
            if values:
                entry[f"mean_{key}"] = float(np.mean(values))
        summary[method] = entry
    return summary


def _rows_to_metrics(rows: list[dict], methods: list[str]) -> dict[str, list[MetricsResult]]:
    grouped: dict[str, list[MetricsResult]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        grouped[method] = [_metrics_to_result(row) for row in method_rows]
    return grouped


def _plot_accuracy_curves(summary: dict[str, dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for method, entry in summary.items():
        curve = np.asarray(entry["mean_accuracy_curve"], dtype=np.float64)
        ax.plot(range(len(curve)), curve, marker="o", linewidth=2, label=method)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("CIFAR-100 Recurrence: Mean Accuracy Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_bars(summary: dict[str, dict], output_path: Path) -> None:
    methods = list(summary.keys())
    metrics = [
        ("mean_final_accuracy", "Final Accuracy"),
        ("mean_accuracy_auc", "Accuracy AUC"),
        ("mean_concept_re_id_accuracy", "Concept Re-ID"),
        ("mean_total_bytes", "Total Bytes"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_arr = np.asarray(axes).reshape(-1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))

    for ax, (key, title) in zip(axes_arr, metrics):
        vals = [summary[m].get(key, np.nan) for m in methods]
        ax.bar(range(len(methods)), vals, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cifar100_full_sweep(args: argparse.Namespace) -> None:
    """Run the full CIFAR-100 recurrence baseline sweep."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(part.strip()) for part in str(args.seeds).split(",") if part.strip()]
    methods = resolve_cifar_sweep_methods(args.methods)

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
            "seed": seed,
            "dataset_config": dataset_config,
            "methods": methods,
            "federation_every": args.federation_every,
            "detector_name": args.detector_name,
            "baseline_lr": args.baseline_lr,
            "baseline_epochs": args.baseline_epochs,
            "fpt_linear_lr": args.fpt_linear_lr,
            "fpt_linear_epochs": args.fpt_linear_epochs,
            "fpt_linear_loss_novelty_threshold": args.fpt_linear_loss_novelty_threshold,
            "fpt_linear_merge_threshold": args.fpt_linear_merge_threshold,
            "fpt_linear_max_concepts": args.fpt_linear_max_concepts,
            "fpt_linear_dormant_recall": args.fpt_linear_dormant_recall,
            "fpt_adapter_lr": args.fpt_adapter_lr,
            "fpt_adapter_epochs": args.fpt_adapter_epochs,
            "fpt_adapter_loss_novelty_threshold": args.fpt_adapter_loss_novelty_threshold,
            "fpt_adapter_merge_threshold": args.fpt_adapter_merge_threshold,
            "fpt_adapter_max_concepts": args.fpt_adapter_max_concepts,
            "fpt_adapter_dormant_recall": args.fpt_adapter_dormant_recall,
            "fpt_adapter_global_shared_aggregation": args.fpt_adapter_global_shared_aggregation,
            "fpt_adapter_hidden_dim": args.fpt_adapter_hidden_dim,
            "fpt_adapter_dim": args.fpt_adapter_dim,
            "fpt_adapter_routed_local_training": args.fpt_adapter_routed_local_training,
            "ifca_lr": args.ifca_lr,
            "ifca_epochs": args.ifca_epochs,
            "pfedme_local_epochs": args.pfedme_local_epochs,
            "pfedme_k_steps": args.pfedme_k_steps,
            "pfedme_lamda": args.pfedme_lamda,
            "pfedme_personal_learning_rate": args.pfedme_personal_learning_rate,
            "apfl_alpha": args.apfl_alpha,
            "apfl_alpha_lr": args.apfl_alpha_lr,
            "apfl_local_steps": args.apfl_local_steps,
            "fedem_local_epochs": args.fedem_local_epochs,
            "cfl_eps_1": args.cfl_eps_1,
            "cfl_eps_2": args.cfl_eps_2,
            "cfl_warmup_rounds": args.cfl_warmup_rounds,
            "cfl_max_clusters": args.cfl_max_clusters,
            "fedccfa_cluster_eps": args.fedccfa_cluster_eps,
            "fedccfa_reid_similarity_threshold": args.fedccfa_reid_similarity_threshold,
            "fedccfa_prototype_mix": args.fedccfa_prototype_mix,
            "feddrift_similarity_threshold": args.feddrift_similarity_threshold,
            "tracked_summary_similarity_threshold": args.tracked_summary_similarity_threshold,
            "flash_distill_alpha": args.flash_distill_alpha,
            "atp_base_lr": args.atp_base_lr,
            "atp_meta_lr": args.atp_meta_lr,
            "flux_prior_clusters": args.flux_prior_clusters,
            "compressed_topk_fraction": args.compressed_topk_fraction,
        }
        for seed in seeds
    ]

    worker_count = min(args.max_workers, len(tasks))
    print(
        f"Running {len(methods)} methods x {len(seeds)} seeds "
        f"with {worker_count} worker(s)",
        flush=True,
    )

    raw_rows: list[dict] = []
    t0 = time.time()
    if worker_count <= 1:
        for task in tasks:
            for row in _run_seed_bundle(task):
                raw_rows.append(row)
                reid = row["concept_re_id_accuracy"]
                print(
                    f"  {row['method']} seed={row['seed']} "
                    f"final_acc={row['final_accuracy']:.4f} "
                    f"auc={row['accuracy_auc']:.4f} "
                    f"reid={reid if reid is not None else '--'}",
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            futures = [executor.submit(_run_seed_bundle, task) for task in tasks]
            for future in as_completed(futures):
                for row in future.result():
                    raw_rows.append(row)
                    reid = row["concept_re_id_accuracy"]
                    print(
                        f"  {row['method']} seed={row['seed']} "
                        f"final_acc={row['final_accuracy']:.4f} "
                        f"auc={row['accuracy_auc']:.4f} "
                        f"reid={reid if reid is not None else '--'}",
                        flush=True,
                    )

    elapsed = time.time() - t0
    method_rank = {method: idx for idx, method in enumerate(methods)}
    raw_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            method_rank.get(str(row["method"]), 999),
        )
    )

    summary = _aggregate(raw_rows, methods)
    grouped_results = _rows_to_metrics(raw_rows, methods)

    _write_raw_csv(raw_rows, results_dir / "raw_results.csv")
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(results_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_config": dataset_config,
                "seeds": seeds,
                "methods": methods,
                "federation_every": args.federation_every,
            },
            f,
            indent=2,
        )

    export_summary_csv(grouped_results, results_dir / "formal_comparison.csv")
    generate_main_table(grouped_results, results_dir / "formal_comparison.tex")
    _plot_accuracy_curves(summary, results_dir / "accuracy_curves.png")
    _plot_metric_bars(summary, results_dir / "comparison_metrics.png")

    print("\nSummary", flush=True)
    for method in methods:
        entry = summary.get(method)
        if entry is None:
            continue
        line = (
            f"  {method:24s} final={entry['mean_final_accuracy']:.4f} "
            f"auc={entry['mean_accuracy_auc']:.4f}"
        )
        if "mean_concept_re_id_accuracy" in entry:
            line += f" reid={entry['mean_concept_re_id_accuracy']:.4f}"
        line += f" bytes={entry['mean_total_bytes']:.0f}"
        print(line, flush=True)

    print(
        f"\nCompleted in {elapsed:.1f}s. Results saved to {results_dir}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the full sweep."""
    parser = argparse.ArgumentParser(
        description="Full CIFAR-100 recurrence baseline sweep",
    )
    parser.add_argument("--results-dir", default="results_cifar100_full_sweep")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--max-workers", type=int, default=3)
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
    parser.add_argument("--detector-name", default="ADWIN")
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)

    parser.add_argument("--baseline-lr", type=float, default=0.05)
    parser.add_argument("--baseline-epochs", type=int, default=5)

    parser.add_argument("--fpt-linear-lr", type=float, default=0.05)
    parser.add_argument("--fpt-linear-epochs", type=int, default=5)
    parser.add_argument("--fpt-linear-loss-novelty-threshold", type=float, default=0.565)
    parser.add_argument("--fpt-linear-merge-threshold", type=float, default=0.60)
    parser.add_argument("--fpt-linear-max-concepts", type=int, default=8)
    parser.add_argument(
        "--fpt-linear-dormant-recall",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--fpt-adapter-lr", type=float, default=0.05)
    parser.add_argument("--fpt-adapter-epochs", type=int, default=5)
    parser.add_argument("--fpt-adapter-loss-novelty-threshold", type=float, default=0.575)
    parser.add_argument("--fpt-adapter-merge-threshold", type=float, default=0.60)
    parser.add_argument("--fpt-adapter-max-concepts", type=int, default=8)
    parser.add_argument(
        "--fpt-adapter-dormant-recall",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fpt-adapter-global-shared-aggregation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fpt-adapter-hidden-dim", type=int, default=64)
    parser.add_argument("--fpt-adapter-dim", type=int, default=16)
    parser.add_argument("--fpt-adapter-routed-local-training", action="store_true")

    parser.add_argument("--ifca-lr", type=float, default=0.05)
    parser.add_argument("--ifca-epochs", type=int, default=5)
    parser.add_argument("--pfedme-local-epochs", type=int, default=3)
    parser.add_argument("--pfedme-k-steps", type=int, default=5)
    parser.add_argument("--pfedme-lamda", type=float, default=0.1)
    parser.add_argument("--pfedme-personal-learning-rate", type=float, default=0.05)
    parser.add_argument("--apfl-alpha", type=float, default=0.5)
    parser.add_argument("--apfl-alpha-lr", type=float, default=0.05)
    parser.add_argument("--apfl-local-steps", type=int, default=2)
    parser.add_argument("--fedem-local-epochs", type=int, default=2)
    parser.add_argument("--cfl-eps-1", type=float, default=0.4)
    parser.add_argument("--cfl-eps-2", type=float, default=1.6)
    parser.add_argument("--cfl-warmup-rounds", type=int, default=20)
    parser.add_argument("--cfl-max-clusters", type=int, default=8)
    parser.add_argument("--fedccfa-cluster-eps", type=float, default=0.35)
    parser.add_argument("--fedccfa-reid-similarity-threshold", type=float, default=0.85)
    parser.add_argument("--fedccfa-prototype-mix", type=float, default=0.20)
    parser.add_argument("--feddrift-similarity-threshold", type=float, default=0.5)
    parser.add_argument("--tracked-summary-similarity-threshold", type=float, default=0.5)
    parser.add_argument("--flash-distill-alpha", type=float, default=0.3)
    parser.add_argument("--atp-base-lr", type=float, default=0.05)
    parser.add_argument("--atp-meta-lr", type=float, default=0.15)
    parser.add_argument("--flux-prior-clusters", type=int, default=3)
    parser.add_argument("--compressed-topk-fraction", type=float, default=0.3)
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the full sweep."""
    parser = build_parser()
    args = parser.parse_args(argv)

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
        args.max_workers = min(args.max_workers, 2)

    run_cifar100_full_sweep(args)


if __name__ == "__main__":
    main()
