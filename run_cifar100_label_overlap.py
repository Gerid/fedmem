from __future__ import annotations

"""Stage-1 CIFAR-100 overlap veto for the recurrence winner."""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fedprotrack.experiments.cifar_overlap import (
    CIFARSubsetBenchmarkConfig,
    build_overlap_concept_classes,
    build_subset_dataset,
    make_result_metadata,
    run_fpt_result,
    summarize_root_cause,
)
from fedprotrack.metrics import compute_all_metrics

os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

COMMON_STAGE1 = {
    "dormant_recall": True,
    "loss_novelty_threshold": 0.575,
    "merge_threshold": 0.60,
    "max_concepts": 8,
    "global_shared_aggregation": False,
}
VARIANTS: dict[str, dict[str, object]] = {
    "linear_split": {
        **COMMON_STAGE1,
        "method": "linear_split",
        "model_type": "linear",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_local_shared": {
        **COMMON_STAGE1,
        "method": "feature_local_shared",
        "model_type": "feature_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "factorized_local_shared": {
        **COMMON_STAGE1,
        "method": "factorized_local_shared",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "factorized_weighted_freeze": {
        **COMMON_STAGE1,
        "method": "factorized_weighted_freeze",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
    },
    "factorized_anchor_freeze": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_freeze",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
    },
    "factorized_anchor_cap2_freeze": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_freeze",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "max_spawn_clusters_per_round": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
    },
    "factorized_anchor_cap2_hysteresis": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_hysteresis",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "max_spawn_clusters_per_round": 2,
        "novelty_hysteresis_rounds": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
    },
    "factorized_anchor_cap2_sharpened_read": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_sharpened_read",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "routed_read_top_k": 2,
        "routed_read_temperature": 0.50,
        "max_spawn_clusters_per_round": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
    },
    "factorized_anchor_cap2_conditional_sharpened_read": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_conditional_sharpened_read",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "routed_read_top_k": 2,
        "routed_read_temperature": 0.50,
        "routed_read_only_on_ambiguity": True,
        "routed_read_min_secondary_weight": 0.20,
        "routed_read_max_primary_gap": 0.35,
        "max_spawn_clusters_per_round": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
    },
    "factorized_anchor_cap2_entropy_sharpened_read": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_entropy_sharpened_read",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "routed_read_top_k": 2,
        "routed_read_temperature": 0.50,
        "routed_read_only_on_ambiguity": True,
        "routed_read_min_entropy": 0.20,
        "max_spawn_clusters_per_round": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
    },
    "factorized_anchor_cap2_consolidate": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_consolidate",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "max_spawn_clusters_per_round": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
        "factorized_primary_consolidation_steps": 1,
    },
    "factorized_anchor_cap2_head_consolidate": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_head_consolidate",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "max_spawn_clusters_per_round": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
        "factorized_primary_consolidation_steps": 1,
        "factorized_primary_consolidation_mode": "head_only",
    },
    "factorized_anchor_cap2_top2_freeze": {
        **COMMON_STAGE1,
        "method": "factorized_anchor_cap2_top2_freeze",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "routed_write_top_k": 2,
        "routed_write_min_secondary_weight": 0.20,
        "max_spawn_clusters_per_round": 2,
        "factorized_slot_preserving": True,
        "factorized_primary_anchor_alpha": 0.25,
        "factorized_secondary_anchor_alpha": 0.75,
    },
    "factorized_top2_freeze": {
        **COMMON_STAGE1,
        "method": "factorized_top2_freeze",
        "model_type": "factorized_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
        "routed_write_top_k": 2,
        "routed_write_min_secondary_weight": 0.20,
    },
    "feature_model_embed": {
        **COMMON_STAGE1,
        "method": "feature_model_embed",
        "model_type": "feature_adapter",
        "fingerprint_source": "model_embed",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_pre_adapter_embed": {
        **COMMON_STAGE1,
        "method": "feature_pre_adapter_embed",
        "model_type": "feature_adapter",
        "fingerprint_source": "pre_adapter_embed",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_hybrid_pre_adapter": {
        **COMMON_STAGE1,
        "method": "feature_hybrid_pre_adapter",
        "model_type": "feature_adapter",
        "fingerprint_source": "hybrid_raw_pre_adapter",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_weighted_hybrid_pre_adapter": {
        **COMMON_STAGE1,
        "method": "feature_weighted_hybrid_pre_adapter",
        "model_type": "feature_adapter",
        "fingerprint_source": "weighted_hybrid_raw_pre_adapter",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_centered_hybrid_pre_adapter": {
        **COMMON_STAGE1,
        "method": "feature_centered_hybrid_pre_adapter",
        "model_type": "feature_adapter",
        "fingerprint_source": "centered_hybrid_raw_pre_adapter",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_attenuated_hybrid_pre_adapter": {
        **COMMON_STAGE1,
        "method": "feature_attenuated_hybrid_pre_adapter",
        "model_type": "feature_adapter",
        "fingerprint_source": "attenuated_hybrid_raw_pre_adapter",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_double_raw_hybrid_pre_adapter": {
        **COMMON_STAGE1,
        "method": "feature_double_raw_hybrid_pre_adapter",
        "model_type": "feature_adapter",
        "fingerprint_source": "double_raw_hybrid_pre_adapter",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_bootstrap_raw_hybrid_pre_adapter": {
        **COMMON_STAGE1,
        "method": "feature_bootstrap_raw_hybrid_pre_adapter",
        "model_type": "feature_adapter",
        "fingerprint_source": "bootstrap_raw_hybrid_pre_adapter",
        "expert_update_policy": "map_only",
        "shared_update_policy": "always",
    },
    "feature_weighted_freeze": {
        **COMMON_STAGE1,
        "method": "feature_weighted_freeze",
        "model_type": "feature_adapter",
        "fingerprint_source": "raw_input",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
    },
    "feature_combo": {
        **COMMON_STAGE1,
        "method": "feature_combo",
        "model_type": "feature_adapter",
        "fingerprint_source": "model_embed",
        "expert_update_policy": "posterior_weighted",
        "shared_update_policy": "freeze_on_multiroute",
    },
}
RAW_RESULT_COLUMNS = [
    "method",
    "seed",
    "overlap_ratio",
    "model_type",
    "fingerprint_source",
    "expert_update_policy",
    "shared_update_policy",
    "global_shared_aggregation",
    "routed_write_top_k",
    "routed_write_min_secondary_weight",
    "routed_read_top_k",
    "routed_read_temperature",
    "routed_read_only_on_ambiguity",
    "routed_read_min_entropy",
    "routed_read_min_secondary_weight",
    "routed_read_max_primary_gap",
    "max_spawn_clusters_per_round",
    "novelty_hysteresis_rounds",
    "factorized_slot_preserving",
    "factorized_primary_anchor_alpha",
    "factorized_secondary_anchor_alpha",
    "factorized_primary_consolidation_steps",
    "factorized_primary_consolidation_mode",
    "spawned",
    "merged",
    "active",
    "assignment_switch_rate",
    "avg_clients_per_concept",
    "singleton_group_ratio",
    "memory_reuse_rate",
    "routing_consistency",
    "shared_drift_norm",
    "expert_update_coverage",
    "multi_route_rate",
    "final_accuracy",
    "accuracy_auc",
    "concept_re_id_accuracy",
    "assignment_entropy",
    "wrong_memory_reuse_rate",
    "total_bytes",
]
VETO_TABLE_COLUMNS = [
    "method",
    "overlap_ratio",
    "n_runs",
    "model_type",
    "fingerprint_source",
    "expert_update_policy",
    "shared_update_policy",
    "global_shared_aggregation",
    "routed_write_top_k",
    "routed_write_min_secondary_weight",
    "routed_read_top_k",
    "routed_read_temperature",
    "routed_read_only_on_ambiguity",
    "routed_read_min_entropy",
    "routed_read_min_secondary_weight",
    "routed_read_max_primary_gap",
    "max_spawn_clusters_per_round",
    "novelty_hysteresis_rounds",
    "factorized_slot_preserving",
    "factorized_primary_anchor_alpha",
    "factorized_secondary_anchor_alpha",
    "factorized_primary_consolidation_steps",
    "factorized_primary_consolidation_mode",
    "mean_final_accuracy",
    "std_final_accuracy",
    "mean_concept_re_id_accuracy",
    "std_concept_re_id_accuracy",
    "mean_assignment_switch_rate",
    "std_assignment_switch_rate",
    "mean_avg_clients_per_concept",
    "std_avg_clients_per_concept",
    "mean_singleton_group_ratio",
    "std_singleton_group_ratio",
    "mean_memory_reuse_rate",
    "std_memory_reuse_rate",
    "mean_routing_consistency",
    "std_routing_consistency",
    "mean_shared_drift_norm",
    "std_shared_drift_norm",
    "mean_expert_update_coverage",
    "std_expert_update_coverage",
    "mean_multi_route_rate",
    "std_multi_route_rate",
    "mean_total_bytes",
    "std_total_bytes",
]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _aggregate_rows(
    rows: list[dict[str, object]],
    *,
    group_keys: list[str],
    numeric_keys: list[str],
    extra_keys: list[str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        out = {k: v for k, v in zip(group_keys, key)}
        out["n_runs"] = len(group)
        for field in extra_keys:
            values = [row.get(field) for row in group if row.get(field) not in (None, "")]
            if values:
                out[field] = values[0]
        for field in numeric_keys:
            values = [float(row[field]) for row in group if row.get(field) not in (None, "")]
            if values:
                out[f"mean_{field}"] = float(np.mean(values))
                out[f"std_{field}"] = float(np.std(values))
        aggregated.append(out)
    return aggregated


def _plot_metric_vs_overlap(rows: list[dict[str, object]], output_path: Path) -> None:
    methods = sorted({str(row["method"]) for row in rows})
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
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_variant(
    ds,
    *,
    seed: int,
    federation_every: int,
    epochs: int,
    lr: float,
    variant: dict[str, object],
) -> object:
    return run_fpt_result(
        ds,
        federation_every,
        epochs,
        lr,
        seed,
        dormant_recall=bool(variant["dormant_recall"]),
        loss_novelty_threshold=float(variant["loss_novelty_threshold"]),
        merge_threshold=float(variant["merge_threshold"]),
        max_concepts=int(variant["max_concepts"]),
        model_type=str(variant["model_type"]),
        hidden_dim=64,
        adapter_dim=16,
        global_shared_aggregation=bool(variant["global_shared_aggregation"]),
        fingerprint_source=str(variant["fingerprint_source"]),
        expert_update_policy=str(variant["expert_update_policy"]),
        shared_update_policy=str(variant["shared_update_policy"]),
        routed_write_top_k=(
            int(variant["routed_write_top_k"])
            if "routed_write_top_k" in variant
            else None
        ),
        routed_write_min_secondary_weight=float(
            variant.get("routed_write_min_secondary_weight", 0.0)
        ),
        routed_read_top_k=(
            int(variant["routed_read_top_k"])
            if "routed_read_top_k" in variant
            else None
        ),
        routed_read_temperature=float(
            variant.get("routed_read_temperature", 1.0)
        ),
        routed_read_only_on_ambiguity=bool(
            variant.get("routed_read_only_on_ambiguity", False)
        ),
        routed_read_min_entropy=(
            float(variant["routed_read_min_entropy"])
            if "routed_read_min_entropy" in variant
            else None
        ),
        routed_read_min_secondary_weight=float(
            variant.get("routed_read_min_secondary_weight", 0.0)
        ),
        routed_read_max_primary_gap=(
            float(variant["routed_read_max_primary_gap"])
            if "routed_read_max_primary_gap" in variant
            else None
        ),
        max_spawn_clusters_per_round=(
            int(variant["max_spawn_clusters_per_round"])
            if "max_spawn_clusters_per_round" in variant
            else None
        ),
        novelty_hysteresis_rounds=(
            int(variant["novelty_hysteresis_rounds"])
            if "novelty_hysteresis_rounds" in variant
            else None
        ),
        factorized_slot_preserving=bool(
            variant.get("factorized_slot_preserving", False)
        ),
        factorized_primary_anchor_alpha=float(
            variant.get("factorized_primary_anchor_alpha", 0.25)
        ),
        factorized_secondary_anchor_alpha=float(
            variant.get("factorized_secondary_anchor_alpha", 0.75)
        ),
        factorized_primary_consolidation_steps=int(
            variant.get("factorized_primary_consolidation_steps", 0)
        ),
        factorized_primary_consolidation_mode=str(
            variant.get("factorized_primary_consolidation_mode", "full")
        ),
    )


def _row_from_result(
    *,
    method: str,
    seed: int,
    overlap_ratio: float,
    result,
    variant: dict[str, object],
) -> dict[str, object]:
    metrics = compute_all_metrics(result.to_experiment_log(), identity_capable=True)
    row: dict[str, object] = {
        "method": method,
        "seed": seed,
        "overlap_ratio": overlap_ratio,
        "final_accuracy": metrics.final_accuracy,
        "accuracy_auc": metrics.accuracy_auc,
        "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
        "assignment_entropy": metrics.assignment_entropy,
        "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
        "total_bytes": float(result.total_bytes),
    }
    row.update(
        make_result_metadata(
            model_type=str(variant["model_type"]),
            fingerprint_source=str(variant["fingerprint_source"]),
            expert_update_policy=str(variant["expert_update_policy"]),
            shared_update_policy=str(variant["shared_update_policy"]),
            global_shared_aggregation=bool(variant["global_shared_aggregation"]),
            routed_write_top_k=(
                int(variant["routed_write_top_k"])
                if "routed_write_top_k" in variant
                else None
            ),
            routed_write_min_secondary_weight=(
                float(variant["routed_write_min_secondary_weight"])
                if "routed_write_min_secondary_weight" in variant
                else None
            ),
            routed_read_top_k=(
                int(variant["routed_read_top_k"])
                if "routed_read_top_k" in variant
                else None
            ),
            routed_read_temperature=(
                float(variant["routed_read_temperature"])
                if "routed_read_temperature" in variant
                else None
            ),
            routed_read_only_on_ambiguity=(
                bool(variant["routed_read_only_on_ambiguity"])
                if "routed_read_only_on_ambiguity" in variant
                else None
            ),
            routed_read_min_entropy=(
                float(variant["routed_read_min_entropy"])
                if "routed_read_min_entropy" in variant
                else None
            ),
            routed_read_min_secondary_weight=(
                float(variant["routed_read_min_secondary_weight"])
                if "routed_read_min_secondary_weight" in variant
                else None
            ),
            routed_read_max_primary_gap=(
                float(variant["routed_read_max_primary_gap"])
                if "routed_read_max_primary_gap" in variant
                else None
            ),
            max_spawn_clusters_per_round=(
                int(variant["max_spawn_clusters_per_round"])
                if "max_spawn_clusters_per_round" in variant
                else None
            ),
            novelty_hysteresis_rounds=(
                int(variant["novelty_hysteresis_rounds"])
                if "novelty_hysteresis_rounds" in variant
                else None
            ),
            factorized_slot_preserving=bool(variant.get("factorized_slot_preserving", False)),
            factorized_primary_anchor_alpha=float(
                variant.get("factorized_primary_anchor_alpha", 0.25)
            ) if variant.get("factorized_slot_preserving", False) else None,
            factorized_secondary_anchor_alpha=float(
                variant.get("factorized_secondary_anchor_alpha", 0.75)
            ) if variant.get("factorized_slot_preserving", False) else None,
            factorized_primary_consolidation_steps=(
                int(variant.get("factorized_primary_consolidation_steps", 0))
                if variant.get("factorized_slot_preserving", False)
                else None
            ),
            factorized_primary_consolidation_mode=(
                str(variant.get("factorized_primary_consolidation_mode", "full"))
                if variant.get("factorized_slot_preserving", False)
                else None
            ),
            spawned=int(result.spawned_concepts),
            merged=int(result.merged_concepts),
            active=int(result.active_concepts),
            assignment_switch_rate=result.assignment_switch_rate,
            avg_clients_per_concept=result.avg_clients_per_concept,
            singleton_group_ratio=result.singleton_group_ratio,
            memory_reuse_rate=result.memory_reuse_rate,
            routing_consistency=result.routing_consistency,
            shared_drift_norm=result.shared_drift_norm,
            expert_update_coverage=result.expert_update_coverage,
            multi_route_rate=result.multi_route_rate,
        )
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results_cifar100_label_overlap")
    parser.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["feature_local_shared", "feature_hybrid_pre_adapter", "linear_split"],
    )
    parser.add_argument("--winner", type=str, default=None)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--T", type=int, default=24)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--federation-every", type=int, default=2)
    args = parser.parse_args()

    selected_methods = list(dict.fromkeys(args.variants))
    if "feature_local_shared" not in selected_methods:
        selected_methods.insert(0, "feature_local_shared")
    for method in selected_methods:
        if method not in VARIANTS:
            raise ValueError(f"Unknown variant '{method}'. Choose from {sorted(VARIANTS)}")

    winner_method = args.winner
    if winner_method is None:
        for method in selected_methods:
            if method not in {"feature_local_shared", "linear_split"}:
                winner_method = method
                break

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    phase_concepts = [[0, 1], [2, 3], [0, 1]]

    all_rows: list[dict[str, object]] = []
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
            for method in selected_methods:
                variant = VARIANTS[method]
                result = _run_variant(
                    ds,
                    seed=seed,
                    federation_every=args.federation_every,
                    epochs=args.epochs,
                    lr=args.lr,
                    variant=variant,
                )
                row = _row_from_result(
                    method=method,
                    seed=seed,
                    overlap_ratio=overlap_ratio,
                    result=result,
                    variant=variant,
                )
                all_rows.append(row)
                print(
                    f"overlap={overlap_ratio:.2f} seed={seed} {method:22s} "
                    f"final={float(row['final_accuracy']):.4f} "
                    f"reid={float(row['concept_re_id_accuracy']):.4f} "
                    f"bytes={float(row['total_bytes']):.0f}"
                )

    _write_csv(results_dir / "raw_results.csv", all_rows, RAW_RESULT_COLUMNS)
    overlap_veto_table = _aggregate_rows(
        all_rows,
        group_keys=["method", "overlap_ratio"],
        numeric_keys=[
            "final_accuracy",
            "accuracy_auc",
            "concept_re_id_accuracy",
            "assignment_entropy",
            "assignment_switch_rate",
            "avg_clients_per_concept",
            "singleton_group_ratio",
            "memory_reuse_rate",
            "routing_consistency",
            "shared_drift_norm",
            "expert_update_coverage",
            "multi_route_rate",
            "wrong_memory_reuse_rate",
            "total_bytes",
            "spawned",
            "merged",
            "active",
        ],
        extra_keys=[
            "model_type",
            "fingerprint_source",
            "expert_update_policy",
            "shared_update_policy",
            "global_shared_aggregation",
            "routed_write_top_k",
            "routed_write_min_secondary_weight",
            "routed_read_top_k",
            "routed_read_temperature",
            "routed_read_only_on_ambiguity",
            "routed_read_min_entropy",
            "routed_read_min_secondary_weight",
            "routed_read_max_primary_gap",
            "max_spawn_clusters_per_round",
            "novelty_hysteresis_rounds",
            "factorized_slot_preserving",
            "factorized_primary_anchor_alpha",
            "factorized_secondary_anchor_alpha",
            "factorized_primary_consolidation_steps",
            "factorized_primary_consolidation_mode",
        ],
    )
    _write_json(results_dir / "overlap_veto_table.json", overlap_veto_table)
    _write_csv(results_dir / "overlap_veto_table.csv", overlap_veto_table, VETO_TABLE_COLUMNS)

    if winner_method is not None and winner_method in selected_methods:
        summary_lines = summarize_root_cause(
            [row for row in all_rows if str(row["method"]) == "feature_local_shared"],
            [row for row in all_rows if str(row["method"]) == winner_method],
            final_metric="final_accuracy",
            downstream_metrics=("final_accuracy",),
            bytes_metric="total_bytes",
            overlap_compare=tuple(args.overlaps),
        )
    else:
        summary_lines = [
            "Root cause: no overlap winner was selected for comparison.",
            "Need a recurrence winner plus the `feature_local_shared` baseline to run the veto summary.",
        ]
    (results_dir / "root_cause_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    _plot_metric_vs_overlap(all_rows, results_dir / "diagnostics_vs_overlap.png")
    print(f"Saved overlap veto outputs to {results_dir}")


if __name__ == "__main__":
    main()
