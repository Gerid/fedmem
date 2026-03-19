from __future__ import annotations

"""Stage-1 CIFAR-100 recurrence screening and decision workflow."""

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
    build_subset_dataset,
    make_result_metadata,
    run_fpt_result,
    summarize_root_cause,
)
from fedprotrack.metrics import compute_all_metrics

os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

CONCEPT_CLASSES = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14],
    3: [15, 16, 17, 18, 19],
    4: [0, 1, 5, 6, 10],
    5: [11, 15, 16, 2, 3],
    6: [4, 7, 8, 17, 18],
    7: [9, 12, 13, 14, 19],
}
PHASE_CONCEPTS = [[0, 1, 2], [3, 4, 5], [0, 1, 2]]

SCREEN_SEED = 42
DECISION_SEEDS = [42, 123, 456]
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
SCREEN_ORDER = [
    "feature_local_shared",
    "factorized_local_shared",
    "factorized_weighted_freeze",
    "factorized_anchor_freeze",
    "factorized_anchor_cap2_freeze",
    "factorized_anchor_cap2_hysteresis",
    "factorized_anchor_cap2_sharpened_read",
    "factorized_anchor_cap2_conditional_sharpened_read",
    "factorized_anchor_cap2_entropy_sharpened_read",
    "factorized_anchor_cap2_consolidate",
    "factorized_anchor_cap2_head_consolidate",
    "factorized_anchor_cap2_top2_freeze",
    "factorized_top2_freeze",
    "feature_pre_adapter_embed",
    "feature_hybrid_pre_adapter",
    "feature_weighted_hybrid_pre_adapter",
]
DECISION_BASELINES = ["linear_split", "feature_local_shared"]
RESULT_COLUMNS = [
    "method",
    "seed",
    "stage",
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
    "final",
    "phase1",
    "phase2",
    "phase3",
    "recovery_t20",
    "recovery_next_fed",
    "bytes",
    "concept_re_id_accuracy",
    "assignment_entropy",
    "wrong_memory_reuse_rate",
]
PHASE_A_DIAGNOSTIC_COLUMNS = [
    "method",
    "seed",
    "stage",
    "t",
    "library_size_before",
    "active_after",
    "spawned",
    "merged",
    "n_clients_logged",
    "mean_fp_loss",
    "min_fp_loss",
    "max_fp_loss",
    "mean_effective_threshold",
    "mean_fp_gap",
    "over_threshold_rate",
    "mean_map_prob",
]
SELECTION_COLUMNS = [
    "method",
    "n_runs",
    "selected_for_decision",
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
    "mean_final",
    "std_final",
    "mean_phase3",
    "std_phase3",
    "mean_recovery_next_fed",
    "std_recovery_next_fed",
    "mean_bytes",
    "std_bytes",
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
    "mean_spawned",
    "mean_merged",
    "mean_active",
]


def _next_federation_t(start_t: int, federation_every: int, horizon: int) -> int | None:
    t = start_t
    while t < horizon:
        if (t + 1) % federation_every == 0:
            return t
        t += 1
    return None


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


def _build_dataset(
    *,
    K: int,
    T: int,
    n_samples: int,
    n_features: int,
    seed: int,
) -> object:
    return build_subset_dataset(
        CIFARSubsetBenchmarkConfig(
            K=K,
            T=T,
            n_samples=n_samples,
            n_features=n_features,
            seed=seed,
            generator_type="cifar100_recurrence",
        ),
        concept_classes=CONCEPT_CLASSES,
        phase_concepts=PHASE_CONCEPTS,
    )


def _row_from_result(
    *,
    method: str,
    seed: int,
    stage: str,
    result,
    variant: dict[str, object],
    phase3_start: int,
    recovery_eval_t: int | None,
) -> tuple[dict[str, object], np.ndarray]:
    acc_mat = result.accuracy_matrix
    metrics = compute_all_metrics(result.to_experiment_log(), identity_capable=True)
    row: dict[str, object] = {
        "method": method,
        "seed": seed,
        "stage": stage,
        "final": float(acc_mat[:, -1].mean()),
        "phase1": float(acc_mat[:, :10].mean()),
        "phase2": float(acc_mat[:, 10:20].mean()),
        "phase3": float(acc_mat[:, 20:].mean()),
        "recovery_t20": float(acc_mat[:, phase3_start].mean()) if acc_mat.shape[1] > phase3_start else 0.0,
        "recovery_next_fed": (
            float(acc_mat[:, recovery_eval_t].mean())
            if recovery_eval_t is not None
            else 0.0
        ),
        "bytes": float(result.total_bytes),
        "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
        "assignment_entropy": metrics.assignment_entropy,
        "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
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
    return row, acc_mat.mean(axis=0)


def _phase_a_round_rows(
    *,
    method: str,
    seed: int,
    stage: str,
    result,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for diag in getattr(result, "phase_a_round_diagnostics", []) or []:
        row = {
            "method": method,
            "seed": seed,
            "stage": stage,
        }
        row.update(diag)
        rows.append(row)
    return rows


def _run_variant(
    ds,
    *,
    seed: int,
    fed_every: int,
    epochs: int,
    lr: float,
    variant: dict[str, object],
) -> object:
    return run_fpt_result(
        ds,
        fed_every,
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


def _passes_screen_gate(
    baseline_row: dict[str, object],
    candidate_row: dict[str, object],
) -> bool:
    for metric in ("final", "phase3", "recovery_next_fed"):
        if float(candidate_row[metric]) + 1e-9 < float(baseline_row[metric]):
            return False
    return True


def main() -> None:
    results_dir = Path("results_cifar100_recurrence_gap")
    results_dir.mkdir(parents=True, exist_ok=True)

    K, T = 12, 30
    n_samples = 200
    n_features = 128
    epochs, lr = 5, 0.05
    federation_every = 2
    phase3_start = 20
    recovery_eval_t = _next_federation_t(phase3_start, federation_every, T)

    print("=" * 65)
    print("Stage-1 Recurrence Screening")
    print(f"K={K}, T={T}, seeds={DECISION_SEEDS}")
    print("Screen seed: 42")
    print("Decision baselines: linear_split, feature_local_shared")
    print(f"Adapter candidates: {', '.join(SCREEN_ORDER[1:])}")
    print("=" * 65)

    all_rows: list[dict[str, object]] = []
    phase_a_rows: list[dict[str, object]] = []
    all_curves: dict[str, list[np.ndarray]] = {}

    screen_ds = _build_dataset(
        K=K,
        T=T,
        n_samples=n_samples,
        n_features=n_features,
        seed=SCREEN_SEED,
    )
    screen_rows: dict[str, dict[str, object]] = {}
    baseline_row: dict[str, object] | None = None

    for method in DECISION_BASELINES + SCREEN_ORDER[1:]:
        variant = VARIANTS[method]
        result = _run_variant(
            screen_ds,
            seed=SCREEN_SEED,
            fed_every=federation_every,
            epochs=epochs,
            lr=lr,
            variant=variant,
        )
        row, curve = _row_from_result(
            method=method,
            seed=SCREEN_SEED,
            stage="screen",
            result=result,
            variant=variant,
            phase3_start=phase3_start,
            recovery_eval_t=recovery_eval_t,
        )
        screen_rows[method] = row
        all_rows.append(row)
        phase_a_rows.extend(
            _phase_a_round_rows(
                method=method,
                seed=SCREEN_SEED,
                stage="screen",
                result=result,
            )
        )
        all_curves.setdefault(method, []).append(curve)
        if method == "feature_local_shared":
            baseline_row = row
        print(
            f"[screen] {method:22s} final={float(row['final']):.4f} "
            f"P3={float(row['phase3']):.4f} recov@fed={float(row['recovery_next_fed']):.4f} "
            f"bytes={float(row['bytes']):.0f}"
        )

    if baseline_row is None:
        raise RuntimeError("feature_local_shared baseline row was not produced")

    screened_candidates = [
        method
        for method in SCREEN_ORDER[1:]
        if _passes_screen_gate(baseline_row, screen_rows[method])
    ]
    decision_methods = DECISION_BASELINES + screened_candidates
    if "linear_split" not in decision_methods:
        decision_methods.insert(0, "linear_split")

    print(f"Selected for decision: {decision_methods}")

    for seed in DECISION_SEEDS:
        ds = screen_ds if seed == SCREEN_SEED else _build_dataset(
            K=K,
            T=T,
            n_samples=n_samples,
            n_features=n_features,
            seed=seed,
        )
        for method in decision_methods:
            if seed == SCREEN_SEED and method in screen_rows:
                screen_rows[method]["stage"] = "decision"
                continue
            variant = VARIANTS[method]
            result = _run_variant(
                ds,
                seed=seed,
                fed_every=federation_every,
                epochs=epochs,
                lr=lr,
                variant=variant,
            )
            row, curve = _row_from_result(
                method=method,
                seed=seed,
                stage="decision",
                result=result,
                variant=variant,
                phase3_start=phase3_start,
                recovery_eval_t=recovery_eval_t,
            )
            all_rows.append(row)
            phase_a_rows.extend(
                _phase_a_round_rows(
                    method=method,
                    seed=seed,
                    stage="decision",
                    result=result,
                )
            )
            all_curves.setdefault(method, []).append(curve)
            print(
                f"[decision] seed={seed} {method:22s} final={float(row['final']):.4f} "
                f"P3={float(row['phase3']):.4f} recov@fed={float(row['recovery_next_fed']):.4f} "
                f"bytes={float(row['bytes']):.0f}"
            )

    selection_rows = _aggregate_rows(
        all_rows,
        group_keys=["method"],
        numeric_keys=[
            "final",
            "phase1",
            "phase2",
            "phase3",
            "recovery_t20",
            "recovery_next_fed",
            "bytes",
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
    for row in selection_rows:
        row["selected_for_decision"] = str(row["method"]) in decision_methods
    selection_rows.sort(key=lambda row: str(row["method"]))

    _write_json(results_dir / "results.json", all_rows)
    _write_json(results_dir / "phase_a_round_diagnostics.json", phase_a_rows)
    _write_json(results_dir / "recurrence_selection_table.json", selection_rows)
    _write_csv(
        results_dir / "phase_a_round_diagnostics.csv",
        phase_a_rows,
        PHASE_A_DIAGNOSTIC_COLUMNS,
    )
    _write_csv(results_dir / "recurrence_selection_table.csv", selection_rows, SELECTION_COLUMNS)

    candidate_methods = [method for method in decision_methods if method not in DECISION_BASELINES]
    if candidate_methods:
        winner_row = max(
            (row for row in selection_rows if str(row["method"]) in candidate_methods),
            key=lambda row: float(row.get("mean_final", float("-inf"))),
        )
        winner_method = str(winner_row["method"])
        summary_lines = summarize_root_cause(
            [row for row in all_rows if str(row["method"]) == "feature_local_shared"],
            [row for row in all_rows if str(row["method"]) == winner_method],
        )
    else:
        summary_lines = [
            "Root cause: no candidate survived the seed=42 screening gate.",
            "Strongest signal: every candidate underperformed `feature_local_shared` on at least one primary recurrence metric.",
            "Next experiment: inspect Phase A and local-training diagnostics before moving to Stage 2.",
            "Gate: stop tuning thresholds until one candidate clears the screening gate.",
        ]
    (results_dir / "root_cause_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    plot_methods = decision_methods
    fig, ax = plt.subplots(figsize=(12, 6))
    for method in plot_methods:
        curves = all_curves.get(method, [])
        if not curves:
            continue
        ax.plot(range(T), np.mean(curves, axis=0), linewidth=2, label=method)
    ax.axvline(10, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(20, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("CIFAR recurrence Stage-1 selection")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "recurrence_gap_curves.png", dpi=150)
    plt.close(fig)

    print(f"Saved recurrence outputs to {results_dir}")


if __name__ == "__main__":
    main()
