from __future__ import annotations

"""Autonomous scaffold-mutation loop for adapter rescue experiments.

The loop is intentionally scoped to experiment-scaffold mutations:
fingerprint source, local-update policy, and narrow threshold changes.
It does not rewrite arbitrary model source files.
"""

import argparse
import copy
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import run_cifar100_recurrence_gap as recurrence_gap
from fedprotrack.experiments.cifar_overlap import summarize_root_cause


AGG_NUMERIC_KEYS = [
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
]
AGG_EXTRA_KEYS = [
    "model_type",
    "fingerprint_source",
    "expert_update_policy",
    "shared_update_policy",
    "global_shared_aggregation",
]
VARIANT_KEYS = [
    "model_type",
    "fingerprint_source",
    "expert_update_policy",
    "shared_update_policy",
    "loss_novelty_threshold",
    "merge_threshold",
    "max_concepts",
    "global_shared_aggregation",
]


@dataclass
class LoopConfig:
    results_dir: Path
    rounds: int
    screen_seed: int
    decision_seeds: list[int]
    K: int = 12
    T: int = 30
    n_samples: int = 200
    n_features: int = 128
    epochs: int = 5
    lr: float = 0.05
    federation_every: int = 2
    phase3_start: int = 20


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_signature(variant: dict[str, object]) -> str:
    return "|".join(f"{key}={variant[key]}" for key in VARIANT_KEYS)


def _make_candidate_variant(
    *,
    fingerprint_source: str,
    expert_update_policy: str,
    shared_update_policy: str,
    loss_novelty_threshold: float,
    merge_threshold: float,
    max_concepts: int,
    global_shared_aggregation: bool,
) -> dict[str, object]:
    variant = copy.deepcopy(recurrence_gap.COMMON_STAGE1)
    variant.update(
        {
            "method": "auto_candidate",
            "model_type": "feature_adapter",
            "fingerprint_source": fingerprint_source,
            "expert_update_policy": expert_update_policy,
            "shared_update_policy": shared_update_policy,
            "loss_novelty_threshold": loss_novelty_threshold,
            "merge_threshold": merge_threshold,
            "max_concepts": max_concepts,
            "global_shared_aggregation": global_shared_aggregation,
        }
    )
    return variant


def _failure_profile(summary: dict[str, object] | None) -> dict[str, bool]:
    if not summary:
        return {"routing_collapsed": True, "shared_drift_high": False, "multi_route_dead": True}
    spawned = float(summary.get("mean_spawned", 0.0) or 0.0)
    active = float(summary.get("mean_active", 0.0) or 0.0)
    switch_rate = float(summary.get("mean_assignment_switch_rate", 0.0) or 0.0)
    multi_route = float(summary.get("mean_multi_route_rate", 0.0) or 0.0)
    shared_drift = float(summary.get("mean_shared_drift_norm", 0.0) or 0.0)
    routing_collapsed = spawned <= 0.5 and active <= 1.5 and switch_rate <= 0.01
    return {
        "routing_collapsed": routing_collapsed,
        "shared_drift_high": shared_drift >= 0.16,
        "multi_route_dead": multi_route <= 0.01,
    }


def _candidate_catalog(champion_summary: dict[str, object] | None) -> list[dict[str, object]]:
    profile = _failure_profile(champion_summary)
    if profile["routing_collapsed"]:
        fingerprint_sources = ["pre_adapter_embed", "hybrid_raw_pre_adapter", "raw_input"]
        update_pairs = [("map_only", "always")]
        novelty_values = [0.575, 0.56, 0.59]
    else:
        fingerprint_sources = ["hybrid_raw_pre_adapter", "pre_adapter_embed", "raw_input"]
        update_pairs = [("map_only", "always"), ("posterior_weighted", "freeze_on_multiroute")]
        novelty_values = [0.575, 0.59, 0.56]

    merge_values = [0.60, 0.58, 0.62]
    max_concepts_values = [8, 6, 10]
    shared_values = [False] if profile["shared_drift_high"] else [False, True]

    catalog: list[dict[str, object]] = []
    for fingerprint_source, (expert_update_policy, shared_update_policy), loss_th, merge_th, max_concepts, global_shared in product(
        fingerprint_sources,
        update_pairs,
        novelty_values,
        merge_values,
        max_concepts_values,
        shared_values,
    ):
        if shared_update_policy == "freeze_on_multiroute" and fingerprint_source == "model_embed":
            continue
        catalog.append(
            _make_candidate_variant(
                fingerprint_source=fingerprint_source,
                expert_update_policy=expert_update_policy,
                shared_update_policy=shared_update_policy,
                loss_novelty_threshold=loss_th,
                merge_threshold=merge_th,
                max_concepts=max_concepts,
                global_shared_aggregation=global_shared,
            )
        )
    return catalog


def _dataset_cache(config: LoopConfig) -> dict[int, object]:
    cache: dict[int, object] = {}
    for seed in [config.screen_seed] + [seed for seed in config.decision_seeds if seed != config.screen_seed]:
        cache[seed] = recurrence_gap._build_dataset(
            K=config.K,
            T=config.T,
            n_samples=config.n_samples,
            n_features=config.n_features,
            seed=seed,
        )
    return cache


def _evaluate_variant(
    *,
    method: str,
    variant: dict[str, object],
    datasets: dict[int, object],
    config: LoopConfig,
    seed: int,
    stage: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    result = recurrence_gap._run_variant(
        datasets[seed],
        seed=seed,
        fed_every=config.federation_every,
        epochs=config.epochs,
        lr=config.lr,
        variant=variant,
    )
    row, _ = recurrence_gap._row_from_result(
        method=method,
        seed=seed,
        stage=stage,
        result=result,
        variant=variant,
        phase3_start=config.phase3_start,
        recovery_eval_t=recurrence_gap._next_federation_t(
            config.phase3_start,
            config.federation_every,
            config.T,
        ),
    )
    round_rows = recurrence_gap._phase_a_round_rows(
        method=method,
        seed=seed,
        stage=stage,
        result=result,
    )
    return row, round_rows


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    summary_rows = recurrence_gap._aggregate_rows(
        rows,
        group_keys=["method"],
        numeric_keys=AGG_NUMERIC_KEYS,
        extra_keys=AGG_EXTRA_KEYS,
    )
    if not summary_rows:
        raise RuntimeError("expected at least one summary row")
    return summary_rows[0]


def _screen_pass(
    baseline_screen_row: dict[str, object],
    candidate_screen_row: dict[str, object],
) -> bool:
    return recurrence_gap._passes_screen_gate(baseline_screen_row, candidate_screen_row)


def _should_accept_candidate(
    champion_summary: dict[str, object],
    candidate_summary: dict[str, object],
    linear_summary: dict[str, object],
) -> tuple[bool, str]:
    c_final = float(candidate_summary["mean_final"])
    c_phase3 = float(candidate_summary["mean_phase3"])
    c_recovery = float(candidate_summary["mean_recovery_next_fed"])
    c_bytes = float(candidate_summary["mean_bytes"])
    h_final = float(champion_summary["mean_final"])
    h_phase3 = float(champion_summary["mean_phase3"])
    h_recovery = float(champion_summary["mean_recovery_next_fed"])
    h_bytes = float(champion_summary["mean_bytes"])
    l_bytes = float(linear_summary["mean_bytes"])

    if c_bytes > 2.5 * l_bytes:
        return False, "rejected: candidate violates the 2.5x linear bytes cap"

    dominates = (
        c_final >= h_final - 1e-6
        and c_phase3 >= h_phase3 - 1e-6
        and c_recovery >= h_recovery - 1e-6
        and c_bytes <= h_bytes + 1e-6
        and (
            c_final > h_final + 1e-6
            or c_phase3 > h_phase3 + 1e-6
            or c_recovery > h_recovery + 1e-6
            or c_bytes < h_bytes - 1e-6
        )
    )
    if dominates:
        return True, "accepted: candidate weakly dominates the current champion"

    material_gain = (
        c_final >= h_final + 0.01
        and c_phase3 >= h_phase3 - 0.005
        and c_recovery >= h_recovery - 0.005
        and c_bytes <= h_bytes * 1.10
    )
    if material_gain:
        return True, "accepted: candidate delivers material final-accuracy gain within the bytes guardrail"

    return False, "rejected: candidate does not clear the champion-improvement gate"


def _initial_state() -> dict[str, Any]:
    return {
        "round_index": 0,
        "tried_signatures": [],
        "accepted_signatures": [],
        "champion_signature": None,
        "champion_variant": None,
        "champion_summary": None,
        "baseline_summary": None,
        "linear_summary": None,
    }


def _ensure_baselines(
    *,
    state: dict[str, Any],
    datasets: dict[int, object],
    config: LoopConfig,
) -> tuple[dict[str, object], dict[str, object], dict[int, dict[str, object]]]:
    baseline_variant = recurrence_gap.VARIANTS["feature_local_shared"]
    linear_variant = recurrence_gap.VARIANTS["linear_split"]
    baseline_rows: list[dict[str, object]] = []
    linear_rows: list[dict[str, object]] = []
    baseline_by_seed: dict[int, dict[str, object]] = {}

    for seed in config.decision_seeds:
        baseline_row, _ = _evaluate_variant(
            method="feature_local_shared",
            variant=baseline_variant,
            datasets=datasets,
            config=config,
            seed=seed,
            stage="baseline",
        )
        linear_row, _ = _evaluate_variant(
            method="linear_split",
            variant=linear_variant,
            datasets=datasets,
            config=config,
            seed=seed,
            stage="baseline",
        )
        baseline_rows.append(baseline_row)
        linear_rows.append(linear_row)
        baseline_by_seed[seed] = baseline_row

    baseline_summary = _summarize_rows(baseline_rows)
    linear_summary = _summarize_rows(linear_rows)
    state["baseline_summary"] = baseline_summary
    state["linear_summary"] = linear_summary
    if state["champion_summary"] is None:
        state["champion_summary"] = baseline_summary
        state["champion_variant"] = copy.deepcopy(baseline_variant)
        state["champion_signature"] = _variant_signature(baseline_variant)
    return baseline_summary, linear_summary, baseline_by_seed


def run_loop(config: LoopConfig, *, resume: bool) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    state_path = config.results_dir / "loop_state.json"
    state = _load_json(state_path) if resume and state_path.exists() else _initial_state()
    datasets = _dataset_cache(config)
    baseline_summary, linear_summary, baseline_by_seed = _ensure_baselines(
        state=state,
        datasets=datasets,
        config=config,
    )

    for _ in range(config.rounds):
        catalog = _candidate_catalog(state.get("champion_summary"))
        candidate = next(
            (
                variant
                for variant in catalog
                if _variant_signature(variant) not in set(state["tried_signatures"])
            ),
            None,
        )
        if candidate is None:
            break

        round_index = int(state["round_index"]) + 1
        candidate_signature = _variant_signature(candidate)
        round_dir = config.results_dir / f"round_{round_index:04d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        method = f"auto_round_{round_index:04d}"
        candidate["method"] = method
        screen_row, screen_phase_a = _evaluate_variant(
            method=method,
            variant=candidate,
            datasets=datasets,
            config=config,
            seed=config.screen_seed,
            stage="screen",
        )
        passed_screen = _screen_pass(baseline_by_seed[config.screen_seed], screen_row)

        decision_rows = [screen_row]
        decision_phase_a = list(screen_phase_a)
        if passed_screen:
            for seed in config.decision_seeds:
                if seed == config.screen_seed:
                    continue
                row, phase_a_rows = _evaluate_variant(
                    method=method,
                    variant=candidate,
                    datasets=datasets,
                    config=config,
                    seed=seed,
                    stage="decision",
                )
                decision_rows.append(row)
                decision_phase_a.extend(phase_a_rows)

        candidate_summary = _summarize_rows(decision_rows)
        keep, keep_reason = _should_accept_candidate(
            state["champion_summary"],
            candidate_summary,
            linear_summary,
        ) if passed_screen else (False, "rejected: candidate failed the seed=42 screening gate")

        root_cause_lines = summarize_root_cause(
            [baseline_by_seed[config.screen_seed]],
            [screen_row],
        )
        decision_payload = {
            "round_index": round_index,
            "candidate_signature": candidate_signature,
            "screen_passed": passed_screen,
            "decision": "keep" if keep else "revert",
            "reason": keep_reason,
            "root_cause_summary": root_cause_lines,
            "candidate_summary": candidate_summary,
            "champion_before": state["champion_summary"],
            "baseline_summary": baseline_summary,
            "linear_summary": linear_summary,
        }

        _write_json(round_dir / "candidate_variant.json", candidate)
        _write_json(round_dir / "screen_row.json", screen_row)
        _write_json(round_dir / "decision_rows.json", decision_rows)
        _write_json(round_dir / "phase_a_round_diagnostics.json", decision_phase_a)
        _write_json(round_dir / "decision.json", decision_payload)
        (round_dir / "root_cause_summary.txt").write_text(
            "\n".join(root_cause_lines) + "\n",
            encoding="utf-8",
        )

        state["tried_signatures"].append(candidate_signature)
        state["round_index"] = round_index
        if keep:
            state["accepted_signatures"].append(candidate_signature)
            state["champion_signature"] = candidate_signature
            state["champion_variant"] = copy.deepcopy(candidate)
            state["champion_summary"] = candidate_summary
        _write_json(state_path, state)

        print(
            f"[round {round_index:04d}] {decision_payload['decision']} "
            f"{candidate_signature} | {keep_reason}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adapter scaffold research loop")
    parser.add_argument("--results-dir", type=Path, default=Path("results_adapter_research_loop"))
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--screen-seed", type=int, default=42)
    parser.add_argument("--decision-seeds", type=int, nargs="+", default=[42, 123, 456])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_loop(
        LoopConfig(
            results_dir=args.results_dir,
            rounds=args.rounds,
            screen_seed=args.screen_seed,
            decision_seeds=list(args.decision_seeds),
        ),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
