"""Resume diagnostics for an existing Phase 3 results directory.

Reads an existing claim_check.json and, if gate notes are non-empty,
generates the missing diagnostic artifacts (alpha_diagnostics.json,
diagnostic_summary.md) without re-running the full 375-setting grid.

Usage:
    E:/anaconda3/python.exe resume_diagnostics.py \
        --results-dir E:/fedprotrack/results_phase3_v2_sine \
        [--seeds 42,123,456,789,1024] [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fedprotrack.baselines.budget_sweep import BudgetPoint
from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_local_only,
    run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.baselines.runners import (
    run_compressed_fedavg_full,
    run_feddrift_full,
    run_fedproto_full,
    run_flash_full,
    run_ifca_full,
    run_tracked_summary_full,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.budget_metrics import compute_accuracy_auc
from fedprotrack.metrics.experiment_log import ExperimentLog, MetricsResult
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.experiments.ablations import run_module_ablation
from fedprotrack.experiments.budget_analysis import (
    run_fedprotrack_budget_points,
    generate_full_budget_frontier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_metric(value: float | None) -> str:
    """Format an optional metric for console logs."""
    if value is None:
        return "--"
    return f"{value:.3f}"


def _non_none_mean(values: list[float | None]) -> float | None:
    """Return the mean of non-None values."""
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


# ---------------------------------------------------------------------------
# Single-setting runner (copied from run_phase3_experiments for isolation)
# ---------------------------------------------------------------------------

def _make_experiment_log(
    method_name: str,
    accuracy_matrix: np.ndarray,
    predicted_concepts: np.ndarray,
    ground_truth: np.ndarray,
    total_bytes: float = 0.0,
) -> ExperimentLog:
    """Create an ExperimentLog from raw arrays."""
    return ExperimentLog(
        ground_truth=ground_truth,
        predicted=predicted_concepts,
        accuracy_curve=accuracy_matrix,
        total_bytes=total_bytes if total_bytes > 0 else None,
        method_name=method_name,
    )


def run_single_setting(
    gen_config: GeneratorConfig,
    seed: int,
    quick: bool = False,
) -> dict[str, MetricsResult]:
    """Run all methods on one parameter setting."""
    dataset = generate_drift_dataset(gen_config)
    gt = dataset.concept_matrix

    results: dict[str, MetricsResult] = {}

    # 1. FedProTrack (our method)
    fpt_cfg = TwoPhaseConfig()
    fpt_runner = FedProTrackRunner(config=fpt_cfg, seed=seed)
    fpt_result = fpt_runner.run(dataset)
    fpt_log = fpt_result.to_experiment_log()
    results["FedProTrack"] = compute_all_metrics(fpt_log)

    # 2. LocalOnly
    exp_cfg = ExperimentConfig(generator_config=gen_config)
    lo_result = run_local_only(exp_cfg, dataset=dataset)
    lo_log = _make_experiment_log(
        "LocalOnly", lo_result.accuracy_matrix,
        lo_result.predicted_concept_matrix, gt,
    )
    results["LocalOnly"] = compute_all_metrics(lo_log)

    # 3. FedAvg
    fa_result = run_fedavg_baseline(exp_cfg, dataset=dataset)
    fa_log = _make_experiment_log(
        "FedAvg", fa_result.accuracy_matrix,
        fa_result.predicted_concept_matrix, gt,
    )
    results["FedAvg"] = compute_all_metrics(fa_log)

    # 4. Oracle
    oracle_result = run_oracle_baseline(exp_cfg, dataset=dataset)
    oracle_log = _make_experiment_log(
        "Oracle", oracle_result.accuracy_matrix,
        oracle_result.predicted_concept_matrix, gt,
    )
    results["Oracle"] = compute_all_metrics(oracle_log)

    # 5. FedProto
    fp_result = run_fedproto_full(dataset)
    fp_log = fp_result.to_experiment_log(gt)
    results["FedProto"] = compute_all_metrics(fp_log)

    # 6. TrackedSummary
    ts_result = run_tracked_summary_full(dataset)
    ts_log = ts_result.to_experiment_log(gt)
    results["TrackedSummary"] = compute_all_metrics(ts_log)

    # 7. Flash
    flash_result = run_flash_full(dataset)
    flash_log = flash_result.to_experiment_log(gt)
    results["Flash"] = compute_all_metrics(flash_log)

    # 8. FedDrift
    fd_result = run_feddrift_full(dataset)
    fd_log = fd_result.to_experiment_log(gt)
    results["FedDrift"] = compute_all_metrics(fd_log)

    # 9. IFCA
    ifca_result = run_ifca_full(dataset)
    ifca_log = ifca_result.to_experiment_log(gt)
    results["IFCA"] = compute_all_metrics(ifca_log)

    # 10. CompressedFedAvg
    cfed_result = run_compressed_fedavg_full(dataset)
    cfed_log = cfed_result.to_experiment_log(gt)
    results["CompressedFedAvg"] = compute_all_metrics(cfed_log)

    return results


# ---------------------------------------------------------------------------
# Alpha diagnostics
# ---------------------------------------------------------------------------

def run_alpha_diagnostics(
    seeds: list[int],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Run a focused alpha sweep diagnostic on the default SINE setting.

    Parameters
    ----------
    seeds : list[int]
        Random seeds for multi-seed averaging.

    Returns
    -------
    dict
        Nested dict: alpha_key -> method_name -> metric_name -> value.
    """
    diagnostics: dict[str, dict[str, dict[str, float | None]]] = {}
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        per_method: dict[str, list[MetricsResult]] = {
            "FedProTrack": [],
            "IFCA": [],
            "FedDrift": [],
        }
        for seed in seeds:
            cfg = GeneratorConfig(
                K=10, T=20, n_samples=500,
                rho=5.0, alpha=alpha, delta=0.5,
                generator_type="sine", seed=seed,
            )
            metrics = run_single_setting(cfg, seed, quick=False)
            for method_name in per_method:
                per_method[method_name].append(metrics[method_name])

        diagnostics[f"alpha={alpha}"] = {}
        for method_name, result_list in per_method.items():
            diagnostics[f"alpha={alpha}"][method_name] = {
                "mean_final_accuracy": _non_none_mean(
                    [r.final_accuracy for r in result_list]
                ),
                "mean_accuracy_auc": _non_none_mean(
                    [r.accuracy_auc for r in result_list]
                ),
                "mean_worst_window_dip": _non_none_mean(
                    [r.worst_window_dip for r in result_list]
                ),
                "mean_worst_window_recovery": _non_none_mean(
                    [float(r.worst_window_recovery)
                     if r.worst_window_recovery is not None else None
                     for r in result_list]
                ),
            }
    return diagnostics


# ---------------------------------------------------------------------------
# Budget diagnostics (collect points for the default SINE setting)
# ---------------------------------------------------------------------------

def collect_default_budget_points(
    federation_every_values: list[int] | None = None,
) -> dict[str, list[BudgetPoint]]:
    """Collect budget points for the default (rho=5, alpha=0.5, delta=0.5) setting.

    Parameters
    ----------
    federation_every_values : list[int], optional
        Federation frequency values to sweep. Default [1, 2, 5, 10].

    Returns
    -------
    dict
        Method name -> list of BudgetPoint.
    """
    if federation_every_values is None:
        federation_every_values = [1, 2, 5, 10]

    cfg = GeneratorConfig(
        K=10, T=20, n_samples=500,
        rho=5.0, alpha=0.5, delta=0.5,
        generator_type="sine", seed=42,
    )
    dataset = generate_drift_dataset(cfg)

    from fedprotrack.baselines.budget_sweep import run_budget_sweep
    baseline_points = run_budget_sweep(
        dataset, federation_every_values, similarity_threshold=0.5,
    )
    fpt_points = run_fedprotrack_budget_points(
        dataset, federation_every_values, seed=42,
    )

    method_points: dict[str, list[BudgetPoint]] = {}
    for p in baseline_points + fpt_points:
        if p.method_name not in method_points:
            method_points[p.method_name] = []
        method_points[p.method_name].append(p)

    return method_points


def budget_points_to_records(
    method_points: dict[str, list[BudgetPoint]],
) -> list[dict[str, float | int | str]]:
    """Convert budget points to JSON-serialisable records."""
    records: list[dict[str, float | int | str]] = []
    for method_name, points in sorted(method_points.items()):
        for point in points:
            records.append({
                "method_name": method_name,
                "federation_every": int(point.federation_every),
                "total_bytes": float(point.total_bytes),
                "accuracy_auc": float(point.accuracy_auc),
            })
    return records


def load_budget_points_from_json(
    budget_points_path: Path,
) -> dict[str, list[BudgetPoint]]:
    """Load budget points from a JSON records file.

    Parameters
    ----------
    budget_points_path : Path
        Path to ``budget_points.json``.

    Returns
    -------
    dict
        Method name -> list of BudgetPoint.
    """
    records = json.loads(budget_points_path.read_text(encoding="utf-8"))
    method_points: dict[str, list[BudgetPoint]] = {}
    for rec in records:
        name = rec["method_name"]
        bp = BudgetPoint(
            method_name=name,
            federation_every=int(rec["federation_every"]),
            total_bytes=float(rec["total_bytes"]),
            accuracy_auc=float(rec["accuracy_auc"]),
        )
        if name not in method_points:
            method_points[name] = []
        method_points[name].append(bp)
    return method_points


# ---------------------------------------------------------------------------
# Diagnostic summary writer
# ---------------------------------------------------------------------------

def write_diagnostic_summary(
    diagnostic_path: Path,
    claim_check: dict[str, object],
    budget_method_points: dict[str, list[BudgetPoint]],
    alpha_diagnostics: dict[str, dict[str, dict[str, float | None]]],
    module_ablation_results: dict[str, MetricsResult],
) -> None:
    """Write a concise markdown summary for a failed gate run.

    Parameters
    ----------
    diagnostic_path : Path
        Output path for the markdown file.
    claim_check : dict
        The claim check artifact.
    budget_method_points : dict
        Budget points keyed by method name.
    alpha_diagnostics : dict
        Alpha sweep diagnostic results.
    module_ablation_results : dict
        Module ablation results keyed by label.
    """
    lines = [
        "# Gate Diagnostics",
        "",
        "## Gate Status",
    ]
    notes = claim_check.get("notes", [])
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No gate failures recorded.")

    lines.extend([
        "",
        "## Default Budget Points",
    ])
    for record in budget_points_to_records(budget_method_points):
        lines.append(
            f"- {record['method_name']} fe={record['federation_every']}: "
            f"bytes={record['total_bytes']:.1f}, auc={record['accuracy_auc']:.4f}"
        )

    lines.extend([
        "",
        "## Alpha Sweep Diagnostics",
    ])
    for alpha_key, per_method in alpha_diagnostics.items():
        lines.append(f"- {alpha_key}")
        for method_name, stats in per_method.items():
            lines.append(
                f"  {method_name}: final_acc={_fmt_metric(stats['mean_final_accuracy'])}, "
                f"auc={_fmt_metric(stats['mean_accuracy_auc'])}, "
                f"dip={_fmt_metric(stats['mean_worst_window_dip'])}, "
                f"recovery={_fmt_metric(stats['mean_worst_window_recovery'])}"
            )

    lines.extend([
        "",
        "## Module Ablations",
    ])
    for label, metrics in module_ablation_results.items():
        lines.append(
            f"- {label}: reid={_fmt_metric(metrics.concept_re_id_accuracy)}, "
            f"wrong_mem={_fmt_metric(metrics.wrong_memory_reuse_rate)}, "
            f"entropy={_fmt_metric(metrics.assignment_entropy)}"
        )

    diagnostic_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main resume logic
# ---------------------------------------------------------------------------

def resume_diagnostics(
    results_dir: Path,
    seeds: list[int],
    force: bool = False,
) -> dict[str, Path]:
    """Resume diagnostics for an existing results directory.

    Reads claim_check.json and, if gate notes are non-empty, generates
    missing diagnostic artifacts without re-running the main grid.

    Parameters
    ----------
    results_dir : Path
        Existing results directory (e.g. results_phase3_v2_sine).
    seeds : list[int]
        Seeds used in the original run.
    force : bool
        If True, regenerate diagnostics even if files already exist.

    Returns
    -------
    dict[str, Path]
        Mapping of artifact name to path for each generated file.

    Raises
    ------
    FileNotFoundError
        If claim_check.json is missing from the results directory.
    ValueError
        If claim_check has no gate failures (notes is empty) and force
        is False.
    """
    logs_dir = results_dir / "logs"
    claim_check_path = logs_dir / "claim_check.json"

    if not claim_check_path.exists():
        raise FileNotFoundError(
            f"claim_check.json not found in {logs_dir}. "
            f"Cannot resume diagnostics without a completed gate run."
        )

    claim_check = json.loads(claim_check_path.read_text(encoding="utf-8"))
    notes = claim_check.get("notes", [])

    if not notes and not force:
        raise ValueError(
            "claim_check.json has no gate failures (notes is empty). "
            "Use --force to regenerate diagnostics anyway."
        )

    generated: dict[str, Path] = {}

    alpha_diag_path = logs_dir / "alpha_diagnostics.json"
    budget_points_path = logs_dir / "budget_points.json"
    summary_path = logs_dir / "diagnostic_summary.md"

    # Step 1: Load or generate budget points
    if budget_points_path.exists() and not force:
        print(f"  Reusing existing budget points from {budget_points_path}")
        budget_method_points = load_budget_points_from_json(budget_points_path)
    else:
        print("  Collecting default budget points...")
        t0 = time.time()
        budget_method_points = collect_default_budget_points()
        print(f"  Budget points collected in {time.time() - t0:.1f}s")
        budget_points_path.write_text(
            json.dumps(budget_points_to_records(budget_method_points), indent=2),
            encoding="utf-8",
        )
        generated["budget_points"] = budget_points_path

    # Step 2: Run or reuse alpha diagnostics
    if alpha_diag_path.exists() and not force:
        print(f"  Reusing existing alpha diagnostics from {alpha_diag_path}")
        alpha_diagnostics = json.loads(
            alpha_diag_path.read_text(encoding="utf-8")
        )
    else:
        print("  Running alpha sweep diagnostics...")
        t0 = time.time()
        alpha_diagnostics = run_alpha_diagnostics(seeds)
        print(f"  Alpha diagnostics completed in {time.time() - t0:.1f}s")
        alpha_diag_path.write_text(
            json.dumps(alpha_diagnostics, indent=2),
            encoding="utf-8",
        )
        generated["alpha_diagnostics"] = alpha_diag_path

    # Step 3: Run module ablations
    print("  Running module ablations...")
    ablations_dir = results_dir / "figures" / "ablations"
    ablations_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    module_ablation_results = run_module_ablation(
        GeneratorConfig(
            K=10, T=20, n_samples=500,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type="sine", seed=42,
        ),
        output_dir=ablations_dir,
    )
    print(f"  Module ablations completed in {time.time() - t0:.1f}s")

    # Step 4: Write diagnostic summary
    print("  Writing diagnostic summary...")
    write_diagnostic_summary(
        summary_path,
        claim_check,
        budget_method_points,
        alpha_diagnostics,
        module_ablation_results,
    )
    generated["diagnostic_summary"] = summary_path
    print(f"  Diagnostic summary -> {summary_path}")

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume diagnostics for an existing Phase 3 results directory.",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Path to existing results directory (e.g. results_phase3_v2_sine)",
    )
    parser.add_argument(
        "--seeds", default="42,123,456,789,1024",
        help="Comma-separated random seeds (must match original run)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate diagnostics even if files already exist",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"Results dir: {results_dir}")
    print(f"Seeds: {seeds}")
    print(f"Force: {args.force}")

    generated = resume_diagnostics(results_dir, seeds, force=args.force)

    print(f"\nDone. Generated {len(generated)} artifact(s):")
    for name, path in generated.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
