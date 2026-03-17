"""Phase 3 experiment suite for FedProTrack (NeurIPS 2026).

Runs all methods across a (rho x alpha x delta x generator x seed) grid,
computes paper metrics, generates LaTeX tables, phase diagrams, budget
frontiers, ablation studies, and scalability experiments.

Usage:
    python run_phase3_experiments.py [--quick] [--results-dir DIR]
           [--generators sine,sea,circle] [--seeds 42,123,456]
           [--skip-ablation] [--skip-scalability]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_local_only,
    run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig, ExperimentRunner
from fedprotrack.baselines.runners import (
    MethodResult,
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
from fedprotrack.experiments.tables import (
    generate_main_table,
    generate_per_axis_table,
    generate_overhead_table,
    export_summary_csv,
)
from fedprotrack.experiments.figures import (
    generate_accuracy_curves,
    generate_axis_sweep_plot,
    generate_difference_heatmap,
    generate_dip_recovery_boxplot,
    generate_phase_diagrams,
)
from fedprotrack.experiments.budget_analysis import generate_full_budget_frontier
from fedprotrack.experiments.ablations import AblationConfig, run_ablation_study, run_module_ablation
from fedprotrack.experiments.method_registry import identity_metrics_valid
from fedprotrack.experiments.scalability import run_scalability_K, run_scalability_T


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid(
    quick: bool,
    generators: list[str],
    seeds: list[int],
) -> list[tuple[GeneratorConfig, int]]:
    """Build (GeneratorConfig, seed) pairs for the full experiment grid."""
    if quick:
        rho_values = [5.0]
        alpha_values = [0.0, 0.5, 1.0]
        delta_values = [0.3, 0.7]
        K, T, n_samples = 5, 8, 200
    else:
        rho_values = [2.0, 5.0, 10.0]
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        delta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        K, T, n_samples = 10, 20, 500

    grid: list[tuple[GeneratorConfig, int]] = []
    for gen_type in generators:
        for rho in rho_values:
            for alpha in alpha_values:
                for delta in delta_values:
                    for seed in seeds:
                        cfg = GeneratorConfig(
                            K=K, T=T, n_samples=n_samples,
                            rho=rho, alpha=alpha, delta=delta,
                            generator_type=gen_type, seed=seed,
                        )
                        grid.append((cfg, seed))
    return grid


# ---------------------------------------------------------------------------
# Running a single setting
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

    def _compute(name: str, log: ExperimentLog) -> MetricsResult:
        return compute_all_metrics(log, identity_capable=identity_metrics_valid(name))

    # 1. FedProTrack (our method)
    fpt_cfg = TwoPhaseConfig()
    fpt_runner = FedProTrackRunner(config=fpt_cfg, seed=seed)
    fpt_result = fpt_runner.run(dataset)
    fpt_log = fpt_result.to_experiment_log()
    results["FedProTrack"] = _compute("FedProTrack", fpt_log)

    # 2. LocalOnly
    exp_cfg = ExperimentConfig(generator_config=gen_config)
    lo_result = run_local_only(exp_cfg, dataset=dataset)
    lo_log = _make_experiment_log(
        "LocalOnly", lo_result.accuracy_matrix,
        lo_result.predicted_concept_matrix, gt,
    )
    results["LocalOnly"] = _compute("LocalOnly", lo_log)

    # 3. FedAvg
    fa_result = run_fedavg_baseline(exp_cfg, dataset=dataset)
    fa_log = _make_experiment_log(
        "FedAvg", fa_result.accuracy_matrix,
        fa_result.predicted_concept_matrix, gt,
    )
    results["FedAvg"] = _compute("FedAvg", fa_log)

    # 4. Oracle
    oracle_result = run_oracle_baseline(exp_cfg, dataset=dataset)
    oracle_log = _make_experiment_log(
        "Oracle", oracle_result.accuracy_matrix,
        oracle_result.predicted_concept_matrix, gt,
    )
    results["Oracle"] = _compute("Oracle", oracle_log)

    # 5. FedProto
    fp_result = run_fedproto_full(dataset)
    fp_log = fp_result.to_experiment_log(gt)
    results["FedProto"] = _compute("FedProto", fp_log)

    # 6. TrackedSummary
    ts_result = run_tracked_summary_full(dataset)
    ts_log = ts_result.to_experiment_log(gt)
    results["TrackedSummary"] = _compute("TrackedSummary", ts_log)

    # 7. Flash
    flash_result = run_flash_full(dataset)
    flash_log = flash_result.to_experiment_log(gt)
    results["Flash"] = _compute("Flash", flash_log)

    # 8. FedDrift
    fd_result = run_feddrift_full(dataset)
    fd_log = fd_result.to_experiment_log(gt)
    results["FedDrift"] = _compute("FedDrift", fd_log)

    # 9. IFCA
    ifca_result = run_ifca_full(dataset)
    ifca_log = ifca_result.to_experiment_log(gt)
    results["IFCA"] = _compute("IFCA", ifca_log)

    # 10. CompressedFedAvg
    cfed_result = run_compressed_fedavg_full(dataset)
    cfed_log = cfed_result.to_experiment_log(gt)
    results["CompressedFedAvg"] = _compute("CompressedFedAvg", cfed_log)

    return results


# ---------------------------------------------------------------------------
# Helper: extract axis-sweep data from by_axis dicts
# ---------------------------------------------------------------------------

def _extract_axis_sweep(
    by_axis: dict[float, dict[str, list[MetricsResult]]],
    metric: str,
    methods: list[str] | None = None,
) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]]]:
    """Extract mean and std for a metric across axis values.

    Returns
    -------
    axis_values, method_means, method_stds
    """
    axis_values = sorted(by_axis.keys())
    if methods is None:
        methods = list(next(iter(by_axis.values())).keys())

    method_means: dict[str, list[float]] = {m: [] for m in methods}
    method_stds: dict[str, list[float]] = {m: [] for m in methods}

    for av in axis_values:
        for m in methods:
            results = by_axis[av].get(m, [])
            vals = []
            for r in results:
                v = getattr(r, metric, None)
                if v is not None:
                    vals.append(float(v))
            method_means[m].append(float(np.mean(vals)) if vals else float("nan"))
            method_stds[m].append(float(np.std(vals)) if len(vals) > 1 else 0.0)

    return axis_values, method_means, method_stds


# ---------------------------------------------------------------------------
# E4 helper: budget study across alpha values
# ---------------------------------------------------------------------------

def _run_budget_study(
    generators: list[str],
    alpha_values: list[float],
    federation_every_values: list[int],
    quick: bool,
    results_dir: Path,
) -> None:
    """Run E4 budget-regime crossover study."""
    budget_dir = results_dir / "figures" / "budget"
    budget_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables" / "appendix"
    tables_dir.mkdir(parents=True, exist_ok=True)

    K = 5 if quick else 10
    T = 8 if quick else 20
    n_samples = 200 if quick else 500

    # For budget x alpha heatmap: best method at each (fe, alpha)
    best_method_grid: list[list[str]] = []
    budget_labels: list[str] = [f"fe={fe}" for fe in federation_every_values]

    for fe in federation_every_values:
        best_at_alpha: list[str] = []
        for alpha in alpha_values:
            cfg = GeneratorConfig(
                K=K, T=T, n_samples=n_samples,
                rho=5.0, alpha=alpha, delta=0.5,
                generator_type=generators[0], seed=42,
            )
            dataset = generate_drift_dataset(cfg)

            # Run key methods at this federation frequency
            method_aucs: dict[str, float] = {}

            # FedProTrack
            fpt_runner = FedProTrackRunner(
                config=TwoPhaseConfig(), federation_every=fe, seed=42,
            )
            fpt_result = fpt_runner.run(dataset)
            method_aucs["FedProTrack"] = float(compute_accuracy_auc(fpt_result.accuracy_matrix))

            # FedAvg
            exp_cfg = ExperimentConfig(generator_config=cfg)
            fa_result = run_fedavg_baseline(exp_cfg, dataset=dataset)
            method_aucs["FedAvg"] = float(compute_accuracy_auc(fa_result.accuracy_matrix))

            # FedProto
            fp_result = run_fedproto_full(dataset)
            method_aucs["FedProto"] = float(compute_accuracy_auc(fp_result.accuracy_matrix))

            # IFCA
            ifca_result = run_ifca_full(dataset)
            method_aucs["IFCA"] = float(compute_accuracy_auc(ifca_result.accuracy_matrix))

            best = max(method_aucs, key=lambda m: method_aucs[m])
            best_at_alpha.append(best)

        best_method_grid.append(best_at_alpha)

    # Generate budget frontier for default setting
    try:
        budget_cfg = GeneratorConfig(
            K=K, T=T, n_samples=n_samples,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type=generators[0], seed=42,
        )
        budget_dataset = generate_drift_dataset(budget_cfg)
        generate_full_budget_frontier(
            budget_dataset, budget_dir / "budget_frontier.png",
            federation_every_values=federation_every_values,
        )
    except Exception as e:
        print(f"  Budget frontier failed: {e}")

    # Generate budget x alpha heatmap
    from fedprotrack.experiments.figures import generate_budget_alpha_heatmap
    generate_budget_alpha_heatmap(
        alpha_values, budget_labels, best_method_grid,
        budget_dir / "budget_alpha_best_method.png",
    )
    print(f"  Budget alpha heatmap -> {budget_dir / 'budget_alpha_best_method.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FedProTrack Phase 3 Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid for fast testing")
    parser.add_argument("--results-dir", default="results_phase3",
                        help="Output directory")
    parser.add_argument("--generators", default="sine",
                        help="Comma-separated generator types (sine,sea,circle)")
    parser.add_argument("--seeds", default="42",
                        help="Comma-separated random seeds")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-scalability", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 = all cores, 1 = serial)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    generators = [g.strip() for g in args.generators.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print(f"Generators: {generators}")
    print(f"Seeds: {seeds}")
    print(f"Quick mode: {args.quick}")

    # -----------------------------------------------------------------------
    # 1. Main experiments
    # -----------------------------------------------------------------------
    grid = build_grid(args.quick, generators, seeds)
    n_jobs = args.n_jobs
    n_cores = os.cpu_count() or 1
    effective_jobs = n_cores if n_jobs == -1 else min(n_jobs, len(grid))
    print(f"\nRunning {len(grid)} settings x 10 methods "
          f"({effective_jobs} parallel workers)...")

    all_results: dict[str, list[MetricsResult]] = {}
    # Track by axis for per-axis tables and E1 line plots
    by_rho: dict[float, dict[str, list[MetricsResult]]] = {}
    by_alpha: dict[float, dict[str, list[MetricsResult]]] = {}
    by_delta: dict[float, dict[str, list[MetricsResult]]] = {}
    # For phase diagrams — per alpha slice
    phase_by_alpha: dict[float, dict[tuple[float, float], dict[str, list[MetricsResult]]]] = {}

    t_start = time.time()

    def _run_one(idx_cfg_seed: tuple[int, GeneratorConfig, int]) -> tuple[GeneratorConfig, int, dict[str, MetricsResult] | None, str]:
        idx, gen_cfg, seed = idx_cfg_seed
        setting = f"{gen_cfg.generator_type}_rho{gen_cfg.rho}_a{gen_cfg.alpha}_d{gen_cfg.delta}_s{seed}"
        try:
            metrics = run_single_setting(gen_cfg, seed, args.quick)
            acc_str = ", ".join(
                f"{mn}={mr.concept_re_id_accuracy:.3f}"
                if mr.concept_re_id_accuracy is not None
                else f"{mn}=--"
                for mn, mr in metrics.items()
            )
            return gen_cfg, seed, metrics, f"  [{idx+1}/{len(grid)}] {setting} -> {acc_str}"
        except Exception as e:
            return gen_cfg, seed, None, f"  [{idx+1}/{len(grid)}] {setting} ERROR: {e}"

    tasks = [(i, cfg, s) for i, (cfg, s) in enumerate(grid)]

    def _accumulate(gen_cfg: GeneratorConfig, seed: int, metrics: dict[str, MetricsResult] | None, msg: str) -> None:
        print(msg, flush=True)
        if metrics is None:
            return

        # Accumulate
        for method_name, mr in metrics.items():
            if method_name not in all_results:
                all_results[method_name] = []
            all_results[method_name].append(mr)

        # By axis
        for axis_dict, axis_val in [
            (by_rho, gen_cfg.rho),
            (by_alpha, gen_cfg.alpha),
            (by_delta, gen_cfg.delta),
        ]:
            if axis_val not in axis_dict:
                axis_dict[axis_val] = {}
            for mn, mr in metrics.items():
                if mn not in axis_dict[axis_val]:
                    axis_dict[axis_val][mn] = []
                axis_dict[axis_val][mn].append(mr)

        # Phase diagram: collect per alpha slice
        alpha_key = round(gen_cfg.alpha, 2)
        if alpha_key not in phase_by_alpha:
            phase_by_alpha[alpha_key] = {}
        rd_key = (gen_cfg.rho, gen_cfg.delta)
        if rd_key not in phase_by_alpha[alpha_key]:
            phase_by_alpha[alpha_key][rd_key] = {}
        for mn, mr in metrics.items():
            if mn not in phase_by_alpha[alpha_key][rd_key]:
                phase_by_alpha[alpha_key][rd_key][mn] = []
            phase_by_alpha[alpha_key][rd_key][mn].append(mr)

    if n_jobs == 1:
        # Sequential: run in-process for real-time streaming output
        for t in tasks:
            gen_cfg, seed, metrics, msg = _run_one(t)
            _accumulate(gen_cfg, seed, metrics, msg)
    else:
        job_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one)(t) for t in tasks
        )
        for gen_cfg, seed, metrics, msg in job_results:
            _accumulate(gen_cfg, seed, metrics, msg)

    elapsed = time.time() - t_start
    print(f"\nMain experiments completed in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Save summary JSON + CSV
    # -----------------------------------------------------------------------
    summary: dict[str, dict] = {}
    for method_name, results_list in all_results.items():
        reid_vals = [r.concept_re_id_accuracy for r in results_list
                     if r.concept_re_id_accuracy is not None]
        fa_vals = [r.final_accuracy for r in results_list if r.final_accuracy is not None]
        auc_vals = [r.accuracy_auc for r in results_list if r.accuracy_auc is not None]
        summary[method_name] = {
            "n_settings": len(results_list),
            "mean_re_id_accuracy": float(np.mean(reid_vals)) if reid_vals else None,
            "std_re_id_accuracy": float(np.std(reid_vals)) if reid_vals else None,
            "mean_final_accuracy": float(np.mean(fa_vals)) if fa_vals else None,
            "mean_accuracy_auc": float(np.mean(auc_vals)) if auc_vals else None,
        }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    export_summary_csv(all_results, results_dir / "summary.csv")
    print(f"\nSummary saved to {results_dir / 'summary.json'} and summary.csv")

    # -----------------------------------------------------------------------
    # 3. LaTeX tables (E5)
    # -----------------------------------------------------------------------
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    appendix_dir = tables_dir / "appendix"
    appendix_dir.mkdir(exist_ok=True)

    # Main table (without Oracle/LocalOnly)
    primary_methods = [
        "FedAvg", "FedProto", "TrackedSummary", "Flash",
        "FedDrift", "IFCA", "FedProTrack",
    ]
    primary_results = {m: all_results[m] for m in primary_methods if m in all_results}
    generate_main_table(primary_results, tables_dir / "main_table.tex")
    print(f"Main table -> {tables_dir / 'main_table.tex'}")

    # Main table with Oracle (appendix)
    if "Oracle" in all_results:
        with_oracle = dict(primary_results)
        with_oracle["Oracle"] = all_results["Oracle"]
        generate_main_table(with_oracle, appendix_dir / "main_table_with_oracle.tex")

    # Per-axis tables
    for axis_name, axis_dict in [("rho", by_rho), ("alpha", by_alpha), ("delta", by_delta)]:
        if axis_dict:
            generate_per_axis_table(
                axis_dict, axis_name,
                tables_dir / f"table_{axis_name}.tex",
            )
            print(f"Per-{axis_name} table -> {tables_dir / f'table_{axis_name}.tex'}")

    # -----------------------------------------------------------------------
    # 4. E1: Identity inference quality figures
    # -----------------------------------------------------------------------
    print("\nGenerating E1 identity inference figures...")
    e1_dir = results_dir / "figures" / "memory_phase"
    e1_dir.mkdir(parents=True, exist_ok=True)

    e1_methods = [
        "FedAvg", "FedProto", "TrackedSummary", "FedDrift",
        "IFCA", "Flash", "FedProTrack",
    ]

    # Re-ID vs delta
    if by_delta:
        axis_vals, means, stds = _extract_axis_sweep(
            by_delta, "concept_re_id_accuracy", e1_methods,
        )
        generate_axis_sweep_plot(
            axis_vals, means, "delta", "Re-ID Accuracy",
            e1_dir / "reid_vs_delta.png",
            title="Concept Re-ID Accuracy vs Delta",
            error_bars=stds,
        )
        print(f"  reid_vs_delta -> {e1_dir / 'reid_vs_delta.png'}")

    # Wrong memory vs delta
    if by_delta:
        axis_vals, means, stds = _extract_axis_sweep(
            by_delta, "wrong_memory_reuse_rate", e1_methods,
        )
        generate_axis_sweep_plot(
            axis_vals, means, "delta", "Wrong Memory Reuse Rate",
            e1_dir / "wrong_memory_vs_delta.png",
            title="Wrong Memory Reuse vs Delta",
            error_bars=stds,
        )
        print(f"  wrong_memory_vs_delta -> {e1_dir / 'wrong_memory_vs_delta.png'}")

    # Re-ID vs rho
    if by_rho:
        axis_vals, means, stds = _extract_axis_sweep(
            by_rho, "concept_re_id_accuracy", e1_methods,
        )
        generate_axis_sweep_plot(
            axis_vals, means, "rho", "Re-ID Accuracy",
            e1_dir / "reid_vs_rho.png",
            title="Concept Re-ID Accuracy vs Rho",
            error_bars=stds,
        )

    # Assignment entropy heatmap (for alpha=0.5 slice)
    if 0.5 in phase_by_alpha:
        phase_05 = _average_phase_grid(phase_by_alpha[0.5])
        generate_axis_sweep_plot(
            sorted(by_delta.keys()) if by_delta else [],
            {m: [getattr(phase_05.get((5.0, d), {}).get(m, MetricsResult(
                concept_re_id_accuracy=None, assignment_entropy=None,
                wrong_memory_reuse_rate=None, worst_window_dip=None,
                worst_window_recovery=None, budget_normalized_score=None,
                per_client_re_id=None, per_timestep_re_id=None,
            )), "assignment_entropy", float("nan")) for d in sorted(by_delta.keys())]
             for m in e1_methods if m in all_results},
            "delta", "Assignment Entropy",
            e1_dir / "assignment_entropy_vs_delta.png",
            title="Assignment Entropy vs Delta (rho=5, alpha=0.5)",
        )

    # -----------------------------------------------------------------------
    # 4b. E2: Drift-window adaptation speed
    # -----------------------------------------------------------------------
    print("\nGenerating E2 recovery figures...")
    e2_methods = ["FedAvg", "Flash", "FedDrift", "IFCA", "FedProTrack"]

    if by_alpha:
        # Recovery vs alpha
        axis_vals, means, stds = _extract_axis_sweep(
            by_alpha, "worst_window_dip", e2_methods,
        )
        generate_axis_sweep_plot(
            axis_vals, means, "alpha", "Worst Window Dip",
            e1_dir / "dip_vs_alpha.png",
            title="Worst-Window Dip vs Alpha",
            error_bars=stds,
        )

        axis_vals, means, stds = _extract_axis_sweep(
            by_alpha, "accuracy_auc", e2_methods,
        )
        generate_axis_sweep_plot(
            axis_vals, means, "alpha", "AUC(acc(t))",
            e1_dir / "auc_vs_alpha.png",
            title="Accuracy AUC vs Alpha",
            error_bars=stds,
        )

        # Dip/recovery boxplot across all settings
        method_dr: dict[str, list[tuple[float | None, int | None]]] = {}
        for m in e2_methods:
            if m in all_results:
                method_dr[m] = [
                    (r.worst_window_dip, r.worst_window_recovery)
                    for r in all_results[m]
                ]
        if method_dr:
            generate_dip_recovery_boxplot(
                method_dr, e1_dir / "dip_recovery_boxplot.png",
            )

    # -----------------------------------------------------------------------
    # 5. E3: Memory benefit phase diagrams (per alpha slice)
    # -----------------------------------------------------------------------
    print("\nGenerating E3 phase diagrams and difference heatmaps...")
    phase_dir = results_dir / "figures" / "phase_diagrams"

    for alpha_val, rd_lists in phase_by_alpha.items():
        alpha_tag = f"alpha{alpha_val}".replace(".", "")
        phase_grid = _average_phase_grid(rd_lists)

        if len(phase_grid) < 2:
            continue

        # Standard phase diagrams per method
        generate_phase_diagrams(
            phase_grid, "rho", "delta",
            phase_dir / alpha_tag,
        )

        # Difference heatmaps: FedProTrack - baseline
        for baseline in ["FedAvg", "FedProto", "IFCA"]:
            if baseline not in all_results:
                continue
            generate_difference_heatmap(
                phase_grid, "FedProTrack", baseline,
                "rho", "delta", "concept_re_id_accuracy",
                e1_dir / f"memory_gain_vs_{baseline.lower()}_{alpha_tag}.png",
                title=f"FedProTrack - {baseline}: Re-ID ({alpha_tag})",
            )
        print(f"  Phase diagrams + diff heatmaps for alpha={alpha_val}")

    # -----------------------------------------------------------------------
    # 6. E4: Budget-regime crossover
    # -----------------------------------------------------------------------
    print("\nRunning E4 budget study...")
    alpha_vals_budget = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 0.75, 1.0]
    fe_vals = [1, 2, 5] if args.quick else [1, 2, 5, 10]
    _run_budget_study(generators, alpha_vals_budget, fe_vals, args.quick, results_dir)
    print("  E4 budget study complete")

    # -----------------------------------------------------------------------
    # 7. Ablation studies (E7)
    # -----------------------------------------------------------------------
    if not args.skip_ablation:
        print("\nRunning ablation studies (E7)...")
        abl_cfg = AblationConfig()
        if args.quick:
            abl_cfg.gen_config = GeneratorConfig(
                K=5, T=8, n_samples=200,
                rho=5.0, alpha=0.5, delta=0.5,
                generator_type="sine", seed=42,
            )
            abl_cfg.omega_values = [0.5, 1.0, 2.0]
            abl_cfg.kappa_values = [0.7, 0.8, 0.9]
            abl_cfg.novelty_threshold_values = [0.2, 0.3]
            abl_cfg.merge_threshold_values = [0.8, 0.9]
            abl_cfg.federation_every_values = [1, 5]

        abl_dir = results_dir / "figures" / "ablations"
        run_ablation_study(abl_cfg, output_dir=abl_dir)
        print(f"  Scalar ablation plots -> {abl_dir}")

        # Module-level ablations (E7)
        print("  Running module ablations...")
        module_gen = abl_cfg.gen_config
        run_module_ablation(module_gen, output_dir=abl_dir)
        print(f"  Module ablation plots -> {abl_dir}")

    # -----------------------------------------------------------------------
    # 8. Scalability
    # -----------------------------------------------------------------------
    if not args.skip_scalability:
        print("\nRunning scalability experiments...")
        scal_dir = results_dir / "figures" / "scalability"

        if args.quick:
            k_vals = [3, 5]
            t_vals = [5, 8]
            n_samp = 200
        else:
            k_vals = [5, 10, 20]
            t_vals = [10, 20, 50]
            n_samp = 500

        run_scalability_K(K_values=k_vals, T=8 if args.quick else 20,
                          n_samples=n_samp, output_dir=scal_dir)
        run_scalability_T(T_values=t_vals, K=5 if args.quick else 10,
                          n_samples=n_samp, output_dir=scal_dir)
        print(f"  Scalability plots -> {scal_dir}")

    # -----------------------------------------------------------------------
    # 9. E9: Overhead table
    # -----------------------------------------------------------------------
    print("\nGenerating E9 overhead table...")
    overhead_cfg = GeneratorConfig(
        K=5 if args.quick else 10,
        T=8 if args.quick else 20,
        n_samples=200 if args.quick else 500,
        rho=5.0, alpha=0.5, delta=0.5,
        generator_type=generators[0], seed=42,
    )
    overhead_dataset = generate_drift_dataset(overhead_cfg)
    overhead_stats = _collect_overhead_stats(overhead_dataset, overhead_cfg)
    generate_overhead_table(overhead_stats, appendix_dir / "overhead_table.tex")
    # Also save as JSON
    with open(appendix_dir / "overhead_stats.json", "w") as f:
        json.dump(overhead_stats, f, indent=2)
    print(f"  Overhead table -> {appendix_dir / 'overhead_table.tex'}")

    # -----------------------------------------------------------------------
    # 10. Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)
    for method_name, info in sorted(summary.items()):
        fa_str = f", final_acc={info['mean_final_accuracy']:.4f}" if info.get('mean_final_accuracy') is not None else ""
        reid_str = (
            f"re_id_acc = {info['mean_re_id_accuracy']:.4f} "
            f"+/- {info['std_re_id_accuracy']:.4f}"
            if info.get("mean_re_id_accuracy") is not None
            else "re_id_acc = --"
        )
        print(f"  {method_name:18s}: {reid_str}{fa_str} "
              f"(n={info['n_settings']})")
    print(f"\nAll results saved to {results_dir}/")


# ---------------------------------------------------------------------------
# Overhead collection helper
# ---------------------------------------------------------------------------

def _collect_overhead_stats(
    dataset,
    gen_config: GeneratorConfig,
) -> dict[str, dict[str, float]]:
    """Run each method once and collect byte/time stats."""
    stats: dict[str, dict[str, float]] = {}
    gt = dataset.concept_matrix

    # FedProTrack
    t0 = time.time()
    fpt_runner = FedProTrackRunner(config=TwoPhaseConfig(), seed=42)
    fpt_result = fpt_runner.run(dataset)
    stats["FedProTrack"] = {
        "total_bytes": fpt_result.total_bytes,
        "phase_a_bytes": fpt_result.phase_a_bytes,
        "phase_b_bytes": fpt_result.phase_b_bytes,
        "wall_clock_s": time.time() - t0,
        "active_concepts": float(fpt_result.active_concepts),
        "spawned_concepts": float(fpt_result.spawned_concepts),
        "merged_concepts": float(fpt_result.merged_concepts),
        "pruned_concepts": float(fpt_result.pruned_concepts),
    }

    # FedAvg
    exp_cfg = ExperimentConfig(generator_config=gen_config)
    t0 = time.time()
    fa_result = run_fedavg_baseline(exp_cfg, dataset=dataset)
    stats["FedAvg"] = {
        "total_bytes": 0.0,
        "phase_a_bytes": 0.0,
        "phase_b_bytes": 0.0,
        "wall_clock_s": time.time() - t0,
        "active_concepts": 1.0,
    }

    # FedProto
    t0 = time.time()
    fp_result = run_fedproto_full(dataset)
    stats["FedProto"] = {
        "total_bytes": fp_result.total_bytes,
        "phase_a_bytes": fp_result.total_bytes,
        "phase_b_bytes": 0.0,
        "wall_clock_s": time.time() - t0,
        "active_concepts": 1.0,
    }

    # IFCA
    t0 = time.time()
    ifca_result = run_ifca_full(dataset)
    stats["IFCA"] = {
        "total_bytes": ifca_result.total_bytes,
        "phase_a_bytes": 0.0,
        "phase_b_bytes": ifca_result.total_bytes,
        "wall_clock_s": time.time() - t0,
        "active_concepts": float(len(set(
            ifca_result.predicted_concept_matrix[:, -1]
        ))),
    }

    # FedDrift
    t0 = time.time()
    fd_result = run_feddrift_full(dataset)
    stats["FedDrift"] = {
        "total_bytes": fd_result.total_bytes,
        "phase_a_bytes": 0.0,
        "phase_b_bytes": fd_result.total_bytes,
        "wall_clock_s": time.time() - t0,
        "active_concepts": float(len(set(
            fd_result.predicted_concept_matrix[:, -1]
        ))),
    }

    # Flash
    t0 = time.time()
    flash_result = run_flash_full(dataset)
    stats["Flash"] = {
        "total_bytes": flash_result.total_bytes,
        "phase_a_bytes": 0.0,
        "phase_b_bytes": flash_result.total_bytes,
        "wall_clock_s": time.time() - t0,
        "active_concepts": float(len(set(
            flash_result.predicted_concept_matrix[:, -1]
        ))),
    }

    return stats


# ---------------------------------------------------------------------------
# Phase grid averaging helper
# ---------------------------------------------------------------------------

def _average_phase_grid(
    rd_lists: dict[tuple[float, float], dict[str, list[MetricsResult]]],
) -> dict[tuple[float, float], dict[str, MetricsResult]]:
    """Average a list-based phase grid into single MetricsResult per cell."""
    averaged: dict[tuple[float, float], dict[str, MetricsResult]] = {}
    for key, method_lists in rd_lists.items():
        averaged[key] = {}
        for mn, mr_list in method_lists.items():
            # Identity metrics may be None for non-identity-capable methods
            reid_vals = [m.concept_re_id_accuracy for m in mr_list if m.concept_re_id_accuracy is not None]
            ent_vals = [m.assignment_entropy for m in mr_list if m.assignment_entropy is not None]
            wmrr_vals = [m.wrong_memory_reuse_rate for m in mr_list if m.wrong_memory_reuse_rate is not None]
            pcr_vals = [m.per_client_re_id for m in mr_list if m.per_client_re_id is not None]
            ptr_vals = [m.per_timestep_re_id for m in mr_list if m.per_timestep_re_id is not None]

            averaged[key][mn] = MetricsResult(
                concept_re_id_accuracy=float(np.mean(reid_vals)) if reid_vals else None,
                assignment_entropy=float(np.mean(ent_vals)) if ent_vals else None,
                wrong_memory_reuse_rate=float(np.mean(wmrr_vals)) if wmrr_vals else None,
                worst_window_dip=float(np.mean([m.worst_window_dip for m in mr_list if m.worst_window_dip is not None])) if any(m.worst_window_dip is not None for m in mr_list) else None,
                worst_window_recovery=None,
                budget_normalized_score=None,
                per_client_re_id=np.mean(pcr_vals, axis=0) if pcr_vals else None,
                per_timestep_re_id=np.mean(ptr_vals, axis=0) if ptr_vals else None,
                final_accuracy=float(np.mean([m.final_accuracy for m in mr_list if m.final_accuracy is not None])) if any(m.final_accuracy is not None for m in mr_list) else None,
                accuracy_auc=float(np.mean([m.accuracy_auc for m in mr_list if m.accuracy_auc is not None])) if any(m.accuracy_auc is not None for m in mr_list) else None,
            )
    return averaged


if __name__ == "__main__":
    main()
