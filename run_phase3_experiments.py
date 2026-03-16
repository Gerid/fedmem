"""Phase 3 experiment suite for FedProTrack (NeurIPS 2026).

Runs all 6 methods across a (rho × alpha × delta × generator × seed) grid,
computes 5 paper metrics, generates LaTeX tables, phase diagrams, budget
frontiers, ablation studies, and scalability experiments.

Usage:
    python run_phase3_experiments.py [--quick] [--results-dir DIR]
           [--generators sine,sea,circle] [--seeds 42,123,456]
           [--skip-ablation] [--skip-scalability]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

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
from fedprotrack.metrics.experiment_log import ExperimentLog, MetricsResult
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.experiments.tables import generate_main_table, generate_per_axis_table
from fedprotrack.experiments.figures import (
    generate_accuracy_curves,
    generate_phase_diagrams,
)
from fedprotrack.experiments.budget_analysis import generate_full_budget_frontier
from fedprotrack.experiments.ablations import AblationConfig, run_ablation_study
from fedprotrack.experiments.scalability import run_scalability_K, run_scalability_T


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_grid(
    quick: bool,
    generators: list[str],
    seeds: list[int],
) -> list[tuple[GeneratorConfig, int]]:
    """Build (GeneratorConfig, seed) pairs for the full experiment grid.

    Returns
    -------
    list of (GeneratorConfig, seed)
    """
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
    """Run all 6 methods on one parameter setting.

    Returns
    -------
    dict[str, MetricsResult]
        method_name -> MetricsResult
    """
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
    print(f"\nRunning {len(grid)} settings × 10 methods...")

    all_results: dict[str, list[MetricsResult]] = {}
    # Also track by axis for per-axis tables
    by_rho: dict[float, dict[str, list[MetricsResult]]] = {}
    by_alpha: dict[float, dict[str, list[MetricsResult]]] = {}
    by_delta: dict[float, dict[str, list[MetricsResult]]] = {}
    # For phase diagrams (fixed alpha=0.5 if available) — collect lists, average later
    phase_rho_delta_lists: dict[tuple[float, float], dict[str, list[MetricsResult]]] = {}

    # Track accuracy matrices for a representative setting
    representative_accs: dict[str, np.ndarray] | None = None

    t_start = time.time()

    for i, (gen_cfg, seed) in enumerate(grid):
        setting = f"{gen_cfg.generator_type}_rho{gen_cfg.rho}_a{gen_cfg.alpha}_d{gen_cfg.delta}_s{seed}"
        print(f"  [{i+1}/{len(grid)}] {setting}", end="")
        sys.stdout.flush()

        try:
            metrics = run_single_setting(gen_cfg, seed, args.quick)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

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

        # Phase diagram: collect for alpha=0.5 (or nearest)
        if abs(gen_cfg.alpha - 0.5) < 0.01:
            key = (gen_cfg.rho, gen_cfg.delta)
            if key not in phase_rho_delta_lists:
                phase_rho_delta_lists[key] = {}
            for mn, mr in metrics.items():
                if mn not in phase_rho_delta_lists[key]:
                    phase_rho_delta_lists[key][mn] = []
                phase_rho_delta_lists[key][mn].append(mr)

        acc_str = ", ".join(
            f"{mn}={mr.concept_re_id_accuracy:.3f}" for mn, mr in metrics.items()
        )
        print(f" → {acc_str}")

    elapsed = time.time() - t_start
    print(f"\nMain experiments completed in {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # 2. Save summary JSON
    # -----------------------------------------------------------------------
    summary = {}
    for method_name, results_list in all_results.items():
        vals = [r.concept_re_id_accuracy for r in results_list]
        summary[method_name] = {
            "n_settings": len(results_list),
            "mean_re_id_accuracy": float(np.mean(vals)),
            "std_re_id_accuracy": float(np.std(vals)),
        }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {results_dir / 'summary.json'}")

    # -----------------------------------------------------------------------
    # 3. LaTeX tables
    # -----------------------------------------------------------------------
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    generate_main_table(all_results, tables_dir / "main_table.tex")
    print(f"Main table → {tables_dir / 'main_table.tex'}")

    for axis_name, axis_dict in [("rho", by_rho), ("alpha", by_alpha), ("delta", by_delta)]:
        if axis_dict:
            generate_per_axis_table(
                axis_dict, axis_name,
                tables_dir / f"table_{axis_name}.tex",
            )
            print(f"Per-{axis_name} table → {tables_dir / f'table_{axis_name}.tex'}")

    # -----------------------------------------------------------------------
    # 4. Phase diagrams
    # -----------------------------------------------------------------------
    if phase_rho_delta_lists:
        # Average over seeds/generators per (rho, delta) cell
        phase_rho_delta: dict[tuple[float, float], dict[str, MetricsResult]] = {}
        for key, method_lists in phase_rho_delta_lists.items():
            phase_rho_delta[key] = {}
            for mn, mr_list in method_lists.items():
                K_dim = mr_list[0].per_client_re_id.shape[0]
                T_dim = mr_list[0].per_timestep_re_id.shape[0]
                phase_rho_delta[key][mn] = MetricsResult(
                    concept_re_id_accuracy=float(np.mean([m.concept_re_id_accuracy for m in mr_list])),
                    assignment_entropy=float(np.mean([m.assignment_entropy for m in mr_list])),
                    wrong_memory_reuse_rate=float(np.mean([m.wrong_memory_reuse_rate for m in mr_list])),
                    worst_window_dip=None,
                    worst_window_recovery=None,
                    budget_normalized_score=None,
                    per_client_re_id=np.mean([m.per_client_re_id for m in mr_list], axis=0),
                    per_timestep_re_id=np.mean([m.per_timestep_re_id for m in mr_list], axis=0),
                )
        figures_dir = results_dir / "figures"
        generate_phase_diagrams(
            phase_rho_delta, "rho", "delta", figures_dir / "phase_diagrams",
        )
        print(f"Phase diagrams → {figures_dir / 'phase_diagrams'}")

    # -----------------------------------------------------------------------
    # 5. Budget frontier
    # -----------------------------------------------------------------------
    try:
        # Use the first generator config for budget frontier
        budget_cfg = GeneratorConfig(
            K=5 if args.quick else 10,
            T=8 if args.quick else 20,
            n_samples=200 if args.quick else 500,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type=generators[0], seed=42,
        )
        budget_dataset = generate_drift_dataset(budget_cfg)
        budget_path = results_dir / "figures" / "budget_frontier.png"
        fe_vals = [1, 2, 5] if args.quick else [1, 2, 5, 10]
        generate_full_budget_frontier(
            budget_dataset, budget_path,
            federation_every_values=fe_vals,
        )
        print(f"Budget frontier → {budget_path}")
    except Exception as e:
        print(f"Budget frontier failed: {e}")

    # -----------------------------------------------------------------------
    # 6. Ablation studies
    # -----------------------------------------------------------------------
    if not args.skip_ablation:
        print("\nRunning ablation studies...")
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
        print(f"Ablation plots → {abl_dir}")

    # -----------------------------------------------------------------------
    # 7. Scalability
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
        print(f"Scalability plots → {scal_dir}")

    # -----------------------------------------------------------------------
    # 8. Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)
    for method_name, info in sorted(summary.items()):
        print(f"  {method_name:18s}: re_id_acc = {info['mean_re_id_accuracy']:.4f} "
              f"± {info['std_re_id_accuracy']:.4f} "
              f"(n={info['n_settings']})")
    print(f"\nAll results saved to {results_dir}/")


if __name__ == "__main__":
    main()
