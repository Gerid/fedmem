"""E6: Real/semi-real data consistency experiments.

Runs FedAvg, IFCA, FedProto, and FedProTrack on Rotating MNIST
to validate that the phase-boundary direction is not a synthetic artifact.

Usage:
    python run_e6_real_data.py [--results-dir DIR] [--seeds 42,123,456]
           [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fedprotrack.real_data.rotating_mnist import (
    RotatingMNISTConfig,
    generate_rotating_mnist_dataset,
)
from fedprotrack.posterior import FedProTrackRunner, make_plan_c_config
from fedprotrack.baselines.runners import (
    run_ifca_full,
    run_feddrift_full,
    run_fedproto_full,
)
from fedprotrack.experiment.baselines import run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog, MetricsResult
from fedprotrack.experiments.method_registry import identity_metrics_valid
from fedprotrack.experiments.tables import generate_main_table, export_summary_csv
from fedprotrack.experiments.figures import (
    generate_axis_sweep_plot,
    generate_difference_heatmap,
)


def _make_log(name: str, acc: np.ndarray, pred: np.ndarray,
              gt: np.ndarray, total_bytes: float = 0.0) -> ExperimentLog:
    return ExperimentLog(
        ground_truth=gt,
        predicted=pred,
        accuracy_curve=acc,
        total_bytes=total_bytes if total_bytes > 0 else None,
        method_name=name,
    )


def run_single_setting(
    rm_config: RotatingMNISTConfig,
    seed: int,
) -> dict[str, MetricsResult]:
    """Run all E6 methods on one Rotating MNIST setting."""
    ds = generate_rotating_mnist_dataset(rm_config)
    gt = ds.concept_matrix
    results: dict[str, MetricsResult] = {}

    def _compute(name: str, log: ExperimentLog) -> MetricsResult:
        return compute_all_metrics(log, identity_capable=identity_metrics_valid(name))

    # FedProTrack
    fpt_runner = FedProTrackRunner(
        config=make_plan_c_config(),
        seed=seed,
        lr=0.05,
        n_epochs=3,
        soft_aggregation=True,
        blend_alpha=0.0,
        model_type="feature_adapter",
        hidden_dim=64,
        adapter_dim=16,
    )
    fpt_res = fpt_runner.run(ds)
    results["FedProTrack"] = _compute("FedProTrack", fpt_res.to_experiment_log())

    # FedAvg
    exp_cfg = ExperimentConfig(generator_config=ds.config)
    fa_res = run_fedavg_baseline(exp_cfg, dataset=ds)
    fa_log = _make_log("FedAvg", fa_res.accuracy_matrix,
                       fa_res.predicted_concept_matrix, gt)
    results["FedAvg"] = _compute("FedAvg", fa_log)

    # IFCA
    ifca_res = run_ifca_full(ds)
    results["IFCA"] = _compute("IFCA", ifca_res.to_experiment_log(gt))

    # FedProto
    fp_res = run_fedproto_full(ds)
    results["FedProto"] = _compute("FedProto", fp_res.to_experiment_log(gt))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="E6: Real-data consistency")
    parser.add_argument("--results-dir", default="results_phase3_e6",
                        help="Output directory")
    parser.add_argument("--seeds", default="42,123,456,789,1024",
                        help="Comma-separated random seeds")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Grid for Rotating MNIST
    if args.quick:
        rho_values = [5.0]
        alpha_values = [0.0, 0.5, 1.0]
        delta_values = [0.3, 0.7]
        K, T, n_samples, n_features = 5, 8, 200, 10
    else:
        rho_values = [2.0, 5.0, 10.0]
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        delta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        K, T, n_samples, n_features = 5, 10, 200, 20

    all_results: dict[str, list[MetricsResult]] = {}
    by_alpha: dict[float, dict[str, list[MetricsResult]]] = {}
    by_delta: dict[float, dict[str, list[MetricsResult]]] = {}
    # Phase diagram grid
    phase_grid: dict[tuple[float, float], dict[str, list[MetricsResult]]] = {}

    total_settings = len(rho_values) * len(alpha_values) * len(delta_values) * len(seeds)
    print(f"Running E6 Rotating MNIST: {total_settings} settings x 4 methods")

    t_start = time.time()
    i = 0
    for rho in rho_values:
        for alpha in alpha_values:
            for delta in delta_values:
                for seed in seeds:
                    i += 1
                    rm_cfg = RotatingMNISTConfig(
                        K=K, T=T, n_samples=n_samples,
                        rho=rho, alpha=alpha, delta=delta,
                        n_features=n_features, seed=seed,
                    )
                    try:
                        metrics = run_single_setting(rm_cfg, seed)
                        for mn, mr in metrics.items():
                            all_results.setdefault(mn, []).append(mr)
                            by_alpha.setdefault(alpha, {}).setdefault(mn, []).append(mr)
                            by_delta.setdefault(delta, {}).setdefault(mn, []).append(mr)
                            phase_grid.setdefault((rho, delta), {}).setdefault(mn, []).append(mr)

                        reid_str = ", ".join(
                            f"{mn}={mr.concept_re_id_accuracy:.3f}"
                            if mr.concept_re_id_accuracy is not None
                            else f"{mn}=--"
                            for mn, mr in metrics.items()
                        )
                        print(f"  [{i}/{total_settings}] rho={rho} a={alpha} d={delta} s={seed} -> {reid_str}")
                    except Exception as e:
                        print(f"  [{i}/{total_settings}] ERROR: {e}")

    elapsed = time.time() - t_start
    print(f"\nE6 completed in {elapsed:.1f}s")

    # Summary
    summary: dict[str, dict] = {}
    for mn, results in all_results.items():
        reids = [r.concept_re_id_accuracy for r in results
                 if r.concept_re_id_accuracy is not None]
        accs = [r.final_accuracy for r in results if r.final_accuracy is not None]
        switches = [r.assignment_switch_rate for r in results
                    if r.assignment_switch_rate is not None]
        singletons = [r.singleton_group_ratio for r in results
                      if r.singleton_group_ratio is not None]
        routings = [r.routing_consistency for r in results
                    if r.routing_consistency is not None]
        reuse = [r.memory_reuse_rate for r in results
                 if r.memory_reuse_rate is not None]
        summary[mn] = {
            "n_settings": len(results),
            "mean_re_id_accuracy": float(np.mean(reids)) if reids else None,
            "std_re_id_accuracy": float(np.std(reids)) if reids else None,
            "mean_final_accuracy": float(np.mean(accs)) if accs else None,
            "mean_assignment_switch_rate": float(np.mean(switches)) if switches else None,
            "mean_singleton_group_ratio": float(np.mean(singletons)) if singletons else None,
            "mean_routing_consistency": float(np.mean(routings)) if routings else None,
            "mean_memory_reuse_rate": float(np.mean(reuse)) if reuse else None,
        }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Table 2: Real-data main table
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    generate_main_table(all_results, tables_dir / "realdata_main_table.tex")
    export_summary_csv(all_results, results_dir / "summary.csv")
    print(f"Table 2 -> {tables_dir / 'realdata_main_table.tex'}")

    # Phase diagram (alpha=0.5 slice if available)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Re-ID vs delta line plot
    if by_delta:
        methods = list(all_results.keys())
        axis_vals = sorted(by_delta.keys())
        means: dict[str, list[float]] = {m: [] for m in methods}
        stds: dict[str, list[float]] = {m: [] for m in methods}
        for dv in axis_vals:
            for m in methods:
                vals = [r.concept_re_id_accuracy for r in by_delta[dv].get(m, [])
                        if r.concept_re_id_accuracy is not None]
                means[m].append(float(np.mean(vals)) if vals else float("nan"))
                stds[m].append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        generate_axis_sweep_plot(
            axis_vals, means, "delta", "Re-ID Accuracy",
            fig_dir / "realdata_reid_vs_delta.png",
            title="Rotating MNIST: Re-ID Accuracy vs Delta",
            error_bars=stds,
        )
        print(f"Fig -> {fig_dir / 'realdata_reid_vs_delta.png'}")

    # Phase direction heatmap (FedProTrack - FedAvg)
    if phase_grid:
        from fedprotrack.experiments.figures import generate_phase_diagrams

        # Average across seeds
        averaged_grid: dict[tuple[float, float], dict[str, MetricsResult]] = {}
        for key, method_lists in phase_grid.items():
            averaged_grid[key] = {}
            for mn, mr_list in method_lists.items():
                reids = [m.concept_re_id_accuracy for m in mr_list
                         if m.concept_re_id_accuracy is not None]
                accs = [m.final_accuracy for m in mr_list
                        if m.final_accuracy is not None]
                averaged_grid[key][mn] = MetricsResult(
                    concept_re_id_accuracy=float(np.mean(reids)) if reids else None,
                    assignment_entropy=None,
                    wrong_memory_reuse_rate=None,
                    worst_window_dip=None,
                    worst_window_recovery=None,
                    budget_normalized_score=None,
                    per_client_re_id=None,
                    per_timestep_re_id=None,
                    final_accuracy=float(np.mean(accs)) if accs else None,
                    accuracy_auc=None,
                )

        if len(averaged_grid) >= 2:
            generate_difference_heatmap(
                averaged_grid, "FedProTrack", "FedAvg",
                "rho", "delta", "concept_re_id_accuracy",
                fig_dir / "realdata_phase_direction.png",
                title="Rotating MNIST: FedProTrack - FedAvg (Re-ID)",
            )
            print(f"Fig -> {fig_dir / 'realdata_phase_direction.png'}")

    # Final summary
    print("\n" + "=" * 60)
    print("E6 ROTATING MNIST RESULTS")
    print("=" * 60)
    for mn, info in sorted(summary.items()):
        reid_str = (
            f"re_id = {info['mean_re_id_accuracy']:.4f} +/- {info['std_re_id_accuracy']:.4f}"
            if info.get("mean_re_id_accuracy") is not None
            else "re_id = --"
        )
        acc_str = (
            f", final_acc = {info['mean_final_accuracy']:.4f}"
            if info.get("mean_final_accuracy") is not None
            else ""
        )
        diag_parts = []
        if info.get("mean_assignment_switch_rate") is not None:
            diag_parts.append(f"switch = {info['mean_assignment_switch_rate']:.4f}")
        if info.get("mean_singleton_group_ratio") is not None:
            diag_parts.append(f"singleton = {info['mean_singleton_group_ratio']:.4f}")
        if info.get("mean_routing_consistency") is not None:
            diag_parts.append(f"routing = {info['mean_routing_consistency']:.4f}")
        diag_str = f", {'; '.join(diag_parts)}" if diag_parts else ""
        print(f"  {mn:15s}: {reid_str}{acc_str}{diag_str} (n={info['n_settings']})")


if __name__ == "__main__":
    main()
