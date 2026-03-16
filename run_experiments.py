"""Run the full FedProTrack experiment suite.

Compares FedProTrack against baselines across a grid of
(rho, alpha, delta) settings and generates all result figures.

Usage:
    python run_experiments.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.experiment.runner import ExperimentConfig, ExperimentRunner
from fedprotrack.experiment.baselines import (
    run_local_only,
    run_fedavg_baseline,
    run_oracle_baseline,
)
from fedprotrack.experiment.visualization import (
    plot_accuracy_over_time,
    plot_concept_matrix_comparison,
    plot_method_comparison_bar,
    generate_all_figures,
)


def build_experiment_grid(quick: bool = False) -> list[GeneratorConfig]:
    """Build parameter grid for experiments.

    Parameters
    ----------
    quick : bool
        If True, use a reduced grid for fast testing.
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

    configs = []
    for rho in rho_values:
        for alpha in alpha_values:
            for delta in delta_values:
                configs.append(GeneratorConfig(
                    K=K, T=T, n_samples=n_samples,
                    rho=rho, alpha=alpha, delta=delta,
                    generator_type="sine", seed=42,
                ))
    return configs


def run_single_setting(gen_config: GeneratorConfig, results_dir: Path) -> dict:
    """Run all methods on a single parameter setting."""
    setting_name = gen_config.dir_name
    print(f"\n{'='*60}")
    print(f"Setting: {setting_name}")
    print(f"  rho={gen_config.rho}, alpha={gen_config.alpha}, delta={gen_config.delta}")
    print(f"{'='*60}")

    # Generate dataset once, share across methods
    dataset = generate_drift_dataset(gen_config)

    exp_config = ExperimentConfig(
        generator_config=gen_config,
        method_name="FedProTrack",
        detector_name="ADWIN",
        similarity_threshold=0.6,
    )

    # Run all methods
    methods = {}

    print("  Running FedProTrack...")
    runner = ExperimentRunner(exp_config)
    methods["FedProTrack"] = runner.run(dataset=dataset)

    print("  Running LocalOnly...")
    methods["LocalOnly"] = run_local_only(exp_config, dataset=dataset)

    print("  Running FedAvg...")
    methods["FedAvg"] = run_fedavg_baseline(exp_config, dataset=dataset)

    print("  Running Oracle...")
    methods["Oracle"] = run_oracle_baseline(exp_config, dataset=dataset)

    # Save per-setting results
    setting_dir = results_dir / setting_name
    setting_dir.mkdir(parents=True, exist_ok=True)

    results_list = list(methods.values())
    for r in results_list:
        r.save(setting_dir / f"{r.method_name}.json")

    # Generate figures
    generate_all_figures(results_list, setting_dir / "figures")

    # Summary
    summary = {}
    for name, r in methods.items():
        summary[name] = r.summary
        print(f"  {name}: acc={r.mean_accuracy:.3f}, "
              f"concept_track={r.concept_tracking_accuracy:.3f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run FedProTrack experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Use reduced parameter grid for quick testing")
    parser.add_argument("--results-dir", default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    configs = build_experiment_grid(quick=args.quick)
    print(f"Running {len(configs)} experiment settings...")

    all_summaries = {}
    for gen_config in configs:
        summary = run_single_setting(gen_config, results_dir)
        all_summaries[gen_config.dir_name] = summary

    # Save overall summary
    with open(results_dir / "summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll experiments complete. Results saved to {results_dir}/")
    print(f"Total settings: {len(configs)}")

    # Print aggregate comparison
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    method_accs: dict[str, list[float]] = {}
    for setting, methods in all_summaries.items():
        for method_name, metrics in methods.items():
            if method_name not in method_accs:
                method_accs[method_name] = []
            method_accs[method_name].append(metrics["mean_accuracy"])

    for method_name, accs in sorted(method_accs.items()):
        print(f"  {method_name:15s}: mean_acc={np.mean(accs):.4f} "
              f"(+/- {np.std(accs):.4f})")


if __name__ == "__main__":
    main()
