"""CIFAR-10 shrinkage coefficient (lambda-hat) diagnostic for rebuttal W3/Q3.

Runs FedProTrack on CIFAR-10 with explicit DRCT shrinkage enabled
to measure the per-round lambda distribution. Addresses the reviewer's
question: "Why does FPT not degrade gracefully to FedAvg on CIFAR-10?"

Outputs:
  - results_cifar10_lambda/lambda_diagnostic.json  (per-seed, per-round lambda)
  - results_cifar10_lambda/lambda_summary.json      (aggregated statistics)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from fedprotrack.experiment.baselines import run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.posterior import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR10RecurrenceConfig,
    generate_cifar10_recurrence_dataset,
    prepare_cifar10_recurrence_feature_cache,
)


def run_seed(seed: int, out_dir: Path) -> dict:
    """Run one seed and return diagnostics."""
    # Dataset: CIFAR-10 recurrence, matching main table settings
    cfg = CIFAR10RecurrenceConfig(
        K=20, T=100, n_samples=200, rho=25, alpha=0.5, delta=0.5,
        n_features=128, batch_size=64, n_workers=0,
        data_root=".cifar10_cache",
        feature_cache_dir=".feature_cache",
        feature_seed=2718,
        samples_per_class=20,
        seed=seed,
    )
    prepare_cifar10_recurrence_feature_cache(cfg)
    dataset = generate_cifar10_recurrence_dataset(cfg)
    n_concepts = int(dataset.concept_matrix.max()) + 1

    # FedProTrack with DRCT shrinkage enabled (to measure lambda explicitly)
    config = TwoPhaseConfig(
        omega=2.0,
        kappa=0.7,
        novelty_threshold=0.25,
        loss_novelty_threshold=0.15,
        sticky_dampening=1.5,
        sticky_posterior_gate=0.35,
        merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, n_concepts + 3),
        merge_every=2,
        shrink_every=6,
        drct_shrinkage=True,   # Enable explicit shrinkage
        drct_d_eff_ratio=0.9,
        drct_min_concepts=2,
    )
    runner = FedProTrackRunner(
        config=config,
        seed=seed,
        federation_every=1,
        detector_name="ADWIN",
        lr=0.01,
        n_epochs=5,
        soft_aggregation=False,  # Use hard aggregation + explicit DRCT
        blend_alpha=0.0,
    )

    t0 = time.time()
    result = runner.run(dataset)
    elapsed = time.time() - t0

    # Also run FedAvg for comparison
    exp_cfg = ExperimentConfig(generator_config=dataset.config, federation_every=1)
    fedavg_result = run_fedavg_baseline(
        exp_cfg, dataset=dataset, lr=0.01, n_epochs=5, seed=seed,
    )

    # Extract lambda log
    lambda_log = result.drct_lambda_log

    # Per-round lambda means
    round_lambdas = [(entry["round"], entry["lambda_mean"]) for entry in lambda_log]

    # Summary stats
    all_means = [entry["lambda_mean"] for entry in lambda_log]

    return {
        "seed": seed,
        "fpt_final_accuracy": float(result.accuracy_matrix[:, -1].mean()),
        "fpt_mean_accuracy": float(result.accuracy_matrix.mean()),
        "fedavg_final_accuracy": float(fedavg_result.accuracy_matrix[:, -1].mean()),
        "fedavg_mean_accuracy": float(fedavg_result.accuracy_matrix.mean()),
        "n_rounds_with_shrinkage": len(lambda_log),
        "lambda_overall_mean": float(np.mean(all_means)) if all_means else None,
        "lambda_overall_std": float(np.std(all_means)) if all_means else None,
        "lambda_overall_min": float(np.min(all_means)) if all_means else None,
        "lambda_overall_max": float(np.max(all_means)) if all_means else None,
        "lambda_first_5_rounds": float(np.mean(all_means[:5])) if len(all_means) >= 5 else None,
        "lambda_last_5_rounds": float(np.mean(all_means[-5:])) if len(all_means) >= 5 else None,
        "round_lambdas": round_lambdas,
        "full_lambda_log": lambda_log,
        "elapsed_s": elapsed,
    }


def main() -> None:
    out_dir = Path("results_cifar10_lambda")
    out_dir.mkdir(exist_ok=True)

    seeds = [42, 43, 44]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  CIFAR-10 Lambda Diagnostic — seed {seed}")
        print(f"{'='*60}")
        result = run_seed(seed, out_dir)
        all_results.append(result)

        print(f"  FPT final acc: {result['fpt_final_accuracy']:.4f}")
        print(f"  FedAvg final acc: {result['fedavg_final_accuracy']:.4f}")
        print(f"  Lambda mean: {result['lambda_overall_mean']}")
        print(f"  Lambda std: {result['lambda_overall_std']}")
        print(f"  Lambda first 5 rounds: {result['lambda_first_5_rounds']}")
        print(f"  Lambda last 5 rounds: {result['lambda_last_5_rounds']}")

    # Save per-seed results
    with open(out_dir / "lambda_diagnostic.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Aggregate summary
    fpt_accs = [r["fpt_final_accuracy"] for r in all_results]
    fedavg_accs = [r["fedavg_final_accuracy"] for r in all_results]
    lam_means = [r["lambda_overall_mean"] for r in all_results if r["lambda_overall_mean"] is not None]

    summary = {
        "fpt_final_acc_mean": float(np.mean(fpt_accs)),
        "fpt_final_acc_std": float(np.std(fpt_accs)),
        "fedavg_final_acc_mean": float(np.mean(fedavg_accs)),
        "fedavg_final_acc_std": float(np.std(fedavg_accs)),
        "lambda_grand_mean": float(np.mean(lam_means)) if lam_means else None,
        "lambda_grand_std": float(np.std(lam_means)) if lam_means else None,
        "n_seeds": len(seeds),
    }

    with open(out_dir / "lambda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  AGGREGATE SUMMARY")
    print(f"{'='*60}")
    print(f"  FPT acc: {summary['fpt_final_acc_mean']:.4f} ± {summary['fpt_final_acc_std']:.4f}")
    print(f"  FedAvg acc: {summary['fedavg_final_acc_mean']:.4f} ± {summary['fedavg_final_acc_std']:.4f}")
    print(f"  Lambda: {summary['lambda_grand_mean']:.4f} ± {summary['lambda_grand_std']:.4f}")
    print(f"\n  Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
