"""Clean Scheme A clustering error measurement for rebuttal W1.

Tests whether η_A^(clean) ≥ η_C even when Scheme A methods receive
the correct (concept-matched) initialization — eliminating V(w^init).

If η_A^(clean) > η_C: validates Assumption A5 (information indirectness).
If η_A^(clean) ≈ η_C: the theorem's value is purely in the stale-init term.

On CIFAR-100 recurrence (K=20, T=100, ρ=25), we run:
  - FPT (Scheme C): standard fingerprint clustering → η_C
  - IFCA-Clean: IFCA with Oracle-provided concept-matched init → η_A^(clean)
  - IFCA-Standard: standard IFCA with stale init → η_A (for comparison)

Outputs:
  - results_clean_scheme_a/clean_eta_results.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from fedprotrack.baselines.runners import run_ifca_full
from fedprotrack.experiment.baselines import run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def _compute_eta(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute symmetric clustering error η using Hungarian matching.

    Parameters
    ----------
    predicted : np.ndarray
        Shape (K, T). Predicted concept assignments.
    ground_truth : np.ndarray
        Shape (K, T). True concept assignments.

    Returns
    -------
    float
        Symmetric clustering error in [0, 1].
    """
    from scipy.optimize import linear_sum_assignment

    K, T = ground_truth.shape
    gt_ids = sorted(set(ground_truth.ravel()))
    pred_ids = sorted(set(predicted.ravel()))

    # Build confusion matrix across all (k, t) pairs
    n_gt = len(gt_ids)
    n_pred = len(pred_ids)
    cost = np.zeros((n_gt, n_pred))
    gt_map = {v: i for i, v in enumerate(gt_ids)}
    pred_map = {v: i for i, v in enumerate(pred_ids)}

    total = 0
    for k in range(K):
        for t in range(T):
            g = gt_map.get(ground_truth[k, t])
            p = pred_map.get(predicted[k, t])
            if g is not None and p is not None:
                cost[g, p] += 1
                total += 1

    # Hungarian matching (maximise agreement → minimise -cost)
    row_ind, col_ind = linear_sum_assignment(-cost)
    matched = sum(cost[r, c] for r, c in zip(row_ind, col_ind))
    return 1.0 - matched / total if total > 0 else 1.0


def run_seed(seed: int) -> dict:
    """Run one seed with all three variants."""
    # Match the main table config exactly: same data distribution as the
    # K=20, ρ=25 variant in the 5-config CIFAR-100 sweep.
    cfg = CIFAR100RecurrenceConfig(
        K=20, T=100, n_samples=400, rho=25, alpha=0.75, delta=0.85,
        n_features=128, batch_size=64, n_workers=0,
        data_root=".cifar100_cache",
        feature_cache_dir=".feature_cache",
        feature_seed=2718,
        samples_per_coarse_class=120,
        seed=seed,
    )
    prepare_cifar100_recurrence_feature_cache(cfg)
    dataset = generate_cifar100_recurrence_dataset(cfg)
    ground_truth = dataset.concept_matrix
    n_concepts = int(ground_truth.max()) + 1

    exp_cfg = ExperimentConfig(generator_config=dataset.config, federation_every=1)
    results = {}

    # 1. FPT (Scheme C) — standard
    print(f"  [seed={seed}] Running FPT...")
    fpt_config = TwoPhaseConfig(
        omega=2.0, kappa=0.7, novelty_threshold=0.25,
        loss_novelty_threshold=0.15, sticky_dampening=1.5,
        sticky_posterior_gate=0.35, merge_threshold=0.85,
        min_count=5.0, max_concepts=max(6, n_concepts + 3),
        merge_every=2, shrink_every=6,
    )
    fpt_runner = FedProTrackRunner(
        config=fpt_config, seed=seed,
        federation_every=1, detector_name="ADWIN",
        lr=0.01, n_epochs=5, soft_aggregation=True,
        blend_alpha=0.0,
    )
    fpt_result = fpt_runner.run(dataset)
    eta_c = _compute_eta(fpt_result.predicted_concept_matrix, ground_truth)
    results["FPT"] = {
        "final_accuracy": float(fpt_result.accuracy_matrix[:, -1].mean()),
        "eta": eta_c,
    }

    # 2. IFCA — standard (stale init)
    print(f"  [seed={seed}] Running IFCA (standard)...")
    ifca_result = run_ifca_full(
        dataset, federation_every=1, n_clusters=n_concepts,
        lr=0.01, n_epochs=5,
    )
    ifca_log = ifca_result.to_experiment_log(ground_truth)
    eta_a_stale = _compute_eta(ifca_log.predicted, ground_truth)
    results["IFCA-Stale"] = {
        "final_accuracy": float(ifca_result.accuracy_matrix[:, -1].mean()),
        "eta": eta_a_stale,
    }

    # 3. IFCA-Clean — concept-matched initialization via Oracle
    # Run Oracle first to get per-concept models, then use them as IFCA init
    print(f"  [seed={seed}] Running IFCA-Clean (oracle-init)...")
    ifca_clean_result = run_ifca_full(
        dataset, federation_every=1, n_clusters=n_concepts,
        lr=0.01, n_epochs=5,
        oracle_init=True,  # Use ground-truth concept models as cluster centers
    )
    ifca_clean_log = ifca_clean_result.to_experiment_log(ground_truth)
    eta_a_clean = _compute_eta(ifca_clean_log.predicted, ground_truth)
    results["IFCA-Clean"] = {
        "final_accuracy": float(ifca_clean_result.accuracy_matrix[:, -1].mean()),
        "eta": eta_a_clean,
    }

    # 4. Oracle (upper bound)
    print(f"  [seed={seed}] Running Oracle...")
    oracle_result = run_oracle_baseline(
        exp_cfg, dataset=dataset, lr=0.01, n_epochs=5, seed=seed,
    )
    results["Oracle"] = {
        "final_accuracy": float(oracle_result.accuracy_matrix[:, -1].mean()),
        "eta": 0.0,
    }

    return {"seed": seed, "methods": results}


def main() -> None:
    out_dir = Path("results_clean_scheme_a")
    out_dir.mkdir(exist_ok=True)

    seeds = [42, 43, 44]
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Clean Scheme A η — seed {seed}")
        print(f"{'='*60}")
        result = run_seed(seed)
        all_results.append(result)
        for method, data in result["methods"].items():
            print(f"  {method:15s}: acc={data['final_accuracy']:.4f}, η={data['eta']:.4f}")

    # Aggregate
    methods_agg = {}
    for method in ["FPT", "IFCA-Stale", "IFCA-Clean", "Oracle"]:
        accs = [r["methods"][method]["final_accuracy"] for r in all_results
                if method in r["methods"]]
        etas = [r["methods"][method]["eta"] for r in all_results
                if method in r["methods"]]
        methods_agg[method] = {
            "acc_mean": float(np.mean(accs)) if accs else None,
            "acc_std": float(np.std(accs)) if accs else None,
            "eta_mean": float(np.mean(etas)) if etas else None,
            "eta_std": float(np.std(etas)) if etas else None,
        }

    summary = {"per_seed": all_results, "aggregate": methods_agg, "n_seeds": len(seeds)}

    with open(out_dir / "clean_eta_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("  AGGREGATE η COMPARISON")
    print(f"{'='*60}")
    for method, data in methods_agg.items():
        if data["eta_mean"] is not None:
            print(f"  {method:15s}: acc={data['acc_mean']:.4f}±{data['acc_std']:.4f}, "
                  f"η={data['eta_mean']:.4f}±{data['eta_std']:.4f}")

    eta_clean = methods_agg.get("IFCA-Clean", {}).get("eta_mean")
    eta_c = methods_agg.get("FPT", {}).get("eta_mean")
    if eta_clean is not None and eta_c is not None:
        if eta_clean > eta_c + 0.02:
            print(f"\n  → η_A^(clean) ({eta_clean:.3f}) > η_C ({eta_c:.3f}): "
                  "Validates A5 (information indirectness)")
        else:
            print(f"\n  → η_A^(clean) ({eta_clean:.3f}) ≈ η_C ({eta_c:.3f}): "
                  "Theorem value is primarily in stale-init term")

    print(f"\n  Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
