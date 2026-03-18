"""FMOW Scaled Experiment — paper-quality initial run.

Grid: T ∈ {20, 40}, K=5, rho ∈ {5, 10}, alpha ∈ {0.0, 0.5, 1.0},
      delta ∈ {0.3, 0.7}, 3 seeds.
Methods: FedProTrack, IFCA, FedAvg, FedProto.
Total settings: 2 * 2 * 3 * 2 * 3 = 72 runs per method.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fedprotrack.real_data.fmow import FMOWConfig, generate_fmow_dataset
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.baselines.runners import run_ifca_full, run_fedproto_full
from fedprotrack.experiment.baselines import run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog


OUTPUT_DIR = Path("results_fmow_scaled")

# Grid parameters
T_VALUES = [20, 40]
K = 5
RHO_VALUES = [5.0, 10.0]
ALPHA_VALUES = [0.0, 0.5, 1.0]
DELTA_VALUES = [0.3, 0.7]
SEEDS = [42, 123, 777]
N_SAMPLES = 100
N_CONCEPTS = 4
N_FEATURES = 16
N_CLASSES = 5


def run_single(
    T: int, rho: float, alpha: float, delta: float, seed: int,
) -> dict[str, dict]:
    """Run all methods on one (T, rho, alpha, delta, seed) setting."""
    cfg = FMOWConfig(
        K=K, T=T, n_samples=N_SAMPLES, n_concepts=N_CONCEPTS,
        n_features=N_FEATURES, n_classes=N_CLASSES,
        rho=rho, alpha=alpha, delta=delta, seed=seed,
        data_root=".fmow_scaled_cache",
        feature_cache_dir=".fmow_scaled_features",
    )
    dataset = generate_fmow_dataset(cfg)
    gt = dataset.concept_matrix
    results: dict[str, dict] = {}

    # FedProTrack
    fpt_config = TwoPhaseConfig()
    fpt_runner = FedProTrackRunner(
        config=fpt_config, seed=seed, federation_every=1,
        detector_name="ADWIN",
    )
    fpt_res = fpt_runner.run(dataset)
    fpt_log = fpt_res.to_experiment_log()
    fpt_m = compute_all_metrics(fpt_log, identity_capable=True)
    results["FedProTrack"] = {
        "re_id": fpt_m.concept_re_id_accuracy,
        "final_acc": fpt_m.final_accuracy,
        "auc": fpt_m.accuracy_auc,
        "bytes": fpt_res.total_bytes,
    }

    # IFCA
    ifca_res = run_ifca_full(dataset, n_clusters=N_CONCEPTS)
    ifca_log = ifca_res.to_experiment_log(gt)
    ifca_m = compute_all_metrics(ifca_log, identity_capable=True)
    results["IFCA"] = {
        "re_id": ifca_m.concept_re_id_accuracy,
        "final_acc": ifca_m.final_accuracy,
        "auc": ifca_m.accuracy_auc,
        "bytes": ifca_res.total_bytes,
    }

    # FedAvg
    exp_cfg = ExperimentConfig(generator_config=dataset.config)
    fa_res = run_fedavg_baseline(exp_cfg, dataset=dataset)
    fa_log = ExperimentLog(
        ground_truth=gt, predicted=np.zeros_like(gt),
        accuracy_curve=fa_res.accuracy_matrix,
        total_bytes=fa_res.total_bytes if hasattr(fa_res, "total_bytes") else 1.0,
        method_name="FedAvg",
    )
    fa_m = compute_all_metrics(fa_log, identity_capable=False)
    results["FedAvg"] = {
        "re_id": None,
        "final_acc": fa_m.final_accuracy,
        "auc": fa_m.accuracy_auc,
        "bytes": fa_log.total_bytes,
    }

    # FedProto
    fp_res = run_fedproto_full(dataset)
    fp_log = fp_res.to_experiment_log(gt)
    fp_m = compute_all_metrics(fp_log, identity_capable=False)
    results["FedProto"] = {
        "re_id": None,
        "final_acc": fp_m.final_accuracy,
        "auc": fp_m.accuracy_auc,
        "bytes": fp_res.total_bytes,
    }

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = [
        (T, rho, alpha, delta, seed)
        for T in T_VALUES
        for rho in RHO_VALUES
        for alpha in ALPHA_VALUES
        for delta in DELTA_VALUES
        for seed in SEEDS
    ]
    total = len(grid)
    print(f"FMOW Scaled Experiment: {total} settings x 4 methods")
    print(f"Grid: T={T_VALUES}, K={K}, rho={RHO_VALUES}, "
          f"alpha={ALPHA_VALUES}, delta={DELTA_VALUES}, seeds={SEEDS}")
    print("=" * 70)

    all_results: list[dict] = []
    t_start = time.time()

    for i, (T, rho, alpha, delta, seed) in enumerate(grid):
        tag = f"T{T}_rho{rho:.0f}_a{alpha:.2f}_d{delta:.2f}_s{seed}"
        print(f"\n[{i+1}/{total}] {tag} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            results = run_single(T, rho, alpha, delta, seed)
            elapsed = time.time() - t0

            row = {
                "T": T, "rho": rho, "alpha": alpha, "delta": delta,
                "seed": seed, "time": round(elapsed, 1),
            }
            for method, metrics in results.items():
                for k, v in metrics.items():
                    row[f"{method}_{k}"] = v

            all_results.append(row)
            # Print compact summary
            fpt_reid = results["FedProTrack"]["re_id"]
            ifca_reid = results["IFCA"]["re_id"]
            fpt_acc = results["FedProTrack"]["final_acc"]
            ifca_acc = results["IFCA"]["final_acc"]
            winner = "FPT" if (fpt_reid or 0) > (ifca_reid or 0) else "IFCA"
            print(f"({elapsed:.1f}s) FPT re-ID={fpt_reid:.3f} acc={fpt_acc:.3f} | "
                  f"IFCA re-ID={ifca_reid:.3f} acc={ifca_acc:.3f} [{winner}]")

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    t_total = time.time() - t_start

    # Save raw results
    results_path = OUTPUT_DIR / "raw_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {results_path}")

    # --- Aggregate and print tables ---
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (mean ± std across seeds)")
    print("=" * 70)

    methods = ["FedProTrack", "IFCA", "FedAvg", "FedProto"]
    metric_keys = ["re_id", "final_acc", "auc"]

    # Group by (T, rho, alpha, delta)
    from collections import defaultdict
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in all_results:
        key = (row["T"], row["rho"], row["alpha"], row["delta"])
        grouped[key].append(row)

    # Overall means
    print(f"\n{'Method':<15} {'Re-ID':>12} {'Final Acc':>12} {'AUC':>12}")
    print("-" * 55)
    for method in methods:
        vals = {k: [] for k in metric_keys}
        for row in all_results:
            for k in metric_keys:
                v = row.get(f"{method}_{k}")
                if v is not None:
                    vals[k].append(v)
        parts = []
        for k in metric_keys:
            if vals[k]:
                m, s = np.mean(vals[k]), np.std(vals[k])
                parts.append(f"{m:.3f}±{s:.3f}")
            else:
                parts.append("     N/A    ")
        print(f"{method:<15} {parts[0]:>12} {parts[1]:>12} {parts[2]:>12}")

    # By T
    for T in T_VALUES:
        print(f"\n--- T = {T} ---")
        print(f"{'Method':<15} {'Re-ID':>12} {'Final Acc':>12} {'AUC':>12}")
        print("-" * 55)
        for method in methods:
            vals = {k: [] for k in metric_keys}
            for row in all_results:
                if row["T"] != T:
                    continue
                for k in metric_keys:
                    v = row.get(f"{method}_{k}")
                    if v is not None:
                        vals[k].append(v)
            parts = []
            for k in metric_keys:
                if vals[k]:
                    m, s = np.mean(vals[k]), np.std(vals[k])
                    parts.append(f"{m:.3f}±{s:.3f}")
                else:
                    parts.append("     N/A    ")
            print(f"{method:<15} {parts[0]:>12} {parts[1]:>12} {parts[2]:>12}")

    # Win rate: FedProTrack vs IFCA on re-ID
    fpt_wins = 0
    total_cmp = 0
    for row in all_results:
        fpt_r = row.get("FedProTrack_re_id")
        ifca_r = row.get("IFCA_re_id")
        if fpt_r is not None and ifca_r is not None:
            total_cmp += 1
            if fpt_r > ifca_r:
                fpt_wins += 1

    print(f"\nFedProTrack vs IFCA win rate (re-ID): "
          f"{fpt_wins}/{total_cmp} = {fpt_wins/max(total_cmp,1)*100:.1f}%")
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")


if __name__ == "__main__":
    main()
