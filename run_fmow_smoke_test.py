"""FMOW Smoke Test — Step 2: minimal E2E validation.

K=3 clients, T=5 steps, 1 seed, synthetic proxy (no real FMOW download).
Runs FedProTrack + IFCA + FedAvg + FedProto, prints comparison table.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from fedprotrack.real_data.fmow import FMOWConfig, generate_fmow_dataset
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.baselines.runners import (
    run_ifca_full,
    run_fedproto_full,
)
from fedprotrack.experiment.baselines import run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog


def main() -> None:
    print("=" * 60)
    print("FMOW Smoke Test (synthetic proxy)")
    print("=" * 60)

    # --- Generate dataset ---
    t0 = time.time()
    cfg = FMOWConfig(
        K=3,
        T=5,
        n_samples=100,
        n_concepts=3,
        n_features=16,
        n_classes=5,
        rho=5.0,
        alpha=0.5,
        delta=0.5,
        seed=42,
        data_root=".fmow_smoke_cache",
        feature_cache_dir=".fmow_smoke_features",
    )
    print(f"\nGenerating FMOW dataset: K={cfg.K}, T={cfg.T}, "
          f"n_samples={cfg.n_samples}, n_concepts={cfg.n_concepts}, "
          f"n_features={cfg.n_features}, n_classes={cfg.n_classes}")
    dataset = generate_fmow_dataset(cfg)
    gt = dataset.concept_matrix
    t_gen = time.time() - t0
    print(f"  Dataset generated in {t_gen:.1f}s")
    print(f"  Concept matrix:\n{gt}")
    print(f"  Data shape sample: {dataset.data[(0, 0)][0].shape}")
    print(f"  Unique labels: {np.unique(dataset.data[(0, 0)][1])}")

    results: dict[str, dict] = {}

    # --- FedProTrack ---
    print("\n--- Running FedProTrack ---")
    t0 = time.time()
    fpt_config = TwoPhaseConfig()
    fpt_runner = FedProTrackRunner(
        config=fpt_config,
        seed=42,
        federation_every=1,
        detector_name="ADWIN",
    )
    fpt_result = fpt_runner.run(dataset)
    fpt_log = fpt_result.to_experiment_log()
    fpt_metrics = compute_all_metrics(fpt_log, identity_capable=True)
    t_fpt = time.time() - t0
    results["FedProTrack"] = {
        "re_id": fpt_metrics.concept_re_id_accuracy,
        "final_acc": fpt_metrics.final_accuracy,
        "auc": fpt_metrics.accuracy_auc,
        "bytes": fpt_result.total_bytes,
        "time": t_fpt,
    }
    print(f"  Done in {t_fpt:.1f}s | "
          f"re-ID={fpt_metrics.concept_re_id_accuracy:.3f} | "
          f"final_acc={fpt_metrics.final_accuracy:.3f}")

    # --- IFCA ---
    print("\n--- Running IFCA ---")
    t0 = time.time()
    ifca_result = run_ifca_full(dataset, n_clusters=cfg.n_concepts)
    ifca_log = ifca_result.to_experiment_log(gt)
    ifca_metrics = compute_all_metrics(ifca_log, identity_capable=True)
    t_ifca = time.time() - t0
    results["IFCA"] = {
        "re_id": ifca_metrics.concept_re_id_accuracy,
        "final_acc": ifca_metrics.final_accuracy,
        "auc": ifca_metrics.accuracy_auc,
        "bytes": ifca_result.total_bytes,
        "time": t_ifca,
    }
    print(f"  Done in {t_ifca:.1f}s | "
          f"re-ID={ifca_metrics.concept_re_id_accuracy:.3f} | "
          f"final_acc={ifca_metrics.final_accuracy:.3f}")

    # --- FedAvg ---
    print("\n--- Running FedAvg ---")
    t0 = time.time()
    exp_cfg = ExperimentConfig(generator_config=dataset.config)
    fa_result = run_fedavg_baseline(exp_cfg, dataset=dataset)
    fa_log = ExperimentLog(
        ground_truth=gt,
        predicted=np.zeros_like(gt),
        accuracy_curve=fa_result.accuracy_matrix,
        total_bytes=fa_result.total_bytes if hasattr(fa_result, "total_bytes") else 1.0,
        method_name="FedAvg",
    )
    fa_metrics = compute_all_metrics(fa_log, identity_capable=False)
    t_fa = time.time() - t0
    results["FedAvg"] = {
        "re_id": None,
        "final_acc": fa_metrics.final_accuracy,
        "auc": fa_metrics.accuracy_auc,
        "bytes": fa_log.total_bytes,
        "time": t_fa,
    }
    print(f"  Done in {t_fa:.1f}s | "
          f"final_acc={fa_metrics.final_accuracy:.3f}")

    # --- FedProto ---
    print("\n--- Running FedProto ---")
    t0 = time.time()
    fp_result = run_fedproto_full(dataset)
    fp_log = fp_result.to_experiment_log(gt)
    fp_metrics = compute_all_metrics(fp_log, identity_capable=False)
    t_fp = time.time() - t0
    results["FedProto"] = {
        "re_id": None,
        "final_acc": fp_metrics.final_accuracy,
        "auc": fp_metrics.accuracy_auc,
        "bytes": fp_result.total_bytes,
        "time": t_fp,
    }
    print(f"  Done in {t_fp:.1f}s | "
          f"final_acc={fp_metrics.final_accuracy:.3f}")

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print("FMOW SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"{'Method':<15} {'Re-ID':>8} {'Final Acc':>10} {'AUC':>8} {'Bytes':>10} {'Time(s)':>8}")
    print("-" * 60)
    for name, r in results.items():
        re_id_str = f"{r['re_id']:.3f}" if r["re_id"] is not None else "  N/A"
        bytes_str = f"{r['bytes']:.0f}" if r["bytes"] is not None else "N/A"
        print(f"{name:<15} {re_id_str:>8} {r['final_acc']:>10.3f} "
              f"{r['auc']:>8.3f} {bytes_str:>10} {r['time']:>8.1f}")

    print("\nSmoke test complete!")


if __name__ == "__main__":
    main()
