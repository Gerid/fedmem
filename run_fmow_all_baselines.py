"""FMOW full-baseline comparison for paper-quality validation.

Runs all 20+ baselines on the FMOW temporal-drift dataset (satellite imagery)
to provide a second independent validation platform beyond CIFAR-100.

Usage:
    python run_fmow_all_baselines.py --seeds 42 43 44 45 46
    python run_fmow_all_baselines.py --seed 42  # single seed for debug
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _build_fmow_methods(
    dataset,
    *,
    federation_every: int,
    lr: float,
    n_epochs: int,
):
    """Build dict of method-name -> callable returning result."""
    from fedprotrack.baselines.runners import (
        run_apfl_full,
        run_atp_full,
        run_cfl_full,
        run_compressed_fedavg_full,
        run_feddrift_full,
        run_fedccfa_full,
        run_fedem_full,
        run_fedproto_full,
        run_fedrc_full,
        run_fesem_full,
        run_flash_full,
        run_flux_full,
        run_flux_prior_full,
        run_ifca_full,
        run_pfedme_full,
        run_shrinkage_full,
        run_tracked_summary_full,
    )
    from fedprotrack.experiment.baselines import (
        run_fedavg_baseline,
        run_oracle_baseline,
        run_local_only,
    )
    from fedprotrack.experiment.runner import ExperimentConfig
    from fedprotrack.posterior import FedProTrackRunner, TwoPhaseConfig

    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=federation_every,
    )

    C_hat = int(dataset.concept_matrix.max()) + 1

    return {
        "FedProTrack-calibrated": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0, kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, C_hat + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=federation_every,
            detector_name="ADWIN",
            seed=int(dataset.config.seed),
            lr=lr,
            n_epochs=n_epochs,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        ).run(dataset),
        "LocalOnly": lambda: run_local_only(exp_cfg, dataset=dataset),
        "FedAvg": lambda: run_fedavg_baseline(exp_cfg, dataset=dataset),
        "FedAvg-FPTTrain": lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs,
            seed=int(dataset.config.seed),
        ),
        "Oracle": lambda: run_oracle_baseline(exp_cfg, dataset=dataset),
        "FedProto": lambda: run_fedproto_full(dataset, federation_every=federation_every),
        "pFedMe": lambda: run_pfedme_full(dataset, federation_every=federation_every),
        "APFL": lambda: run_apfl_full(dataset, federation_every=federation_every),
        "FedEM": lambda: run_fedem_full(dataset, federation_every=federation_every),
        "FedCCFA": lambda: run_fedccfa_full(dataset, federation_every=federation_every),
        "CFL": lambda: run_cfl_full(dataset, federation_every=federation_every),
        "FeSEM": lambda: run_fesem_full(dataset, federation_every=federation_every),
        "FedRC": lambda: run_fedrc_full(dataset, federation_every=federation_every),
        "TrackedSummary": lambda: run_tracked_summary_full(dataset, federation_every=federation_every),
        "Flash": lambda: run_flash_full(dataset, federation_every=federation_every),
        "FedDrift": lambda: run_feddrift_full(dataset, federation_every=federation_every),
        "IFCA": lambda: run_ifca_full(dataset, federation_every=federation_every, n_clusters=4),
        "ATP": lambda: run_atp_full(dataset, federation_every=federation_every),
        "FLUX": lambda: run_flux_full(dataset, federation_every=federation_every),
        "FLUX-prior": lambda: run_flux_prior_full(dataset, federation_every=federation_every),
        "CompressedFedAvg": lambda: run_compressed_fedavg_full(dataset, federation_every=federation_every),
        "Shrinkage-iso": lambda: run_shrinkage_full(
            dataset, federation_every=federation_every, use_anisotropic=False,
            lr=lr, n_epochs=n_epochs,
        ),
        "Shrinkage-aniso": lambda: run_shrinkage_full(
            dataset, federation_every=federation_every, use_anisotropic=True,
            lr=lr, n_epochs=n_epochs,
        ),
    }


def _extract_accuracy(result) -> float | None:
    """Extract final-round mean accuracy from various result types."""
    if hasattr(result, "accuracy_matrix"):
        mat = np.asarray(result.accuracy_matrix)
        if mat.ndim == 2:
            return float(mat[:, -1].mean())
        return float(mat.mean())
    if isinstance(result, np.ndarray):
        if result.ndim == 2:
            return float(result[:, -1].mean())
        return float(result.mean())
    return None


def _extract_auc(result) -> float | None:
    """Extract accuracy AUC from result."""
    if hasattr(result, "accuracy_matrix"):
        mat = np.asarray(result.accuracy_matrix)
        if mat.ndim == 2:
            return float(mat.mean())
    if isinstance(result, np.ndarray) and result.ndim == 2:
        return float(result.mean())
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="FMOW all-baselines comparison")
    parser.add_argument("--results-dir", default="tmp/fmow_all_baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=None, help="Single seed override")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--n-concepts", type=int, default=4)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--data-root", default=".fmow_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--n-workers", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else args.seeds

    from fedprotrack.real_data.fmow import FMOWConfig, generate_fmow_dataset
    from fedprotrack.estimators.shrinkage import compute_effective_rank

    all_rows: list[dict] = []
    failures: list[dict] = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"FMOW | seed={seed}")
        print(f"{'='*60}")

        cfg = FMOWConfig(
            K=args.K,
            T=args.T,
            n_samples=args.n_samples,
            rho=args.rho,
            alpha=args.alpha,
            delta=args.delta,
            n_concepts=args.n_concepts,
            n_features=args.n_features,
            n_classes=args.n_classes,
            batch_size=256,
            n_workers=args.n_workers,
            data_root=args.data_root,
            feature_cache_dir=args.feature_cache_dir,
            feature_seed=3141,
            seed=seed,
        )

        t0 = time.time()
        dataset = generate_fmow_dataset(cfg)
        extract_time = time.time() - t0
        print(f"  Dataset generated in {extract_time:.1f}s")

        C = int(dataset.concept_matrix.max()) + 1
        all_X = np.concatenate([dataset.data[(k, 0)][0] for k in range(cfg.K)])
        r_eff = compute_effective_rank(all_X)
        print(f"  C={C}, d={args.n_features}, r_eff={r_eff:.1f}")

        methods = _build_fmow_methods(
            dataset,
            federation_every=args.federation_every,
            lr=args.lr,
            n_epochs=args.n_epochs,
        )

        for method_name, runner in methods.items():
            try:
                t1 = time.time()
                result = runner()
                elapsed = time.time() - t1

                acc = _extract_accuracy(result)
                auc = _extract_auc(result)
                total_bytes = float(getattr(result, "total_bytes", 0))

                row = {
                    "seed": seed,
                    "method": method_name,
                    "final_accuracy": round(acc, 4) if acc is not None else None,
                    "accuracy_auc": round(auc, 4) if auc is not None else None,
                    "total_bytes": total_bytes,
                    "r_eff": round(r_eff, 2),
                    "time_s": round(elapsed, 1),
                    "status": "ok",
                }
                all_rows.append(row)
                print(f"  {method_name}: acc={acc:.4f if acc else 'N/A'} ({elapsed:.1f}s)")

            except Exception as e:
                all_rows.append({
                    "seed": seed,
                    "method": method_name,
                    "final_accuracy": None,
                    "accuracy_auc": None,
                    "total_bytes": None,
                    "r_eff": round(r_eff, 2),
                    "time_s": None,
                    "status": f"FAILED",
                })
                failures.append({
                    "seed": seed,
                    "method": method_name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                print(f"  {method_name}: FAILED - {e}")

    # Save results.
    with open(results_dir / "results.json", "w") as f:
        json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)
    if failures:
        with open(results_dir / "failures.json", "w") as f:
            json.dump(failures, f, indent=2)

    # CSV summary.
    csv_path = results_dir / "results.csv"
    fieldnames = ["seed", "method", "final_accuracy", "accuracy_auc",
                   "total_bytes", "r_eff", "time_s", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # Print summary.
    print(f"\n{'='*60}")
    print("FMOW Results Summary (seed-averaged)")
    print(f"{'='*60}")
    try:
        import pandas as pd
        df = pd.DataFrame(all_rows)
        df = df[df["status"] == "ok"]
        summary = df.groupby("method").agg(
            acc_mean=("final_accuracy", "mean"),
            acc_std=("final_accuracy", "std"),
        ).sort_values("acc_mean", ascending=False)
        print(summary.to_string())
    except ImportError:
        # Fallback without pandas.
        from collections import defaultdict
        accs: dict[str, list[float]] = defaultdict(list)
        for row in all_rows:
            if row["status"] == "ok" and row["final_accuracy"] is not None:
                accs[row["method"]].append(row["final_accuracy"])
        sorted_methods = sorted(accs.keys(), key=lambda m: -np.mean(accs[m]))
        for m in sorted_methods:
            vals = accs[m]
            print(f"  {m:<30s} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
