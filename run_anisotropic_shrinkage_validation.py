"""Validate anisotropic shrinkage estimator on CIFAR-100 frozen features.

Sweeps C ∈ {4, 6, 8} and backbones, comparing:
  - λ̂_iso  (isotropic, uses raw d)
  - λ̂_aniso (anisotropic, uses r_eff)
  - λ* (grid-search oracle)
  - accuracy under each λ

Usage:
    python run_anisotropic_shrinkage_validation.py --seeds 42 43 44 45 46
    python run_anisotropic_shrinkage_validation.py --backbone resnet50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Anisotropic shrinkage validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=None, help="Single seed (RunPod compat)")
    parser.add_argument("--results-dir", default="tmp/aniso_shrinkage_validation")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument(
        "--backbone", default="resnet18",
        choices=["resnet18", "resnet50", "vit_b_16", "mobilenet_v2"],
    )
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-epochs", type=int, default=5)
    args = parser.parse_args()

    if args.seed is not None:
        args.seeds = [args.seed]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from fedprotrack.real_data.cifar100_recurrence import (
        CIFAR100RecurrenceConfig,
        generate_cifar100_recurrence_dataset,
    )
    from fedprotrack.baselines.runners import (
        run_shrinkage_full,
        _extract_dims,
    )
    from fedprotrack.estimators.shrinkage import compute_effective_rank

    all_rows: list[dict] = []
    C_values = [4, 6, 8]

    for C in C_values:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n=== C={C}, seed={seed}, backbone={args.backbone} ===")

            cfg = CIFAR100RecurrenceConfig(
                K=args.K,
                T=args.T,
                n_samples=400,
                rho=3.0,
                alpha=0.75,
                delta=0.85,
                n_features=args.n_features,
                samples_per_coarse_class=120,
                batch_size=256,
                n_workers=args.n_workers,
                data_root=args.data_root,
                feature_cache_dir=args.feature_cache_dir,
                feature_seed=2718,
                seed=seed,
                label_split="none",
                backbone=args.backbone,
            )
            # Override n_concepts via rho
            # n_concepts is determined by rho in the concept matrix generator.
            # For C=4, default rho=3.0 gives 4 concepts.
            # For C=6/8, we need higher rho.
            if C == 6:
                cfg = CIFAR100RecurrenceConfig(**{**cfg.__dict__, "rho": 5.0})
            elif C == 8:
                cfg = CIFAR100RecurrenceConfig(**{**cfg.__dict__, "rho": 7.0})

            dataset = generate_cifar100_recurrence_dataset(cfg)
            actual_C = int(dataset.concept_matrix.max()) + 1
            print(f"  actual C={actual_C}")

            # Compute r_eff from the full feature pool.
            all_X = np.concatenate(
                [dataset.data[(k, 0)][0] for k in range(cfg.K)]
            )
            r_eff = compute_effective_rank(all_X)
            print(f"  r_eff={r_eff:.1f}, d={args.n_features}")

            # Run isotropic shrinkage.
            res_iso = run_shrinkage_full(
                dataset,
                federation_every=args.federation_every,
                use_anisotropic=False,
                lr=args.lr,
                n_epochs=args.n_epochs,
            )

            # Run anisotropic shrinkage.
            res_aniso = run_shrinkage_full(
                dataset,
                federation_every=args.federation_every,
                use_anisotropic=True,
                lr=args.lr,
                n_epochs=args.n_epochs,
            )

            # Run Oracle and FedAvg for comparison.
            from fedprotrack.experiment.baselines import (
                run_oracle_baseline,
                run_fedavg_baseline,
            )
            from fedprotrack.experiment.runner import ExperimentConfig
            exp_cfg = ExperimentConfig(
                generator_config=cfg.to_generator_config(),
                federation_every=args.federation_every,
            )
            oracle_result = run_oracle_baseline(
                exp_cfg, dataset=dataset,
                n_epochs=args.n_epochs, lr=args.lr, seed=seed,
            )
            oracle_acc = oracle_result.accuracy_matrix

            fedavg_result = run_fedavg_baseline(
                exp_cfg, dataset=dataset,
                n_epochs=args.n_epochs, lr=args.lr, seed=seed,
            )
            fedavg_acc = fedavg_result.accuracy_matrix

            elapsed = time.time() - t0

            acc_iso = float(res_iso.accuracy_matrix[:, -1].mean())
            acc_aniso = float(res_aniso.accuracy_matrix[:, -1].mean())
            acc_oracle = float(oracle_acc[:, -1].mean()) if oracle_acc.ndim == 2 else float(oracle_acc)
            acc_fedavg = float(fedavg_acc[:, -1].mean()) if fedavg_acc.ndim == 2 else float(fedavg_acc)

            row = {
                "backbone": args.backbone,
                "C": actual_C,
                "seed": seed,
                "d": args.n_features,
                "r_eff": round(r_eff, 2),
                "lambda_iso": round(getattr(res_iso, "lambda_iso", 0.0), 6),
                "lambda_aniso": round(getattr(res_aniso, "lambda_aniso", 0.0), 6),
                "acc_shrinkage_iso": round(acc_iso, 4),
                "acc_shrinkage_aniso": round(acc_aniso, 4),
                "acc_oracle": round(acc_oracle, 4),
                "acc_fedavg": round(acc_fedavg, 4),
                "time_s": round(elapsed, 1),
            }
            all_rows.append(row)
            print(f"  λ_iso={row['lambda_iso']}, λ_aniso={row['lambda_aniso']}")
            print(f"  acc: iso={acc_iso:.4f}, aniso={acc_aniso:.4f}, "
                  f"oracle={acc_oracle:.4f}, fedavg={acc_fedavg:.4f}")
            print(f"  time: {elapsed:.1f}s")

    # Save results.
    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table.
    print("\n=== Summary ===")
    print(f"{'Backbone':<12} {'C':>3} {'d':>4} {'r_eff':>6} "
          f"{'λ_iso':>8} {'λ_aniso':>8} "
          f"{'Acc_iso':>8} {'Acc_aniso':>9} {'Oracle':>7} {'FedAvg':>7}")
    for row in all_rows:
        print(f"{row['backbone']:<12} {row['C']:>3} {row['d']:>4} {row['r_eff']:>6.1f} "
              f"{row['lambda_iso']:>8.4f} {row['lambda_aniso']:>8.4f} "
              f"{row['acc_shrinkage_iso']:>8.4f} {row['acc_shrinkage_aniso']:>9.4f} "
              f"{row['acc_oracle']:>7.4f} {row['acc_fedavg']:>7.4f}")


if __name__ == "__main__":
    main()
