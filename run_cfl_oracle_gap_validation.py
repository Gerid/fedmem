"""CFL > Oracle gap validation: K-scaling × epoch sweep on real data.

Tests two predictions from the implicit-shrinkage theory:
  (1) CFL-Oracle gap shrinks monotonically as K/C grows.
  (2) Increasing local epochs E increases CFL's advantage (oracle overfits
      more with limited per-concept data).

Usage:
    python run_cfl_oracle_gap_validation.py --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _run_one_config(
    K: int,
    n_epochs: int,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    """Run key methods for one (K, n_epochs, seed) configuration."""
    from fedprotrack.real_data.cifar100_recurrence import (
        CIFAR100RecurrenceConfig,
        generate_cifar100_recurrence_dataset,
    )
    from fedprotrack.baselines.runners import (
        run_cfl_full,
        run_ifca_full,
        run_shrinkage_full,
        run_flux_full,
        _extract_dims,
    )
    from fedprotrack.experiment.baselines import (
        run_fedavg_baseline,
        run_oracle_baseline,
    )
    from fedprotrack.experiment.runner import ExperimentConfig
    from fedprotrack.estimators.shrinkage import compute_effective_rank

    cfg = CIFAR100RecurrenceConfig(
        K=K,
        T=args.T,
        n_samples=args.n_samples,
        rho=args.rho,
        alpha=0.75,
        delta=0.85,
        n_features=args.n_features,
        samples_per_coarse_class=120,
        batch_size=256,
        n_workers=args.n_workers,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
        seed=seed,
        label_split="none",
        backbone=args.backbone,
    )

    dataset = generate_cifar100_recurrence_dataset(cfg)
    C = int(dataset.concept_matrix.max()) + 1
    _K, T, n_features, n_classes = _extract_dims(dataset)

    # Compute r_eff.
    all_X = np.concatenate([dataset.data[(k, 0)][0] for k in range(K)])
    r_eff = compute_effective_rank(all_X)

    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=args.federation_every,
    )

    fed_every = args.federation_every
    lr = args.lr

    methods = {
        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs, seed=seed,
        ),
        "Oracle": lambda: run_oracle_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs, seed=seed,
        ),
        "CFL": lambda: run_cfl_full(dataset, federation_every=fed_every),
        "IFCA": lambda: run_ifca_full(
            dataset, federation_every=fed_every, lr=lr, n_epochs=n_epochs,
        ),
        "Shrinkage-aniso": lambda: run_shrinkage_full(
            dataset, federation_every=fed_every, use_anisotropic=True,
            lr=lr, n_epochs=n_epochs,
        ),
        "FLUX": lambda: run_flux_full(dataset, federation_every=fed_every),
    }

    rows: list[dict] = []
    for method_name, runner in methods.items():
        try:
            t0 = time.time()
            result = runner()
            elapsed = time.time() - t0

            if hasattr(result, "accuracy_matrix"):
                mat = np.asarray(result.accuracy_matrix)
                acc = float(mat[:, -1].mean()) if mat.ndim == 2 else float(mat.mean())
            elif isinstance(result, np.ndarray):
                acc = float(result[:, -1].mean()) if result.ndim == 2 else float(result.mean())
            else:
                acc = float(np.mean(result))

            rows.append({
                "K": K,
                "C": C,
                "K_over_C": round(K / C, 2),
                "n_epochs": n_epochs,
                "seed": seed,
                "method": method_name,
                "final_accuracy": round(acc, 4),
                "r_eff": round(r_eff, 2),
                "time_s": round(elapsed, 1),
                "status": "ok",
            })
            print(f"    {method_name}: acc={acc:.4f} ({elapsed:.1f}s)")

        except Exception as e:
            rows.append({
                "K": K, "C": C, "K_over_C": round(K / C, 2),
                "n_epochs": n_epochs, "seed": seed,
                "method": method_name,
                "final_accuracy": None,
                "r_eff": round(r_eff, 2),
                "time_s": None,
                "status": f"FAILED: {e}",
            })
            print(f"    {method_name}: FAILED - {e}")
            traceback.print_exc()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CFL-Oracle gap validation: K-scaling × epoch sweep",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=None, help="Single seed (RunPod compat)")
    parser.add_argument("--K-values", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument("--epoch-values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--results-dir", default="tmp/cfl_oracle_gap_validation")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--backbone", default="resnet18")
    args = parser.parse_args()

    if args.seed is not None:
        args.seeds = [args.seed]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    total_configs = len(args.K_values) * len(args.epoch_values) * len(args.seeds)
    done = 0

    for K in args.K_values:
        for n_epochs in args.epoch_values:
            for seed in args.seeds:
                done += 1
                print(f"\n[{done}/{total_configs}] K={K}, epochs={n_epochs}, seed={seed}")
                rows = _run_one_config(K, n_epochs, seed, args)
                all_rows.extend(rows)

    # Save.
    with open(results_dir / "results.json", "w") as f:
        json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)

    csv_path = results_dir / "results.csv"
    fieldnames = ["K", "C", "K_over_C", "n_epochs", "seed", "method",
                   "final_accuracy", "r_eff", "time_s", "status"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    # Analysis: CFL-Oracle gap by K and epoch.
    print(f"\n{'='*70}")
    print("CFL-Oracle Gap Analysis (seed-averaged)")
    print(f"{'='*70}")

    ok_rows = [r for r in all_rows if r["status"] == "ok"]
    from collections import defaultdict
    acc_table: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    for r in ok_rows:
        key = (r["K"], r["n_epochs"], r["method"])
        if r["final_accuracy"] is not None:
            acc_table[key].append(r["final_accuracy"])

    print(f"\n{'K':>4} {'E':>3} {'K/C':>5} {'CFL':>7} {'Oracle':>7} {'Gap':>7} {'FedAvg':>7} {'Shrink':>7}")
    print("-" * 60)
    for K in args.K_values:
        for E in args.epoch_values:
            cfl = acc_table.get((K, E, "CFL"), [])
            oracle = acc_table.get((K, E, "Oracle"), [])
            fedavg = acc_table.get((K, E, "FedAvg"), [])
            shrink = acc_table.get((K, E, "Shrinkage-aniso"), [])

            cfl_m = np.mean(cfl) if cfl else float("nan")
            orc_m = np.mean(oracle) if oracle else float("nan")
            fav_m = np.mean(fedavg) if fedavg else float("nan")
            shr_m = np.mean(shrink) if shrink else float("nan")
            gap = cfl_m - orc_m

            C_est = max(1, int(round(K / 1.5)))  # approximate
            kc = K / C_est if C_est else K

            print(f"{K:>4} {E:>3} {kc:>5.1f} {cfl_m:>7.4f} {orc_m:>7.4f} "
                  f"{gap:>+7.4f} {fav_m:>7.4f} {shr_m:>7.4f}")

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
