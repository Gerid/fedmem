"""Multi-backbone CIFAR-100 comparison.

Runs the full baseline comparison across multiple frozen backbones to show
that the anisotropic theory generalises across feature extractors with
different r_eff values.

Usage:
    python run_multi_backbone_comparison.py --seeds 42 43 44 45 46
    python run_multi_backbone_comparison.py --backbone resnet18 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Key methods to compare (subset for efficiency).
_KEY_METHODS = [
    "FedAvg", "Oracle", "CFL", "IFCA",
    "Shrinkage-iso", "Shrinkage-aniso",
    "FedProx", "FLUX", "SCAFFOLD",
]


def _run_one_backbone(
    backbone: str,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    """Run key methods on one backbone / seed combo."""
    from fedprotrack.real_data.cifar100_recurrence import (
        CIFAR100RecurrenceConfig,
        generate_cifar100_recurrence_dataset,
    )
    from fedprotrack.baselines.runners import (
        MethodResult,
        run_cfl_full,
        run_ifca_full,
        run_shrinkage_full,
        run_flux_full,
        run_compressed_fedavg_full,
        _extract_dims,
    )
    from fedprotrack.experiment.baselines import (
        run_fedavg_baseline,
        run_oracle_baseline,
    )
    from fedprotrack.estimators.shrinkage import compute_effective_rank
    from fedprotrack.metrics import compute_all_metrics
    from fedprotrack.metrics.experiment_log import ExperimentLog

    cfg = CIFAR100RecurrenceConfig(
        K=args.K,
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
        backbone=backbone,
    )

    t0 = time.time()
    dataset = generate_cifar100_recurrence_dataset(cfg)
    extract_time = time.time() - t0
    print(f"  Feature extraction: {extract_time:.1f}s")

    K, T, n_features, n_classes = _extract_dims(dataset)
    C = int(dataset.concept_matrix.max()) + 1

    # Compute r_eff.
    all_X = np.concatenate([dataset.data[(k, 0)][0] for k in range(cfg.K)])
    r_eff = compute_effective_rank(all_X)
    print(f"  backbone={backbone}, d={n_features}, r_eff={r_eff:.1f}, C={C}")

    fed_every = args.federation_every
    lr = args.lr
    n_epochs = args.n_epochs

    rows: list[dict] = []

    from fedprotrack.experiment.runner import ExperimentConfig
    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=fed_every,
    )

    # Build method runners.
    methods: dict[str, callable] = {}
    methods["FedAvg"] = lambda: ("FedAvg", run_fedavg_baseline(
        exp_cfg, dataset=dataset, n_epochs=n_epochs, lr=lr, seed=seed,
    ))
    methods["Oracle"] = lambda: ("Oracle", run_oracle_baseline(
        exp_cfg, dataset=dataset, n_epochs=n_epochs, lr=lr, seed=seed,
    ))
    methods["CFL"] = lambda: ("CFL", run_cfl_full(
        dataset, federation_every=fed_every,
    ))
    methods["IFCA"] = lambda: ("IFCA", run_ifca_full(
        dataset, federation_every=fed_every, lr=lr, n_epochs=n_epochs,
    ))
    methods["Shrinkage-iso"] = lambda: ("Shrinkage-iso", run_shrinkage_full(
        dataset, federation_every=fed_every, use_anisotropic=False,
        lr=lr, n_epochs=n_epochs,
    ))
    methods["Shrinkage-aniso"] = lambda: ("Shrinkage-aniso", run_shrinkage_full(
        dataset, federation_every=fed_every, use_anisotropic=True,
        lr=lr, n_epochs=n_epochs,
    ))
    methods["FLUX"] = lambda: ("FLUX", run_flux_full(
        dataset, federation_every=fed_every,
    ))

    for method_name, runner in methods.items():
        try:
            t1 = time.time()
            name, result = runner()
            elapsed = time.time() - t1

            if isinstance(result, MethodResult):
                acc = float(result.accuracy_matrix[:, -1].mean())
                lam_iso = getattr(result, "lambda_iso", None)
                lam_aniso = getattr(result, "lambda_aniso", None)
            elif hasattr(result, "accuracy_matrix"):
                # ExperimentResult from FedAvg/Oracle baselines
                acc = float(result.accuracy_matrix[:, -1].mean())
                lam_iso = None
                lam_aniso = None
            elif isinstance(result, np.ndarray):
                acc = float(result[:, -1].mean()) if result.ndim == 2 else float(result)
                lam_iso = None
                lam_aniso = None
            else:
                acc = float(np.mean(result))
                lam_iso = None
                lam_aniso = None

            row = {
                "backbone": backbone,
                "seed": seed,
                "method": name,
                "d": n_features,
                "r_eff": round(r_eff, 2),
                "C": C,
                "final_accuracy": round(acc, 4),
                "lambda_iso": round(lam_iso, 6) if lam_iso is not None else None,
                "lambda_aniso": round(lam_aniso, 6) if lam_aniso is not None else None,
                "time_s": round(elapsed, 1),
                "status": "ok",
            }
            rows.append(row)
            print(f"    {name}: acc={acc:.4f} ({elapsed:.1f}s)")

        except Exception as e:
            rows.append({
                "backbone": backbone,
                "seed": seed,
                "method": method_name,
                "d": n_features,
                "r_eff": round(r_eff, 2),
                "C": C,
                "final_accuracy": None,
                "lambda_iso": None,
                "lambda_aniso": None,
                "time_s": None,
                "status": f"FAILED: {e}",
            })
            print(f"    {method_name}: FAILED - {e}")
            traceback.print_exc()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-backbone CIFAR-100 comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=None, help="Single seed (RunPod compat)")
    parser.add_argument(
        "--backbones", nargs="+",
        default=["resnet18", "resnet50", "vit_b_16", "mobilenet_v2"],
    )
    parser.add_argument("--backbone", default=None, help="Single backbone override")
    parser.add_argument("--results-dir", default="tmp/multi_backbone_comparison")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-epochs", type=int, default=5)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        args.seeds = [args.seed]
    backbones = [args.backbone] if args.backbone else args.backbones
    all_rows: list[dict] = []

    for backbone in backbones:
        for seed in args.seeds:
            print(f"\n=== {backbone} / seed={seed} ===")
            rows = _run_one_backbone(backbone, seed, args)
            all_rows.extend(rows)

    # Save results.
    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary by backbone.
    print("\n=== Summary (seed-averaged) ===")
    import pandas as pd
    try:
        df = pd.DataFrame(all_rows)
        df = df[df["status"] == "ok"]
        summary = df.groupby(["backbone", "method"]).agg(
            acc_mean=("final_accuracy", "mean"),
            acc_std=("final_accuracy", "std"),
            r_eff=("r_eff", "first"),
            d=("d", "first"),
        ).reset_index()
        print(summary.to_string(index=False))
    except ImportError:
        print("(install pandas for pretty summary)")
        for row in all_rows:
            print(row)


if __name__ == "__main__":
    main()
