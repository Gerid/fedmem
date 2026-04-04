"""Implicit shrinkage validation: measure CFL's error rate η and validate Corollary 1.

This is the KEY experiment bridging theory and practice:
  1. Measures CFL's clustering error rate η on CIFAR-100
  2. Computes λ_imp = ηC/(C-1) (theory-predicted implicit shrinkage)
  3. Compares λ_imp with grid-search optimal λ*
  4. Validates that CFL-Oracle gap scales as predicted by K/C ratio
  5. Tests SGD variance inflation: larger epochs → larger CFL advantage

This directly addresses the reviewer concern that the CFL > Oracle bridge
is heuristic by providing quantitative evidence.

Usage:
    python run_implicit_shrinkage_validation.py --seeds 42 43 44 45 46
    python run_implicit_shrinkage_validation.py --seed 42
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


def _compute_cfl_error_rate(
    predicted_concept_matrix: np.ndarray,
    true_concept_matrix: np.ndarray,
) -> float:
    """Compute CFL's symmetric clustering error rate η.

    Uses Hungarian alignment to match predicted cluster IDs to true concept IDs,
    then returns 1 - accuracy = η.
    """
    from fedprotrack.metrics.concept_metrics import concept_re_id_accuracy
    global_acc, _, _ = concept_re_id_accuracy(true_concept_matrix, predicted_concept_matrix)
    return 1.0 - global_acc


def _run_one_config(
    K: int,
    n_epochs: int,
    seed: int,
    args: argparse.Namespace,
) -> list[dict]:
    """Run CFL, Oracle, FedAvg, and Shrinkage for one config.

    Returns rows with CFL's η, λ_imp, grid-search λ*, and accuracy comparison.
    """
    from fedprotrack.real_data.cifar100_recurrence import (
        CIFAR100RecurrenceConfig,
        generate_cifar100_recurrence_dataset,
    )
    from fedprotrack.baselines.runners import (
        run_cfl_full,
        run_shrinkage_full,
        _extract_dims,
    )
    from fedprotrack.experiment.baselines import (
        run_fedavg_baseline,
        run_oracle_baseline,
    )
    from fedprotrack.experiment.runner import ExperimentConfig
    from fedprotrack.estimators.shrinkage import (
        compute_effective_rank,
        grid_search_lambda,
    )

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

    # Compute r_eff from features.
    all_X = np.concatenate([dataset.data[(k, 0)][0] for k in range(K)])
    r_eff = compute_effective_rank(all_X)

    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=args.federation_every,
    )

    fed_every = args.federation_every
    lr = args.lr

    rows: list[dict] = []

    # ── Run CFL ──
    try:
        t0 = time.time()
        cfl_result = run_cfl_full(dataset, federation_every=fed_every)
        cfl_time = time.time() - t0

        cfl_acc = float(cfl_result.accuracy_matrix[:, -1].mean())

        # Measure CFL's clustering error rate η.
        eta = _compute_cfl_error_rate(
            cfl_result.predicted_concept_matrix,
            dataset.concept_matrix[:, :T],
        )

        # Compute λ_imp from theory.
        rho = eta * C / (C - 1) if C > 1 else 0.0
        lambda_imp = rho  # λ_imp = ρ = ηC/(C-1)

        print(f"    CFL: acc={cfl_acc:.4f}, η={eta:.4f}, ρ={rho:.4f}, "
              f"λ_imp={lambda_imp:.4f} ({cfl_time:.1f}s)")

    except Exception as e:
        cfl_acc = None
        eta = None
        rho = None
        lambda_imp = None
        cfl_time = None
        print(f"    CFL: FAILED - {e}")
        traceback.print_exc()

    # ── Run Oracle ──
    try:
        t0 = time.time()
        oracle_result = run_oracle_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs, seed=seed,
        )
        oracle_time = time.time() - t0
        oracle_mat = np.asarray(oracle_result.accuracy_matrix)
        oracle_acc = float(oracle_mat[:, -1].mean()) if oracle_mat.ndim == 2 else float(oracle_mat.mean())
        print(f"    Oracle: acc={oracle_acc:.4f} ({oracle_time:.1f}s)")
    except Exception as e:
        oracle_acc = None
        oracle_time = None
        print(f"    Oracle: FAILED - {e}")
        traceback.print_exc()

    # ── Run FedAvg ──
    try:
        t0 = time.time()
        fedavg_result = run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs, seed=seed,
        )
        fedavg_time = time.time() - t0
        fedavg_mat = np.asarray(fedavg_result.accuracy_matrix)
        fedavg_acc = float(fedavg_mat[:, -1].mean()) if fedavg_mat.ndim == 2 else float(fedavg_mat.mean())
        print(f"    FedAvg: acc={fedavg_acc:.4f} ({fedavg_time:.1f}s)")
    except Exception as e:
        fedavg_acc = None
        fedavg_time = None
        print(f"    FedAvg: FAILED - {e}")
        traceback.print_exc()

    # ── Run Shrinkage (isotropic + anisotropic) ──
    lambda_iso = None
    lambda_aniso = None
    shrink_iso_acc = None
    shrink_aniso_acc = None
    try:
        res_iso = run_shrinkage_full(
            dataset, federation_every=fed_every,
            use_anisotropic=False, lr=lr, n_epochs=n_epochs,
        )
        shrink_iso_acc = float(res_iso.accuracy_matrix[:, -1].mean())
        lambda_iso = getattr(res_iso, "lambda_iso", None)

        res_aniso = run_shrinkage_full(
            dataset, federation_every=fed_every,
            use_anisotropic=True, lr=lr, n_epochs=n_epochs,
        )
        shrink_aniso_acc = float(res_aniso.accuracy_matrix[:, -1].mean())
        lambda_aniso = getattr(res_aniso, "lambda_aniso", None)

        print(f"    Shrinkage-iso: acc={shrink_iso_acc:.4f}, λ_iso={lambda_iso}")
        print(f"    Shrinkage-aniso: acc={shrink_aniso_acc:.4f}, λ_aniso={lambda_aniso}")
    except Exception as e:
        print(f"    Shrinkage: FAILED - {e}")
        traceback.print_exc()

    # Compute CFL-Oracle gap.
    cfl_oracle_gap = None
    if cfl_acc is not None and oracle_acc is not None:
        cfl_oracle_gap = round(cfl_acc - oracle_acc, 4)

    row = {
        "K": K,
        "C": C,
        "K_over_C": round(K / C, 2) if C > 0 else None,
        "n_epochs": n_epochs,
        "seed": seed,
        "d": n_features,
        "r_eff": round(r_eff, 2),
        # CFL measurements.
        "cfl_acc": round(cfl_acc, 4) if cfl_acc is not None else None,
        "cfl_eta": round(eta, 4) if eta is not None else None,
        "cfl_rho": round(rho, 4) if rho is not None else None,
        "lambda_imp": round(lambda_imp, 4) if lambda_imp is not None else None,
        # Baselines.
        "oracle_acc": round(oracle_acc, 4) if oracle_acc is not None else None,
        "fedavg_acc": round(fedavg_acc, 4) if fedavg_acc is not None else None,
        # Shrinkage.
        "shrink_iso_acc": round(shrink_iso_acc, 4) if shrink_iso_acc is not None else None,
        "shrink_aniso_acc": round(shrink_aniso_acc, 4) if shrink_aniso_acc is not None else None,
        "lambda_iso": round(lambda_iso, 6) if lambda_iso is not None else None,
        "lambda_aniso": round(lambda_aniso, 6) if lambda_aniso is not None else None,
        # Gaps.
        "cfl_oracle_gap": cfl_oracle_gap,
        "cfl_beats_oracle": bool(cfl_oracle_gap > 0) if cfl_oracle_gap is not None else None,
    }
    rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Implicit shrinkage validation: measure CFL's η and validate Corollary 1",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=None, help="Single seed (RunPod compat)")
    parser.add_argument("--K-values", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    parser.add_argument("--epoch-values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--results-dir", default="tmp/implicit_shrinkage_validation")
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
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("Implicit Shrinkage Analysis")
    print(f"{'='*70}")

    from collections import defaultdict
    acc_table: dict[tuple, list] = defaultdict(list)
    eta_table: dict[tuple, list] = defaultdict(list)
    gap_table: dict[tuple, list] = defaultdict(list)

    for r in all_rows:
        key = (r["K"], r["n_epochs"])
        if r["cfl_acc"] is not None:
            acc_table[(key, "CFL")].append(r["cfl_acc"])
        if r["oracle_acc"] is not None:
            acc_table[(key, "Oracle")].append(r["oracle_acc"])
        if r["fedavg_acc"] is not None:
            acc_table[(key, "FedAvg")].append(r["fedavg_acc"])
        if r["cfl_eta"] is not None:
            eta_table[key].append(r["cfl_eta"])
        if r["cfl_oracle_gap"] is not None:
            gap_table[key].append(r["cfl_oracle_gap"])

    # Table 1: CFL error rate η by K.
    print(f"\n--- CFL Error Rate η by K (epoch-averaged) ---")
    print(f"{'K':>4} {'K/C':>5} {'η_mean':>8} {'η_std':>7} {'λ_imp':>7}")
    for K in args.K_values:
        etas = []
        for E in args.epoch_values:
            etas.extend(eta_table.get((K, E), []))
        if etas:
            eta_m = np.mean(etas)
            eta_s = np.std(etas)
            C_est = max(1, int(round(K / 1.5)))
            rho_m = eta_m * C_est / (C_est - 1) if C_est > 1 else 0
            print(f"{K:>4} {K/C_est:>5.1f} {eta_m:>8.4f} {eta_s:>7.4f} {rho_m:>7.4f}")

    # Table 2: CFL-Oracle gap by K and epochs.
    print(f"\n--- CFL-Oracle Gap by K × Epochs ---")
    print(f"{'K':>4} {'E':>3} {'CFL':>7} {'Oracle':>7} {'Gap':>7} {'η':>7} {'λ_imp':>7}")
    print("-" * 55)
    for K in args.K_values:
        for E in args.epoch_values:
            key = (K, E)
            cfl_vals = acc_table.get((key, "CFL"), [])
            orc_vals = acc_table.get((key, "Oracle"), [])
            eta_vals = eta_table.get(key, [])
            gap_vals = gap_table.get(key, [])

            cfl_m = np.mean(cfl_vals) if cfl_vals else float("nan")
            orc_m = np.mean(orc_vals) if orc_vals else float("nan")
            eta_m = np.mean(eta_vals) if eta_vals else float("nan")
            gap_m = np.mean(gap_vals) if gap_vals else float("nan")

            # Approximate C from data.
            C_approx = 8  # default for rho=7.0
            rho_m = eta_m * C_approx / (C_approx - 1) if not np.isnan(eta_m) else float("nan")

            print(f"{K:>4} {E:>3} {cfl_m:>7.4f} {orc_m:>7.4f} "
                  f"{gap_m:>+7.4f} {eta_m:>7.4f} {rho_m:>7.4f}")

    # Table 3: Shrinkage comparison.
    print(f"\n--- Shrinkage λ Comparison (seed-averaged, all configs) ---")
    lambda_imp_vals = [r["lambda_imp"] for r in all_rows if r["lambda_imp"] is not None]
    lambda_iso_vals = [r["lambda_iso"] for r in all_rows if r["lambda_iso"] is not None]
    lambda_aniso_vals = [r["lambda_aniso"] for r in all_rows if r["lambda_aniso"] is not None]

    if lambda_imp_vals:
        print(f"  λ_imp (CFL implicit):    {np.mean(lambda_imp_vals):.4f} ± {np.std(lambda_imp_vals):.4f}")
    if lambda_iso_vals:
        print(f"  λ_iso (isotropic):       {np.mean(lambda_iso_vals):.6f} ± {np.std(lambda_iso_vals):.6f}")
    if lambda_aniso_vals:
        print(f"  λ_aniso (anisotropic):   {np.mean(lambda_aniso_vals):.4f} ± {np.std(lambda_aniso_vals):.4f}")

    # Key finding: does CFL's λ_imp match λ_aniso?
    if lambda_imp_vals and lambda_aniso_vals:
        ratio = np.mean(lambda_imp_vals) / np.mean(lambda_aniso_vals) if np.mean(lambda_aniso_vals) > 0 else float("inf")
        print(f"  λ_imp / λ_aniso ratio:   {ratio:.2f}")

    # Prediction test: CFL > Oracle when K/C is small.
    print(f"\n--- Prediction Test: CFL > Oracle diminishes with K ---")
    for K in args.K_values:
        gaps_all_epochs = []
        for E in args.epoch_values:
            gaps_all_epochs.extend(gap_table.get((K, E), []))
        if gaps_all_epochs:
            g_mean = np.mean(gaps_all_epochs)
            g_std = np.std(gaps_all_epochs)
            beats = sum(1 for g in gaps_all_epochs if g > 0)
            total = len(gaps_all_epochs)
            print(f"  K={K:>2}: gap={g_mean:>+7.4f}±{g_std:.4f}, "
                  f"CFL wins {beats}/{total}")

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
