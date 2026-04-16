"""Synthetic extension sweep: validate Theorems 4 (proportional) and 5 (anisotropic).

Completes the 660-config claim by running two grids:
  A. Proportional regime: vary d/(Kn) ratio (gamma)
  B. Anisotropic features: vary r_eff/d ratio

Each checks crossover alignment vs the extended theory predictions.

Usage:
    python run_synthetic_extension_sweep.py --seeds 42 43 44 45 46
    python run_synthetic_extension_sweep.py --grid proportional --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _run_gaussian_experiment(
    K: int,
    C: int,
    delta: float,
    d: int,
    n_per_client: int,
    sigma: float,
    generator_type: str,
    seed: int,
    r_signal: int = 5,
    signal_eigenvalue: float = 10.0,
) -> dict:
    """Run one Gaussian linear experiment and check crossover.

    Returns a dict with config, results, and alignment check.
    """
    from fedprotrack.drift_generator.data_streams import ConceptSpec

    noise_scale = 1.0 - delta
    n_concepts = C

    # Generate per-concept data.
    concept_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for j in range(C):
        spec = ConceptSpec(
            concept_id=j,
            generator_type=generator_type,
            variant=j,
            noise_scale=noise_scale,
        )
        from fedprotrack.drift_generator.data_streams import _generate_gaussian_samples
        X, y = _generate_gaussian_samples(
            spec, n_per_client * (K // C), seed + j * 1000,
            d=d, sigma=sigma, r_signal=r_signal,
            signal_eigenvalue=signal_eigenvalue,
        )
        concept_data[j] = (X, y)

    # Compute OLS estimates.
    # Global: pool all data.
    X_all = np.vstack([concept_data[j][0] for j in range(C)])
    y_all = np.concatenate([concept_data[j][1] for j in range(C)])

    # Concept-level: per-concept data.
    global_mse_list = []
    concept_mse_list = []

    for j in range(C):
        X_j, y_j = concept_data[j]
        y_j_float = y_j.astype(np.float64)

        # Global OLS.
        y_all_float = y_all.astype(np.float64)
        try:
            w_global = np.linalg.lstsq(X_all, y_all_float, rcond=None)[0]
        except np.linalg.LinAlgError:
            w_global = np.zeros(d)

        # Concept-level OLS.
        try:
            w_concept = np.linalg.lstsq(X_j, y_j_float, rcond=None)[0]
        except np.linalg.LinAlgError:
            w_concept = np.zeros(d)

        # Evaluate MSE on concept j's data.
        pred_global = X_j @ w_global
        pred_concept = X_j @ w_concept
        global_mse = float(np.mean((y_j_float - pred_global) ** 2))
        concept_mse = float(np.mean((y_j_float - pred_concept) ** 2))
        global_mse_list.append(global_mse)
        concept_mse_list.append(concept_mse)

    avg_global_mse = np.mean(global_mse_list)
    avg_concept_mse = np.mean(concept_mse_list)

    # Empirical winner.
    empirical_concept_wins = avg_concept_mse < avg_global_mse

    # Theory prediction.
    n_total = K * n_per_client
    gamma_G = d / n_total
    gamma_j = d / (n_total / C)

    # Compute B_j^2 (average concept separation).
    w_stars = []
    for j in range(C):
        w_rng = np.random.default_rng(42 + j * 7919)
        w = w_rng.normal(0, 1, size=d) * delta * 3.0
        if generator_type == "gaussian_anisotropic":
            w[r_signal:] *= 0.1
        w_stars.append(w)
    w_bar = np.mean(w_stars, axis=0)
    B2_avg = np.mean([np.sum((w - w_bar) ** 2) for w in w_stars])

    if generator_type == "gaussian_linear":
        # Base crossover: SNR > C - 1
        snr = n_total * B2_avg / (sigma ** 2 * d)
        theory_concept_wins = snr > (C - 1)
        # Proportional regime: adjust for gamma
        if gamma_G < 1 and gamma_j < 1:
            threshold = (C - 1) / ((1 - C * gamma_G) * (1 - gamma_G))
            theory_concept_wins_ext = snr > threshold
        else:
            theory_concept_wins_ext = False  # interpolation boundary
    elif generator_type == "gaussian_anisotropic":
        # Anisotropic: use r_signal * signal_eigenvalue as effective dimension proxy
        from fedprotrack.estimators.shrinkage import compute_effective_rank
        X_sample = concept_data[0][0]
        r_eff = compute_effective_rank(X_sample)
        snr = n_total * B2_avg / (sigma ** 2 * r_eff)
        theory_concept_wins = snr > (C - 1)
        theory_concept_wins_ext = theory_concept_wins
    else:
        snr = 0
        theory_concept_wins = False
        theory_concept_wins_ext = False

    aligned = empirical_concept_wins == theory_concept_wins_ext

    return {
        "K": K,
        "C": C,
        "delta": delta,
        "d": d,
        "n_per_client": n_per_client,
        "sigma": sigma,
        "generator_type": generator_type,
        "seed": seed,
        "gamma_G": round(gamma_G, 4),
        "snr": round(snr, 2),
        "B2_avg": round(B2_avg, 4),
        "global_mse": round(avg_global_mse, 6),
        "concept_mse": round(avg_concept_mse, 6),
        "empirical_concept_wins": bool(empirical_concept_wins),
        "theory_concept_wins": bool(theory_concept_wins_ext),
        "aligned": bool(aligned),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic extension sweep")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=None, help="Single seed (RunPod compat)")
    parser.add_argument("--results-dir", default="tmp/synthetic_extension_sweep")
    parser.add_argument(
        "--grid", choices=["proportional", "anisotropic", "both"],
        default="both",
    )
    args = parser.parse_args()

    if args.seed is not None:
        args.seeds = [args.seed]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # Base configs (subset of the 108 base case).
    K_values = [10, 20, 40]
    C_values = [2, 4, 8]
    delta_values = [0.3, 1.0, 3.0, 8.0]

    # Grid A: Proportional regime (vary d relative to Kn).
    if args.grid in ("proportional", "both"):
        print("=" * 60)
        print("Grid A: Proportional Regime")
        print("=" * 60)
        d_values = [20, 50, 100, 200]  # different d
        n_values = [50, 100, 200]       # different n
        count = 0
        for K in [10, 20]:
            for C in [2, 4]:
                for delta in [0.3, 1.0, 3.0]:
                    for d in d_values:
                        for n in n_values:
                            gamma_G = d / (K * n)
                            gamma_j = d * C / (K * n)
                            if gamma_G >= 1.0 or gamma_j >= 1.0:
                                continue  # skip interpolation boundary
                            for seed in args.seeds:
                                row = _run_gaussian_experiment(
                                    K=K, C=C, delta=delta, d=d,
                                    n_per_client=n, sigma=1.0,
                                    generator_type="gaussian_linear",
                                    seed=seed,
                                )
                                all_rows.append(row)
                                count += 1
        print(f"  Ran {count} proportional-regime configs")

    # Grid B: Anisotropic features.
    if args.grid in ("anisotropic", "both"):
        print("=" * 60)
        print("Grid B: Anisotropic Features")
        print("=" * 60)
        d = 100
        r_signal_values = [3, 5, 10, 20, 50]  # different r_signal
        s_values = [5.0, 10.0, 50.0]            # signal eigenvalue
        count = 0
        for K in [10, 20]:
            for C in [2, 4]:
                for delta in [0.3, 1.0, 3.0]:
                    for r_sig in r_signal_values:
                        for s_val in s_values:
                            for seed in args.seeds:
                                row = _run_gaussian_experiment(
                                    K=K, C=C, delta=delta, d=d,
                                    n_per_client=200, sigma=1.0,
                                    generator_type="gaussian_anisotropic",
                                    seed=seed,
                                    r_signal=r_sig,
                                    signal_eigenvalue=s_val,
                                )
                                all_rows.append(row)
                                count += 1
        print(f"  Ran {count} anisotropic configs")

    # Save results.
    with open(results_dir / "results.json", "w") as f:
        json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)

    csv_path = results_dir / "results.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)

    # Summary.
    total = len(all_rows)
    aligned = sum(1 for r in all_rows if r["aligned"])
    print(f"\n{'='*60}")
    print(f"Total configs: {total}")
    print(f"Aligned: {aligned}/{total} = {aligned/total*100:.1f}%")

    # Per-grid breakdown.
    for gt in ["gaussian_linear", "gaussian_anisotropic"]:
        subset = [r for r in all_rows if r["generator_type"] == gt]
        if subset:
            n_aligned = sum(1 for r in subset if r["aligned"])
            print(f"  {gt}: {n_aligned}/{len(subset)} = {n_aligned/len(subset)*100:.1f}%")

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
