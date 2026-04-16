from __future__ import annotations

"""Theorem 2 transient analysis: error vs rounds-since-concept-switch.

Demonstrates that after a concept switch, there is a transient period s*
where global aggregation dominates, before concept-level aggregation
takes over. This validates Theorem 2's prediction of time-varying
optimal granularity.
"""

import csv
import json
from pathlib import Path

import numpy as np

from run_granularity_crossover import (
    generate_gaussian_fl_data,
    _ols_fit,
    _mse,
)


def run_transient_experiment(
    K: int = 20,
    C: int = 4,
    d: int = 20,
    delta: float = 3.0,
    sigma: float = 1.0,
    n_per_client: int = 200,
    tau: int = 10,
    T: int = 40,
    federation_every: int = 1,
    seeds: list[int] | None = None,
) -> dict:
    """Track per-round MSE for FedAvg/Oracle/Shrinkage after concept switches."""
    if seeds is None:
        seeds = [42, 43, 44]

    all_fedavg_mse = []
    all_oracle_mse = []
    all_shrink_mse = []

    for seed in seeds:
        concept_matrix, data, w_stars = generate_gaussian_fl_data(
            K=K, T=T, C=C, d=d, delta=delta, sigma=sigma,
            n_per_client=n_per_client, stability_tau=tau, seed=seed,
        )

        # --- FedAvg ---
        global_w = np.zeros(d, dtype=np.float64)
        global_b = np.zeros(1, dtype=np.float64)
        fedavg_mse_per_round = []
        for t in range(T):
            round_mse = []
            uploads_w = []
            for k in range(K):
                X, y = data[(k, t)]
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]
                round_mse.append(_mse(X_test, y_test, global_w, global_b))
                w_local, _ = _ols_fit(X_train, y_train, d)
                uploads_w.append(w_local)
            fedavg_mse_per_round.append(float(np.mean(round_mse)))
            if t < T - 1:
                global_w = np.mean(uploads_w, axis=0)

        # --- Oracle ---
        concept_w: dict[int, np.ndarray] = {}
        oracle_mse_per_round = []
        for t in range(T):
            round_mse = []
            uploads_by_concept: dict[int, list[np.ndarray]] = {}
            for k in range(K):
                X, y = data[(k, t)]
                cid = int(concept_matrix[k, t])
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]
                w_init = concept_w.get(cid, np.zeros(d, dtype=np.float64))
                round_mse.append(_mse(X_test, y_test, w_init, np.zeros(1)))
                w_local, _ = _ols_fit(X_train, y_train, d)
                uploads_by_concept.setdefault(cid, []).append(w_local)
            oracle_mse_per_round.append(float(np.mean(round_mse)))
            if t < T - 1:
                for cid, ws in uploads_by_concept.items():
                    concept_w[cid] = np.mean(ws, axis=0)

        # --- Shrinkage ---
        concept_w_s: dict[int, np.ndarray] = {}
        shrink_mse_per_round = []
        for t in range(T):
            round_mse = []
            uploads_by_concept_s: dict[int, list[np.ndarray]] = {}
            for k in range(K):
                X, y = data[(k, t)]
                cid = int(concept_matrix[k, t])
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]
                w_init = concept_w_s.get(cid, np.zeros(d, dtype=np.float64))
                round_mse.append(_mse(X_test, y_test, w_init, np.zeros(1)))
                w_local, _ = _ols_fit(X_train, y_train, d)
                uploads_by_concept_s.setdefault(cid, []).append(w_local)
            shrink_mse_per_round.append(float(np.mean(round_mse)))
            if t < T - 1:
                agg_w = {}
                for cid, ws in uploads_by_concept_s.items():
                    agg_w[cid] = np.mean(ws, axis=0)
                if len(agg_w) >= 2:
                    all_cids = list(agg_w.keys())
                    g_w = np.mean([agg_w[c] for c in all_cids], axis=0)
                    between = float(np.mean(np.var(
                        np.array([agg_w[c] for c in all_cids]), axis=0)))
                    within_vars = []
                    for cid2, ws2 in uploads_by_concept_s.items():
                        if len(ws2) >= 2:
                            within_vars.append(float(np.mean(np.var(np.array(ws2), axis=0))))
                    within = np.mean(within_vars) if within_vars else 1e-6
                    n_pg = K / max(len(all_cids), 1)
                    sig_B = max(between - within / max(n_pg, 1), 0.0)
                    noise = within / max(n_pg, 1)
                    lam = noise / (noise + sig_B) if sig_B + noise > 1e-12 else 0.5
                    for cid in all_cids:
                        concept_w_s[cid] = (1 - lam) * agg_w[cid] + lam * g_w
                else:
                    concept_w_s.update(agg_w)

        all_fedavg_mse.append(fedavg_mse_per_round)
        all_oracle_mse.append(oracle_mse_per_round)
        all_shrink_mse.append(shrink_mse_per_round)

    # Average over seeds
    fedavg_avg = np.mean(all_fedavg_mse, axis=0).tolist()
    oracle_avg = np.mean(all_oracle_mse, axis=0).tolist()
    shrink_avg = np.mean(all_shrink_mse, axis=0).tolist()

    # Compute rounds-since-switch for the concept matrix
    # (use first seed's concept matrix for reference)
    concept_matrix_ref, _, _ = generate_gaussian_fl_data(
        K=K, T=T, C=C, d=d, delta=delta, sigma=sigma,
        n_per_client=n_per_client, stability_tau=tau, seed=seeds[0],
    )
    switch_rounds = []
    for t in range(T):
        if t == 0 or any(concept_matrix_ref[k, t] != concept_matrix_ref[k, t - 1] for k in range(K)):
            switch_rounds.append(t)

    return {
        "params": {"K": K, "C": C, "d": d, "delta": delta, "sigma": sigma,
                    "n": n_per_client, "tau": tau, "T": T, "n_seeds": len(seeds)},
        "fedavg_mse": [round(v, 6) for v in fedavg_avg],
        "oracle_mse": [round(v, 6) for v in oracle_avg],
        "shrink_mse": [round(v, 6) for v in shrink_avg],
        "switch_rounds": switch_rounds,
        "oracle_dominates_at": [t for t in range(T) if oracle_avg[t] < fedavg_avg[t]],
        "fedavg_dominates_at": [t for t in range(T) if fedavg_avg[t] < oracle_avg[t]],
    }


def main() -> None:
    out_dir = Path("tmp/transient_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run multiple configurations
    configs = [
        # K/C=1: Oracle has 1 client/concept → cold-start transient visible
        {"K": 4, "C": 4, "delta": 1.0, "tau": 5, "label": "kc1_near_fast"},
        {"K": 4, "C": 4, "delta": 1.0, "tau": 10, "label": "kc1_near_slow"},
        {"K": 4, "C": 4, "delta": 2.0, "tau": 5, "label": "kc1_above_fast"},
        # K/C=2: moderate transient
        {"K": 8, "C": 4, "delta": 0.5, "tau": 5, "label": "kc2_near_fast"},
        {"K": 8, "C": 4, "delta": 0.5, "tau": 10, "label": "kc2_near_slow"},
        {"K": 8, "C": 4, "delta": 1.5, "tau": 5, "label": "kc2_above_fast"},
        # K/C=5: baseline — Oracle always dominates (no transient)
        {"K": 20, "C": 4, "delta": 1.5, "tau": 5, "label": "kc5_above_fast"},
        {"K": 20, "C": 4, "delta": 3.0, "tau": 5, "label": "kc5_far_above_fast"},
    ]

    all_results = {}
    for cfg in configs:
        K = cfg.get("K", 20)
        C = cfg.get("C", 4)
        print(f"\n=== {cfg['label']}: K={K}, C={C}, delta={cfg['delta']}, tau={cfg['tau']} ===")
        result = run_transient_experiment(
            K=K, C=C, delta=cfg["delta"], tau=cfg["tau"], T=40,
            federation_every=1, seeds=[42, 43, 44],
        )
        all_results[cfg["label"]] = result

        # Print switch analysis
        n_oracle_dom = len(result["oracle_dominates_at"])
        n_fedavg_dom = len(result["fedavg_dominates_at"])
        print(f"  Switch rounds: {result['switch_rounds']}")
        print(f"  Oracle dominates in {n_oracle_dom}/40 rounds")
        print(f"  FedAvg dominates in {n_fedavg_dom}/40 rounds")

        # Find transient period after switches
        for sr in result["switch_rounds"][1:]:  # skip t=0
            post_switch = []
            for t in range(sr, min(sr + 10, 40)):
                f_mse = result["fedavg_mse"][t]
                o_mse = result["oracle_mse"][t]
                post_switch.append(("FedAvg" if f_mse < o_mse else "Oracle", round(o_mse - f_mse, 4)))
            print(f"  After switch at t={sr}: {post_switch[:5]}")

    with open(out_dir / "transient_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'transient_results.json'}")


if __name__ == "__main__":
    main()
