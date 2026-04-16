from __future__ import annotations

"""Focused experiment: OT concept discovery vs Gibbs vs Oracle vs FedAvg.

Tests the hypothesis that OT spectral clustering discovers the correct
number of concepts (unlike Gibbs which discovers 1-4 for 4+ true),
and that correct concept discovery enables FPT to beat FedAvg.
"""

import json
import os
import sys
import time

import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from fedprotrack.drift_generator.generator import DriftDataset
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiment.baselines import run_oracle_baseline
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data.cifar100_recurrence import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
)
from fedprotrack.experiment.baselines import run_fedavg_baseline


def _make_exp_cfg(ds: DriftDataset) -> ExperimentConfig:
    """Build ExperimentConfig from a DriftDataset."""
    return ExperimentConfig(generator_config=ds.config)


def run_single(
    K: int,
    T: int,
    rho: float,
    seed: int,
    lr: float = 0.05,
    n_epochs: int = 5,
    fpt_mode: str = "ot",
    data_root: str | None = None,
    feature_cache_dir: str | None = None,
) -> dict:
    """Run a single experiment configuration."""
    cfg_kwargs = dict(K=K, T=T, seed=seed, rho=rho)
    if data_root is not None:
        cfg_kwargs["data_root"] = data_root
    if feature_cache_dir is not None:
        cfg_kwargs["feature_cache_dir"] = feature_cache_dir
    cfg = CIFAR100RecurrenceConfig(**cfg_kwargs)
    ds = generate_cifar100_recurrence_dataset(cfg)
    exp_cfg = _make_exp_cfg(ds)
    true_C = int(ds.concept_matrix.max()) + 1
    n_active_per_t = [len(np.unique(ds.concept_matrix[:, t])) for t in range(T)]
    avg_active = np.mean(n_active_per_t)

    results = {}

    # --- FedProTrack-OT ---
    if fpt_mode in ("ot", "both"):
        t0 = time.time()
        ot_runner = FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, true_C + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=1,
            detector_name="ADWIN",
            seed=seed,
            lr=lr,
            n_epochs=n_epochs,
            soft_aggregation=False,
            concept_discovery="ot",
            blend_alpha=0.0,
        )
        ot_result = ot_runner.run(ds)
        elapsed = time.time() - t0

        # Count concepts actually used per round from predicted_concept_matrix.
        # This directly reflects how many distinct concept IDs FPT assigned
        # (independent of phase_a hook behaviour).
        ot_pred = ot_result.predicted_concept_matrix  # shape (K, T)
        ot_counts_per_t = [
            int(len(np.unique(ot_pred[:, tt]))) for tt in range(T)
        ]
        results["FPT-OT"] = {
            "acc": float(ot_result.mean_accuracy),
            "bytes": float(ot_result.total_bytes),
            "time": elapsed,
            "concepts_avg": float(np.mean(ot_counts_per_t)),
            "concepts_max": int(max(ot_counts_per_t)),
            "active_concepts_total": int(ot_result.active_concepts),
        }

    # --- FedProTrack-Gibbs (calibrated) ---
    if fpt_mode in ("gibbs", "both"):
        t0 = time.time()
        gibbs_runner = FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, true_C + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=1,
            detector_name="ADWIN",
            seed=seed,
            lr=lr,
            n_epochs=n_epochs,
            soft_aggregation=True,
            concept_discovery="gibbs",
            similarity_calibration=True,
            blend_alpha=0.0,
        )
        gibbs_result = gibbs_runner.run(ds)
        elapsed = time.time() - t0

        g_pred = gibbs_result.predicted_concept_matrix
        g_counts_per_t = [
            int(len(np.unique(g_pred[:, tt]))) for tt in range(T)
        ]
        results["FPT-Gibbs"] = {
            "acc": float(gibbs_result.mean_accuracy),
            "bytes": float(gibbs_result.total_bytes),
            "time": elapsed,
            "concepts_avg": float(np.mean(g_counts_per_t)),
            "concepts_max": int(max(g_counts_per_t)),
            "active_concepts_total": int(gibbs_result.active_concepts),
        }

    # --- FedAvg ---
    t0 = time.time()
    fa_result = run_fedavg_baseline(
        exp_cfg, dataset=ds, lr=lr, n_epochs=n_epochs,
    )
    results["FedAvg"] = {
        "acc": float(fa_result.mean_accuracy),
        "bytes": float(fa_result.total_bytes),
        "time": time.time() - t0,
    }

    # --- Oracle (old API, lr=0.1/epochs=1 — NOTE mismatch) ---
    t0 = time.time()
    oracle_result = run_oracle_baseline(exp_cfg, dataset=ds)
    results["Oracle"] = {
        "acc": float(oracle_result.mean_accuracy),
        "bytes": 0.0,
        "time": time.time() - t0,
        "note": "lr=0.1/epochs=1 (old API, not matched)",
    }

    meta = {
        "K": K,
        "T": T,
        "rho": rho,
        "seed": seed,
        "true_C": true_C,
        "avg_active": float(avg_active),
        "K_over_C": float(K / avg_active),
    }
    return {"meta": meta, "results": results}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OT concept discovery experiment")
    parser.add_argument("--K", type=int, nargs="+", default=[20, 40])
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--rho", type=float, nargs="+", default=[25.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="One or more seeds (local usage)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Single seed (RunPod handler compatibility)")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--mode", default="both", choices=["ot", "gibbs", "both"])
    parser.add_argument("--output", default=None,
                        help="Explicit output JSON file (overrides --results-dir)")
    parser.add_argument("--results-dir", default=None,
                        help="Output directory (RunPod handler convention); "
                             "results written to <dir>/results.json")
    parser.add_argument("--data-root", default=None,
                        help="CIFAR-100 dataset root (RunPod handler convention)")
    parser.add_argument("--feature-cache-dir", default=None,
                        help="ResNet feature cache dir (RunPod handler convention)")
    parser.add_argument("--n-workers", type=int, default=0,
                        help="Ignored; accepted for RunPod handler compatibility")
    args = parser.parse_args()

    # Resolve seeds: prefer --seed (singular, handler), fall back to --seeds
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = [42]

    # Resolve output path
    if args.output:
        output_path = args.output
    elif args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        output_path = os.path.join(args.results_dir, "results.json")
    else:
        output_path = "results_ot_experiment.json"

    all_results = []
    total = len(args.K) * len(args.rho) * len(seeds)
    idx = 0

    for K in args.K:
        for rho in args.rho:
            for seed in seeds:
                idx += 1
                print(f"\n[{idx}/{total}] K={K} rho={rho} seed={seed}")
                result = run_single(
                    K=K, T=args.T, rho=rho, seed=seed,
                    lr=args.lr, n_epochs=args.n_epochs, fpt_mode=args.mode,
                    data_root=args.data_root,
                    feature_cache_dir=args.feature_cache_dir,
                )
                all_results.append(result)

                # Print summary
                meta = result["meta"]
                print(f"  true_C={meta['true_C']} avg_active={meta['avg_active']:.1f} K/C={meta['K_over_C']:.1f}")
                for name, r in result["results"].items():
                    concepts_info = ""
                    if "concepts_avg" in r:
                        concepts_info = f" C_avg={r['concepts_avg']:.1f} C_max={r['concepts_max']}"
                    print(f"  {name:15s}: acc={r['acc']:.4f} bytes={r['bytes']:.0f}{concepts_info}")

    # Save
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
