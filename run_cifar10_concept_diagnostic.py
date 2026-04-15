"""CIFAR-10 concept-discovery diagnostic.

Answers: why does FPT-OT fall short of Oracle on CIFAR-10 even with warmup?
Key hypothesis checks:
  1. Does FPT discover C>=2 concepts, or does it collapse to C=1 (== FedAvg)?
  2. If C>=2, what is the clustering error eta vs. the true concept matrix?
  3. How many concepts does the data actually have per round?

Runs one seed (44 — the only seed that showed a gap) and dumps per-round
diagnostics for FedProTrack-OT, plus headline accuracies for Oracle/FedAvg.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.posterior import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR10RecurrenceConfig,
    generate_cifar10_recurrence_dataset,
    prepare_cifar10_recurrence_feature_cache,
)


def clustering_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Hungarian-style clustering error: fraction of (k,t) whose predicted
    cluster, after optimal label matching, disagrees with the true concept."""
    from scipy.optimize import linear_sum_assignment
    pred_ids = np.unique(pred)
    true_ids = np.unique(true)
    cost = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for i, p in enumerate(pred_ids):
        for j, t in enumerate(true_ids):
            cost[i, j] = -int(np.sum((pred == p) & (true == t)))
    row, col = linear_sum_assignment(cost)
    matched = sum(-cost[r, c] for r, c in zip(row, col))
    return 1.0 - matched / pred.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--rho", type=float, default=25.0)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.85)
    ap.add_argument("--feature-seed", type=int, default=2718)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--fpt-mode", default="ot", help="Ignored; always OT.")
    ap.add_argument("--no-drct-shrinkage", action="store_true",
                    help="Disable empirical-Bayes shrinkage (pure per-concept aggregation).")
    ap.add_argument("--snr-gate", action="store_true",
                    help="Enable SNR-gated shrinkage (λ=0 when σ_B²/(σ²·d_eff/n̄) < threshold).")
    ap.add_argument("--snr-threshold", type=float, default=1.0)
    ap.add_argument("--sigma-ema-beta", type=float, default=0.0,
                    help="EMA factor for σ², σ_B² smoothing (0=off).")
    ap.add_argument("--data-root", default=".cifar10_cache",
                    help="CIFAR-10 raw data cache dir.")
    ap.add_argument("--feature-cache-dir", default=".feature_cache")
    ap.add_argument("--n-workers", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="If set, write summary.json into this directory (handler convention).")
    args = ap.parse_args()
    if args.out is None:
        args.out = (
            f"{args.results_dir}/summary.json" if args.results_dir
            else "results_cifar10_concept_diag.json"
        )
    if args.results_dir:
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    cfg = CIFAR10RecurrenceConfig(
        K=args.K, T=args.T, n_samples=400, rho=args.rho,
        alpha=args.alpha, delta=args.delta, n_features=128,
        batch_size=256, n_workers=0,
        data_root=args.data_root, feature_cache_dir=args.feature_cache_dir,
        feature_seed=args.feature_seed, seed=args.seed,
    )
    prepare_cifar10_recurrence_feature_cache(cfg)
    dataset = generate_cifar10_recurrence_dataset(cfg)

    true_cm = dataset.concept_matrix  # (K, T)
    n_true_concepts = int(true_cm.max()) + 1
    print(f"Dataset: K={args.K} T={args.T} true_concepts={n_true_concepts}")
    print(f"True concept matrix unique IDs per round (sample):")
    for t in [0, args.T // 4, args.T // 2, 3 * args.T // 4, args.T - 1]:
        uniq = np.unique(true_cm[:, t])
        print(f"  t={t:3d}: concepts active = {uniq.tolist()}")

    # FPT-OT run
    two_phase_cfg = TwoPhaseConfig(
        omega=2.0, kappa=0.7,
        novelty_threshold=0.25, loss_novelty_threshold=0.15,
        sticky_dampening=1.5, sticky_posterior_gate=0.35,
        merge_threshold=0.85, min_count=5.0,
        max_concepts=max(6, n_true_concepts + 3),
        merge_every=2, shrink_every=6,
        drct_shrinkage=not args.no_drct_shrinkage, drct_d_eff_ratio=0.9,
        drct_min_concepts=2, drct_warmup_rounds=args.warmup,
        drct_snr_gate=args.snr_gate, drct_snr_threshold=args.snr_threshold,
        drct_sigma_ema_beta=args.sigma_ema_beta,
    )
    runner = FedProTrackRunner(
        config=two_phase_cfg, seed=args.seed, federation_every=1,
        detector_name="ADWIN", lr=0.01, n_epochs=5,
        soft_aggregation=False, blend_alpha=0.0,
        concept_discovery="ot",
    )
    fpt_result = runner.run(dataset)

    pred_cm = fpt_result.predicted_concept_matrix  # (K, T)
    # Per-round: how many distinct predicted concepts?
    n_pred_per_round = [int(np.unique(pred_cm[:, t]).size) for t in range(args.T)]
    n_true_per_round = [int(np.unique(true_cm[:, t]).size) for t in range(args.T)]
    eta_per_round = []
    for t in range(args.T):
        # Only meaningful if >1 true concept active
        if np.unique(true_cm[:, t]).size < 2 or np.unique(pred_cm[:, t]).size < 2:
            eta_per_round.append(None)
            continue
        eta = clustering_error(pred_cm[:, t], true_cm[:, t])
        eta_per_round.append(float(eta))

    eta_overall = clustering_error(pred_cm, true_cm)

    # Baselines
    exp_cfg = ExperimentConfig(generator_config=dataset.config, federation_every=1)
    fa = run_fedavg_baseline(exp_cfg, dataset=dataset, lr=0.01, n_epochs=5, seed=args.seed)
    orc = run_oracle_baseline(exp_cfg, dataset=dataset, lr=0.01, n_epochs=5, seed=args.seed)

    summary = {
        "seed": args.seed,
        "warmup_rounds": args.warmup,
        "drct_shrinkage": not args.no_drct_shrinkage,
        "snr_gate": args.snr_gate,
        "snr_threshold": args.snr_threshold,
        "sigma_ema_beta": args.sigma_ema_beta,
        "config": {"K": args.K, "T": args.T, "rho": args.rho,
                   "alpha": args.alpha, "delta": args.delta,
                   "feature_seed": args.feature_seed},
        "n_true_concepts_total": n_true_concepts,
        "fpt_active_concepts_final": int(fpt_result.active_concepts),
        "fpt_spawned_concepts": int(fpt_result.spawned_concepts),
        "fpt_merged_concepts": int(fpt_result.merged_concepts),
        "fpt_pruned_concepts": int(fpt_result.pruned_concepts),
        "fpt_eta_overall": float(eta_overall),
        "fpt_final_acc": float(fpt_result.accuracy_matrix[:, -1].mean()),
        "fedavg_final_acc": float(fa.accuracy_matrix[:, -1].mean()),
        "oracle_final_acc": float(orc.accuracy_matrix[:, -1].mean()),
        "n_pred_per_round": n_pred_per_round,
        "n_true_per_round": n_true_per_round,
        "eta_per_round": eta_per_round,
        "predicted_concept_matrix": pred_cm.tolist(),
        "true_concept_matrix": true_cm.tolist(),
        "drct_lambda_log": list(fpt_result.drct_lambda_log),
        "phase_a_round_diagnostics": list(fpt_result.phase_a_round_diagnostics),
        "memory_reuse_rate": float(fpt_result.memory_reuse_rate) if fpt_result.memory_reuse_rate is not None else None,
        "assignment_switch_rate": float(fpt_result.assignment_switch_rate) if fpt_result.assignment_switch_rate is not None else None,
        "avg_clients_per_concept": float(fpt_result.avg_clients_per_concept) if fpt_result.avg_clients_per_concept is not None else None,
        "singleton_group_ratio": float(fpt_result.singleton_group_ratio) if fpt_result.singleton_group_ratio is not None else None,
        "routing_consistency": float(fpt_result.routing_consistency) if fpt_result.routing_consistency is not None else None,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    # Headline table
    print(f"\n=== Headline ===")
    print(f"  FPT-OT final acc     : {summary['fpt_final_acc']:.4f}")
    print(f"  FedAvg final acc     : {summary['fedavg_final_acc']:.4f}")
    print(f"  Oracle final acc     : {summary['oracle_final_acc']:.4f}")
    print(f"  Δ FPT-FedAvg (pp)    : {100*(summary['fpt_final_acc']-summary['fedavg_final_acc']):+.2f}")
    print(f"  Δ FPT-Oracle (pp)    : {100*(summary['fpt_final_acc']-summary['oracle_final_acc']):+.2f}")
    print(f"\n=== Concept discovery ===")
    print(f"  True concepts in data     : {n_true_concepts}")
    print(f"  FPT final active concepts : {fpt_result.active_concepts}")
    print(f"  FPT spawned / merged / pruned : "
          f"{fpt_result.spawned_concepts} / {fpt_result.merged_concepts} / {fpt_result.pruned_concepts}")
    print(f"  Clustering error η (overall): {eta_overall:.4f}")
    print(f"  n_pred_per_round : min={min(n_pred_per_round)} max={max(n_pred_per_round)} "
          f"mean={np.mean(n_pred_per_round):.2f} median={int(np.median(n_pred_per_round))}")
    print(f"  n_true_per_round : min={min(n_true_per_round)} max={max(n_true_per_round)} "
          f"mean={np.mean(n_true_per_round):.2f}")
    # How many rounds had C_pred=1?
    n_collapsed = sum(1 for n in n_pred_per_round if n == 1)
    print(f"  Rounds with predicted C=1 (collapsed to FedAvg): "
          f"{n_collapsed}/{args.T} ({100*n_collapsed/args.T:.0f}%)")

    # Per-round spawn/merge trace
    diag = fpt_result.phase_a_round_diagnostics
    spawns = [d.get("spawned", 0) for d in diag]
    merges = [d.get("merged", 0) for d in diag]
    libs = [d.get("library_size_before", 0) for d in diag]
    actives = [d.get("active_after", 0) for d in diag]
    fp_losses = [d.get("mean_fp_loss") for d in diag if d.get("mean_fp_loss") is not None]
    thresholds = [d.get("mean_effective_threshold") for d in diag if d.get("mean_effective_threshold") is not None]
    over_rates = [d.get("over_threshold_rate") for d in diag if d.get("over_threshold_rate") is not None]
    print(f"\n=== Spawn / merge trace ({len(diag)} rounds) ===")
    print(f"  Total spawns : {sum(spawns)}    total merges : {sum(merges)}")
    print(f"  Rounds with spawn>0 : {sum(1 for s in spawns if s>0)}")
    print(f"  Rounds with merge>0 : {sum(1 for m in merges if m>0)}")
    print(f"  Library size: start={libs[0] if libs else 0} end={libs[-1] if libs else 0} "
          f"max={max(libs) if libs else 0}")
    print(f"  Active after: start={actives[0] if actives else 0} end={actives[-1] if actives else 0} "
          f"max={max(actives) if actives else 0}")
    if fp_losses:
        print(f"  mean_fp_loss          : μ={np.mean(fp_losses):.4f} min={min(fp_losses):.4f} max={max(fp_losses):.4f}")
    if thresholds:
        print(f"  mean_effective_threshold: μ={np.mean(thresholds):.4f} min={min(thresholds):.4f} max={max(thresholds):.4f}")
    if over_rates:
        print(f"  over_threshold_rate   : μ={np.mean(over_rates):.4f}  (novelty trigger rate)")
    print(f"  memory_reuse_rate     : {summary['memory_reuse_rate']}")
    print(f"  assignment_switch_rate: {summary['assignment_switch_rate']}")
    print(f"  singleton_group_ratio : {summary['singleton_group_ratio']}")
    print(f"\n  First 10 rounds: (t, spawned, merged, library_before, active_after)")
    for d in diag[:10]:
        print(f"    t={d.get('t'):3d}  spawn={d.get('spawned'):2d}  merge={d.get('merged'):2d}  "
              f"lib={d.get('library_size_before'):3d}  active={d.get('active_after'):3d}")


if __name__ == "__main__":
    main()
