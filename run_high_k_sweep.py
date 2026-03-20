from __future__ import annotations

"""High-K sweep: test FedProTrack advantage with K >> n_concepts.

Hypothesis: FPT's concept routing hurts accuracy when K ~ n_concepts because
each concept group has only 1-2 clients, eliminating aggregation benefit.
With K >> n_concepts, each concept group should have 3+ clients, making
concept-aware aggregation genuinely useful.

Tests K in [10, 20, 30] with C=4 concepts, T=30, n_samples=500.
Methods: FedProTrack, CFL, FedAvg, Oracle.
Seeds: 42, 123, 456.
"""

import csv
import json
import time
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full, MethodResult
from fedprotrack.drift_generator.concept_matrix import generate_concept_matrix_low_singleton
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.data_streams import make_concept_specs, generate_samples
from fedprotrack.drift_generator.generator import DriftDataset
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_oracle_baseline,
    _infer_n_features,
    _infer_n_classes,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig

# -- Experiment grid ----------------------------------------------------------
K_VALUES = [10, 20, 30]
N_CONCEPTS = 4
T = 30
N_SAMPLES = 500
FEDERATION_EVERY = 2
N_EPOCHS = 5
LR = 0.1
SEEDS = [42, 123, 456]
GENERATOR_TYPE = "sine"
DELTA = 0.5
ALPHA = 0.5
MIN_GROUP_SIZE = 2


def _make_dataset(K: int, seed: int) -> DriftDataset:
    """Build a synthetic DriftDataset with a low-singleton concept matrix."""
    # Use rho such that n_concepts = max(2, T/rho) = 4  =>  rho = T/4 = 7.5
    rho = T / N_CONCEPTS

    concept_matrix = generate_concept_matrix_low_singleton(
        K=K, T=T, n_concepts=N_CONCEPTS, alpha=ALPHA, seed=seed,
        min_group_size=MIN_GROUP_SIZE,
    )

    gc = GeneratorConfig(
        K=K, T=T, n_samples=N_SAMPLES, rho=rho, alpha=ALPHA,
        delta=DELTA, generator_type=GENERATOR_TYPE, seed=seed,
    )

    n_concepts_actual = int(concept_matrix.max()) + 1
    specs = make_concept_specs(n_concepts_actual, GENERATOR_TYPE, DELTA)

    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            concept_id = int(concept_matrix[k, t])
            sample_seed = seed + k * T + t + 10000
            X, y = generate_samples(specs[concept_id], N_SAMPLES, sample_seed)
            data[(k, t)] = (X, y)

    return DriftDataset(
        concept_matrix=concept_matrix, data=data, config=gc, concept_specs=specs,
    )


def _make_experiment_log(
    method_name: str, result: object, ground_truth: np.ndarray,
) -> ExperimentLog:
    """Convert any result object to ExperimentLog."""
    total_bytes = getattr(result, "total_bytes", None)
    if total_bytes is not None and float(total_bytes) <= 0.0:
        total_bytes = None
    if hasattr(result, "to_experiment_log"):
        try:
            return result.to_experiment_log(ground_truth)
        except TypeError:
            return result.to_experiment_log()
    return ExperimentLog(
        ground_truth=ground_truth,
        predicted=np.asarray(
            getattr(result, "predicted_concept_matrix"), dtype=np.int32,
        ),
        accuracy_curve=np.asarray(
            getattr(result, "accuracy_matrix"), dtype=np.float64,
        ),
        total_bytes=total_bytes,
        method_name=method_name,
    )


def _run_fedprotrack(
    dataset: DriftDataset, seed: int, *, label: str = "tuned",
) -> object:
    """Run FedProTrack with specified config preset.

    Parameters
    ----------
    label : str
        "tuned" uses conservative thresholds that keep concepts alive.
        "default" uses the v3 defaults (loss_novelty=0.02, merge=0.85).
    """
    if label == "default":
        tp_cfg = TwoPhaseConfig(
            omega=2.0, kappa=0.7,
            novelty_threshold=0.25,
            loss_novelty_threshold=0.02,
            sticky_dampening=1.0,
            merge_threshold=0.85,
            min_count=5.0,
            max_concepts=max(6, N_CONCEPTS + 3),
            merge_every=2, shrink_every=6,
        )
    else:
        # "tuned" -- lower merge threshold, higher novelty threshold
        # to keep distinct concepts alive for aggregation benefit
        tp_cfg = TwoPhaseConfig(
            omega=2.0, kappa=0.7,
            novelty_threshold=0.15,
            loss_novelty_threshold=0.05,
            sticky_dampening=1.0,
            merge_threshold=0.95,
            min_count=3.0,
            max_concepts=max(8, N_CONCEPTS + 4),
            merge_every=4, shrink_every=10,
        )

    runner = FedProTrackRunner(
        config=tp_cfg,
        federation_every=FEDERATION_EVERY,
        detector_name="ADWIN",
        seed=seed,
        lr=LR,
        n_epochs=N_EPOCHS,
        soft_aggregation=True,
        blend_alpha=0.0,
    )
    return runner.run(dataset)


def _run_fedavg(dataset: DriftDataset, seed: int) -> object:
    """Run FedAvg baseline."""
    gc = dataset.config
    exp_cfg = ExperimentConfig(generator_config=gc, federation_every=FEDERATION_EVERY)
    return run_fedavg_baseline(exp_cfg, dataset, lr=LR, n_epochs=N_EPOCHS, seed=seed)


def _run_oracle(dataset: DriftDataset, seed: int) -> object:
    """Run Oracle baseline."""
    gc = dataset.config
    exp_cfg = ExperimentConfig(generator_config=gc, federation_every=FEDERATION_EVERY)
    return run_oracle_baseline(exp_cfg, dataset, lr=LR, n_epochs=N_EPOCHS, seed=seed)


def _run_cfl(dataset: DriftDataset) -> MethodResult:
    """Run CFL baseline."""
    return run_cfl_full(dataset, federation_every=FEDERATION_EVERY, max_clusters=8)


def _group_size_stats(concept_matrix: np.ndarray) -> dict:
    """Compute per-timestep group size statistics."""
    K, T_mat = concept_matrix.shape
    min_gs, mean_gs, max_gs = [], [], []
    for t in range(T_mat):
        col = concept_matrix[:, t]
        unique, counts = np.unique(col, return_counts=True)
        min_gs.append(int(counts.min()))
        mean_gs.append(float(counts.mean()))
        max_gs.append(int(counts.max()))
    return {
        "min_group_size": int(np.min(min_gs)),
        "mean_group_size": float(np.mean(mean_gs)),
        "max_group_size": int(np.max(max_gs)),
        "n_active_concepts_mean": float(np.mean([
            len(np.unique(concept_matrix[:, t])) for t in range(T_mat)
        ])),
    }


def main() -> None:
    results_dir = Path("tmp/high_k_sweep")
    results_dir.mkdir(parents=True, exist_ok=True)

    methods = ["FedProTrack", "FPT-default", "CFL", "FedAvg", "Oracle"]
    all_rows: list[dict] = []
    t0_global = time.time()

    for K in K_VALUES:
        print(f"\n{'='*60}")
        print(f"  K = {K}  (C={N_CONCEPTS}, expected group size ~{K/N_CONCEPTS:.1f})")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n--- seed={seed} ---")
            dataset = _make_dataset(K, seed)
            gs = _group_size_stats(dataset.concept_matrix)
            print(f"  Group sizes: min={gs['min_group_size']}, "
                  f"mean={gs['mean_group_size']:.1f}, "
                  f"n_active={gs['n_active_concepts_mean']:.1f}")

            for method in methods:
                t0 = time.time()
                try:
                    if method == "FedProTrack":
                        result = _run_fedprotrack(dataset, seed, label="tuned")
                    elif method == "FPT-default":
                        result = _run_fedprotrack(dataset, seed, label="default")
                    elif method == "CFL":
                        result = _run_cfl(dataset)
                    elif method == "FedAvg":
                        result = _run_fedavg(dataset, seed)
                    elif method == "Oracle":
                        result = _run_oracle(dataset, seed)
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    log = _make_experiment_log(method, result, dataset.concept_matrix)
                    identity_capable = method in ("FedProTrack", "FPT-default", "CFL", "Oracle")
                    metrics = compute_all_metrics(log, identity_capable=identity_capable)
                    elapsed = time.time() - t0

                    # Compute mean accuracy directly from accuracy matrix
                    acc_mat = np.asarray(getattr(result, "accuracy_matrix"))
                    mean_acc = float(acc_mat.mean())

                    # Diagnostics for FPT
                    diag_str = ""
                    if method in ("FedProTrack", "FPT-default"):
                        pred = getattr(result, "predicted_concept_matrix")
                        n_pred_concepts = len(np.unique(pred))
                        spawned = getattr(result, "spawned_concepts", None)
                        active = getattr(result, "active_concepts", None)
                        diag_str = (f"  pred_concepts={n_pred_concepts}"
                                    f" spawned={spawned} active={active}")

                    row = {
                        "K": K,
                        "seed": seed,
                        "method": method,
                        "final_acc": round(metrics.final_accuracy or 0.0, 4),
                        "mean_acc": round(mean_acc, 4),
                        "re_id": round(metrics.concept_re_id_accuracy or 0.0, 4),
                        "total_bytes": log.total_bytes or 0.0,
                        "elapsed_s": round(elapsed, 1),
                        **{f"gs_{k2}": v for k2, v in gs.items()},
                    }
                    all_rows.append(row)
                    print(f"  {method:15s}  final_acc={row['final_acc']:.3f}  "
                          f"mean_acc={row['mean_acc']:.3f}  "
                          f"re_id={row['re_id']:.3f}  [{elapsed:.1f}s]"
                          f"{diag_str}")
                except Exception as e:
                    print(f"  {method:15s}  ERROR: {e}")
                    all_rows.append({
                        "K": K, "seed": seed, "method": method,
                        "final_acc": None, "mean_acc": None, "re_id": None,
                        "total_bytes": None, "elapsed_s": None, "error": str(e),
                    })

    # -- Save raw CSV --
    csv_path = results_dir / "raw_results.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        # Ensure all keys present
        for row in all_rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nRaw results saved to {csv_path}")

    # -- Aggregate summary --
    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY (mean +/- std over seeds)")
    print(f"{'='*70}")
    summary_rows = []
    for K in K_VALUES:
        print(f"\nK = {K}  (expected group size ~{K/N_CONCEPTS:.1f})")
        print(f"  {'Method':15s}  {'Final Acc':>12s}  {'Mean Acc':>12s}  {'Re-ID':>12s}")
        print(f"  {'-'*55}")
        for method in methods:
            subset = [r for r in all_rows
                      if r["K"] == K and r["method"] == method and r.get("final_acc") is not None]
            if not subset:
                print(f"  {method:15s}  (no results)")
                continue
            fa = np.array([r["final_acc"] for r in subset])
            ma = np.array([r["mean_acc"] for r in subset])
            ri = np.array([r["re_id"] for r in subset])
            print(f"  {method:15s}  {fa.mean():.3f}+/-{fa.std():.3f}  "
                  f"{ma.mean():.3f}+/-{ma.std():.3f}  "
                  f"{ri.mean():.3f}+/-{ri.std():.3f}")
            summary_rows.append({
                "K": K,
                "method": method,
                "final_acc_mean": round(float(fa.mean()), 4),
                "final_acc_std": round(float(fa.std()), 4),
                "mean_acc_mean": round(float(ma.mean()), 4),
                "mean_acc_std": round(float(ma.std()), 4),
                "re_id_mean": round(float(ri.mean()), 4),
                "re_id_std": round(float(ri.std()), 4),
                "n_seeds": len(subset),
            })

    # FPT vs FedAvg gap
    print(f"\n{'='*70}")
    print("FPT vs FedAvg ACCURACY GAP")
    print(f"{'='*70}")
    for K in K_VALUES:
        fpt = [r for r in all_rows if r["K"] == K and r["method"] == "FedProTrack"
               and r.get("final_acc") is not None]
        fa = [r for r in all_rows if r["K"] == K and r["method"] == "FedAvg"
              and r.get("final_acc") is not None]
        orc = [r for r in all_rows if r["K"] == K and r["method"] == "Oracle"
               and r.get("final_acc") is not None]
        if fpt and fa:
            fpt_mean = np.mean([r["final_acc"] for r in fpt])
            fa_mean = np.mean([r["final_acc"] for r in fa])
            gap = fpt_mean - fa_mean
            sign = "+" if gap > 0 else ""
            print(f"  K={K:3d}: FPT-FedAvg = {sign}{gap:.4f}  "
                  f"(FPT={fpt_mean:.3f}, FedAvg={fa_mean:.3f})")
        if fpt and orc:
            fpt_mean = np.mean([r["final_acc"] for r in fpt])
            orc_mean = np.mean([r["final_acc"] for r in orc])
            ratio = fpt_mean / orc_mean if orc_mean > 0 else 0
            print(f"         FPT/Oracle = {ratio:.3f}  "
                  f"(FPT={fpt_mean:.3f}, Oracle={orc_mean:.3f})")

    # Save summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    elapsed_total = time.time() - t0_global
    print(f"\nTotal wall-clock: {elapsed_total:.0f}s ({elapsed_total/60:.1f}min)")


if __name__ == "__main__":
    main()
