from __future__ import annotations

"""Matched-parameter comparison: FedProTrack linear vs feature_adapter.

Both model types use IDENTICAL hyperparameters:
  - federation_every = 2  (10 federation rounds in T=20)
  - n_epochs = 5
  - T = 20, K = 10, n_samples = 500
  - Same TwoPhaseConfig, same seeds

Also runs CFL and Oracle as reference baselines (Oracle with matched n_epochs=5).
Uses 3 seeds for statistical validity.
"""

import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.experiment.baselines import run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import (
    canonical_method_name,
    identity_metrics_valid,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

# ── Shared experiment parameters ──────────────────────────────────────
SEEDS = [42, 43, 44]
K = 10
T = 20
N_SAMPLES = 500
N_FEATURES = 64
SAMPLES_PER_COARSE_CLASS = 30
FEDERATION_EVERY = 2       # identical for all methods
FPT_LR = 0.05
FPT_EPOCHS = 5             # identical for all methods (including Oracle)

METHODS = [
    "FPT-linear",
    "FPT-adapter",
    "CFL",
    "Oracle",
]

RESULTS_DIR = Path("tmp/matched_comparison")


def _make_log(
    method_name: str,
    result: object,
    ground_truth: np.ndarray,
) -> ExperimentLog:
    """Build an ExperimentLog from a runner result."""
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
            getattr(result, "predicted_concept_matrix"), dtype=np.int32
        ),
        accuracy_curve=np.asarray(
            getattr(result, "accuracy_matrix"), dtype=np.float64
        ),
        total_bytes=total_bytes,
        method_name=method_name,
    )


def _build_fpt_config(dataset) -> TwoPhaseConfig:
    """Shared TwoPhaseConfig for both linear and adapter FPT."""
    return TwoPhaseConfig(
        omega=2.0,
        kappa=0.7,
        novelty_threshold=0.25,
        loss_novelty_threshold=0.15,
        sticky_dampening=1.5,
        sticky_posterior_gate=0.35,
        merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, int(dataset.concept_matrix.max()) + 3),
        merge_every=2,
        shrink_every=6,
    )


def _common_fpt_kwargs() -> dict:
    """Shared FPT kwargs that disable all optional routing features."""
    return {
        "auto_scale": False,
        "similarity_calibration": False,
        "model_signature_weight": 0.0,
        "model_signature_dim": 8,
        "update_ot_weight": 0.0,
        "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0,
        "labelwise_proto_dim": 4,
        "prototype_alignment_mix": 0.0,
        "prototype_alignment_early_rounds": 0,
        "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0,
        "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0,
        "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }


def run_single(method: str, dataset, seed: int) -> object:
    """Run a single method on the dataset."""
    cfg = _build_fpt_config(dataset)
    common = _common_fpt_kwargs()

    if method == "FPT-linear":
        return FedProTrackRunner(
            config=cfg,
            federation_every=FEDERATION_EVERY,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=FPT_EPOCHS,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="linear",
            **common,
        ).run(dataset)

    if method == "FPT-adapter":
        return FedProTrackRunner(
            config=cfg,
            federation_every=FEDERATION_EVERY,
            detector_name="ADWIN",
            seed=seed,
            lr=FPT_LR,
            n_epochs=FPT_EPOCHS,
            soft_aggregation=True,
            blend_alpha=0.0,
            model_type="feature_adapter",
            hidden_dim=64,
            adapter_dim=16,
            **common,
        ).run(dataset)

    if method == "CFL":
        return run_cfl_full(dataset, federation_every=FEDERATION_EVERY)

    if method == "Oracle":
        exp_cfg = ExperimentConfig(
            generator_config=dataset.config,
            federation_every=FEDERATION_EVERY,
        )
        return run_oracle_baseline(
            exp_cfg,
            dataset=dataset,
            lr=FPT_LR,
            n_epochs=FPT_EPOCHS,
            seed=seed,
        )

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_failures: list[dict] = []
    total_combos = len(SEEDS) * len(METHODS)
    done = 0

    # Prepare feature cache once
    print("Preparing CIFAR-100 feature cache...", flush=True)
    cache_cfg = CIFAR100RecurrenceConfig(
        K=K,
        T=T,
        n_samples=N_SAMPLES,
        seed=42,
        n_features=N_FEATURES,
        samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS,
    )
    prepare_cifar100_recurrence_feature_cache(cache_cfg)

    for seed in SEEDS:
        print(f"\n{'='*60}", flush=True)
        print(f"  Seed = {seed}", flush=True)
        print(f"{'='*60}", flush=True)

        dataset_cfg = CIFAR100RecurrenceConfig(
            K=K,
            T=T,
            n_samples=N_SAMPLES,
            rho=2.0,
            alpha=0.75,
            delta=0.9,
            n_features=N_FEATURES,
            samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS,
            batch_size=128,
            n_workers=0,
            seed=seed,
        )
        dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

        for method in METHODS:
            done += 1
            print(f"  [{done}/{total_combos}] {method}...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = run_single(method, dataset, seed)
                log = _make_log(method, result, dataset.concept_matrix)
                canon = canonical_method_name(
                    "FedProTrack" if method.startswith("FPT") else method
                )
                metrics = compute_all_metrics(
                    log, identity_capable=identity_metrics_valid(canon)
                )
                row = {
                    "method": method,
                    "seed": seed,
                    "final_accuracy": metrics.final_accuracy,
                    "accuracy_auc": metrics.accuracy_auc,
                    "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                    "assignment_entropy": metrics.assignment_entropy,
                    "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
                    "total_bytes": float(getattr(result, "total_bytes", 0.0) or 0.0),
                    "wall_clock_s": time.time() - t0,
                }
                all_rows.append(row)
                acc_s = f"acc={row['final_accuracy']:.4f}"
                reid_s = (
                    f"re-ID={row['concept_re_id_accuracy']:.4f}"
                    if row["concept_re_id_accuracy"] is not None
                    else "re-ID=N/A"
                )
                ent_s = (
                    f"entropy={row['assignment_entropy']:.4f}"
                    if row["assignment_entropy"] is not None
                    else "entropy=N/A"
                )
                print(
                    f"{acc_s}  {reid_s}  {ent_s}  bytes={row['total_bytes']:.0f}  ({time.time()-t0:.1f}s)",
                    flush=True,
                )
            except Exception as exc:
                all_failures.append(
                    {
                        "method": method,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "wall_clock_s": time.time() - t0,
                    }
                )
                print(f"FAILED: {exc}", flush=True)

    # ── Save raw results ──────────────────────────────────────────────
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "failures": all_failures}, f, indent=2)

    fieldnames = [
        "method", "seed", "final_accuracy", "accuracy_auc",
        "concept_re_id_accuracy", "assignment_entropy",
        "wrong_memory_reuse_rate", "total_bytes", "wall_clock_s",
    ]
    with open(RESULTS_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "=" * 90, flush=True)
    print("  MATCHED COMPARISON SUMMARY  (mean +/- std over 3 seeds)", flush=True)
    print("  federation_every=2, n_epochs=5, T=20, K=10, n_samples=500", flush=True)
    print("=" * 90, flush=True)
    header = (
        f"{'Method':<16s}"
        f"  {'FinalAcc':>14s}"
        f"  {'Re-ID':>14s}"
        f"  {'Entropy':>14s}"
        f"  {'AUC':>14s}"
        f"  {'Bytes':>10s}"
    )
    print(header, flush=True)
    print("-" * 90, flush=True)

    for method in METHODS:
        subset = [r for r in all_rows if r["method"] == method]
        if not subset:
            continue
        accs = [r["final_accuracy"] for r in subset]
        reids = [
            r["concept_re_id_accuracy"]
            for r in subset
            if r["concept_re_id_accuracy"] is not None
        ]
        ents = [
            r["assignment_entropy"]
            for r in subset
            if r["assignment_entropy"] is not None
        ]
        aucs = [r["accuracy_auc"] for r in subset]
        bytes_vals = [r["total_bytes"] for r in subset]

        acc_str = f"{np.mean(accs):.4f}+/-{np.std(accs):.4f}"
        reid_str = f"{np.mean(reids):.4f}+/-{np.std(reids):.4f}" if reids else "N/A"
        ent_str = f"{np.mean(ents):.4f}+/-{np.std(ents):.4f}" if ents else "N/A"
        auc_str = f"{np.mean(aucs):.4f}+/-{np.std(aucs):.4f}"
        bytes_str = f"{np.mean(bytes_vals):.0f}"

        print(
            f"{method:<16s}"
            f"  {acc_str:>14s}"
            f"  {reid_str:>14s}"
            f"  {ent_str:>14s}"
            f"  {auc_str:>14s}"
            f"  {bytes_str:>10s}",
            flush=True,
        )

    print(f"\nFailures: {len(all_failures)}", flush=True)
    print(f"Results saved to: {RESULTS_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
