from __future__ import annotations

"""Misspecified cluster-count experiment on CIFAR-100.

Fix true C=4 (rho=3.0, T=12), give cluster-aware baselines wrong
n_clusters ∈ {2, 3, 4, 6, 8}.  FPT, CFL, Oracle, FedAvg shown as
cluster-count-free reference lines.

Usage (local):
    python run_misspecified_c.py --seed 42 --n-clusters 2

RunPod submission (5 n_clusters × 3 seeds = 15 jobs):
    for nc in 2 3 4 6 8; do
      python runpod/submit_experiment.py --script run_misspecified_c.py \
        --seeds 42 43 44 --n-clusters $nc \
        --out-file runpod_results_misspec_nc${nc}.json
    done
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from fedprotrack.baselines.runners import (
    run_cfl_full,
    run_fedem_full,
    run_fedrc_full,
    run_fesem_full,
    run_flux_prior_full,
    run_ifca_full,
)
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_oracle_baseline,
)
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


def _make_log(
    method_name: str,
    result: object,
    ground_truth: np.ndarray,
) -> ExperimentLog:
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
        predicted=np.asarray(getattr(result, "predicted_concept_matrix"), dtype=np.int32),
        accuracy_curve=np.asarray(getattr(result, "accuracy_matrix"), dtype=np.float64),
        total_bytes=total_bytes,
        method_name=method_name,
    )


def _build_methods(
    dataset,
    exp_cfg: ExperimentConfig,
    *,
    federation_every: int,
    fpt_lr: float,
    fpt_epochs: int,
    fpt_mode: str,
    n_clusters: int,
    concept_discovery: str = "ot",
) -> dict[str, Callable[[], object]]:
    """Build method dict for the misspecified-C experiment.

    Only includes cluster-aware baselines (which receive *n_clusters*),
    plus cluster-count-free references (FPT, CFL, Oracle, FedAvg).
    """
    fpt_kwargs = {
        "auto_scale": fpt_mode == "auto",
        "similarity_calibration": fpt_mode in {
            "calibrated", "hybrid", "hybrid-proto", "hybrid-proto-early",
            "hybrid-proto-firstagg", "hybrid-proto-subagg", "update-ot",
            "labelwise", "hybrid-labelwise",
        },
        "model_signature_weight": (
            0.55 if fpt_mode in {
                "hybrid", "hybrid-labelwise", "hybrid-proto",
                "hybrid-proto-early", "hybrid-proto-firstagg",
                "hybrid-proto-subagg",
            } else 0.0
        ),
        "model_signature_dim": 8,
        "update_ot_weight": 0.15 if fpt_mode == "update-ot" else 0.0,
        "update_ot_dim": 4,
        "labelwise_proto_weight": (
            0.35 if fpt_mode == "labelwise"
            else 0.25 if fpt_mode == "hybrid-labelwise"
            else 0.0
        ),
        "labelwise_proto_dim": 4,
        "prototype_alignment_mix": 0.25 if fpt_mode in {
            "hybrid-proto", "hybrid-proto-early",
            "hybrid-proto-firstagg", "hybrid-proto-subagg",
        } else 0.0,
        "prototype_alignment_early_rounds": 1 if fpt_mode == "hybrid-proto-early" else 0,
        "prototype_alignment_early_mix": 0.4 if fpt_mode == "hybrid-proto-early" else 0.0,
        "prototype_prealign_early_rounds": 1 if fpt_mode == "hybrid-proto-firstagg" else 0,
        "prototype_prealign_early_mix": 0.3 if fpt_mode == "hybrid-proto-firstagg" else 0.0,
        "prototype_subgroup_early_rounds": 1 if fpt_mode == "hybrid-proto-subagg" else 0,
        "prototype_subgroup_early_mix": 0.3 if fpt_mode == "hybrid-proto-subagg" else 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }
    fpt_name = {
        "base": "FedProTrack-linear-split",
        "auto": "FedProTrack-linear-auto",
        "calibrated": "FedProTrack-linear-calibrated",
        "hybrid": "FedProTrack-linear-hybrid",
        "update-ot": "FedProTrack-linear-update-ot",
        "labelwise": "FedProTrack-linear-labelwise",
        "hybrid-labelwise": "FedProTrack-linear-hybrid-labelwise",
        "hybrid-proto": "FedProTrack-linear-hybrid-proto",
        "hybrid-proto-early": "FedProTrack-linear-hybrid-proto-early",
        "hybrid-proto-firstagg": "FedProTrack-linear-hybrid-proto-firstagg",
        "hybrid-proto-subagg": "FedProTrack-linear-hybrid-proto-subagg",
    }[fpt_mode]
    return {
        # Cluster-count-free references.
        fpt_name: lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
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
            ),
            federation_every=federation_every,
            detector_name="ADWIN",
            seed=int(dataset.config.seed),
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            soft_aggregation=True,
            blend_alpha=0.0,
            concept_discovery=concept_discovery,
            **fpt_kwargs,
        ).run(dataset),
        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=fpt_lr, n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
        "Oracle": lambda: run_oracle_baseline(
            exp_cfg, dataset=dataset, lr=fpt_lr, n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
        "CFL": lambda: run_cfl_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        # Cluster-aware baselines — receive (possibly wrong) n_clusters.
        "IFCA": lambda: run_ifca_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "FeSEM": lambda: run_fesem_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "FedRC": lambda: run_fedrc_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "FedEM": lambda: run_fedem_full(
            dataset, federation_every=federation_every,
            n_components=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "FLUX-prior": lambda: run_flux_prior_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Misspecified cluster-count experiment on CIFAR-100",
    )
    parser.add_argument("--results-dir", default="tmp/misspecified_c")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--rho", type=float, default=3.0,
                        help="Recurrence period (true C = max(2, round(T/rho)))")
    parser.add_argument("--n-clusters", type=int, required=True,
                        help="Cluster count given to cluster-aware baselines (may differ from true C)")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.9)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--samples-per-coarse-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--fpt-lr", type=float, default=0.05)
    parser.add_argument("--fpt-epochs", type=int, default=5)
    parser.add_argument(
        "--fpt-mode",
        choices=[
            "base", "auto", "calibrated", "hybrid", "update-ot",
            "labelwise", "hybrid-labelwise", "hybrid-proto",
            "hybrid-proto-early", "hybrid-proto-firstagg",
            "hybrid-proto-subagg",
        ],
        default="base",
    )
    parser.add_argument("--label-split", default="disjoint",
                        choices=["none", "shared", "disjoint", "overlap"])
    parser.add_argument("--concept-discovery", choices=["gibbs", "ot"], default="ot",
                        help="Concept discovery method for FPT Phase A")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    true_c = max(2, round(args.T / args.rho))
    print(f"Config: T={args.T}, rho={args.rho} -> true C={true_c}, "
          f"given n_clusters={args.n_clusters}, label_split={args.label_split}, "
          f"seed={args.seed}", flush=True)

    dataset_cfg = CIFAR100RecurrenceConfig(
        K=args.K,
        T=args.T,
        n_samples=args.n_samples,
        rho=args.rho,
        alpha=args.alpha,
        delta=args.delta,
        n_features=args.n_features,
        samples_per_coarse_class=args.samples_per_coarse_class,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
        feature_seed=args.feature_seed,
        seed=args.seed,
        label_split=args.label_split,
        dirichlet_alpha=args.dirichlet_alpha,
    )

    print("Preparing CIFAR-100 feature cache...", flush=True)
    prepare_cifar100_recurrence_feature_cache(dataset_cfg)
    dataset = generate_cifar100_recurrence_dataset(dataset_cfg)
    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=args.federation_every,
    )

    methods = _build_methods(
        dataset,
        exp_cfg,
        federation_every=args.federation_every,
        fpt_lr=args.fpt_lr,
        fpt_epochs=args.fpt_epochs,
        fpt_mode=args.fpt_mode,
        n_clusters=args.n_clusters,
        concept_discovery=args.concept_discovery,
    )

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    fpt_active_concepts: int | None = None

    for method_name in methods:
        print(f"Running {method_name} (n_clusters={args.n_clusters})...", flush=True)
        t0 = time.time()
        try:
            result = methods[method_name]()

            if method_name.startswith("FedProTrack"):
                fpt_active_concepts = getattr(result, "active_concepts", None)

            log = _make_log(method_name, result, dataset.concept_matrix)
            metrics = compute_all_metrics(
                log,
                identity_capable=identity_metrics_valid(
                    "FedProTrack" if method_name.startswith("FedProTrack") else method_name
                ),
            )
            rows.append(
                {
                    "method": method_name,
                    "canonical_method": canonical_method_name(
                        "FedProTrack" if method_name.startswith("FedProTrack") else method_name
                    ),
                    "status": "ok",
                    "true_c": true_c,
                    "given_n_clusters": args.n_clusters,
                    "seed": args.seed,
                    "final_accuracy": metrics.final_accuracy,
                    "accuracy_auc": metrics.accuracy_auc,
                    "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                    "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
                    "assignment_entropy": metrics.assignment_entropy,
                    "total_bytes": float(getattr(result, "total_bytes", 0.0) or 0.0),
                    "wall_clock_s": time.time() - t0,
                    "fpt_active_concepts": fpt_active_concepts if method_name.startswith("FedProTrack") else None,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "method": method_name,
                    "status": "failed",
                    "true_c": true_c,
                    "given_n_clusters": args.n_clusters,
                    "seed": args.seed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "wall_clock_s": time.time() - t0,
                    "traceback": traceback.format_exc(),
                }
            )

    rows.sort(
        key=lambda row: (
            row["final_accuracy"] is None,
            -(float(row["final_accuracy"]) if row["final_accuracy"] is not None else -1.0),
        )
    )

    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "misspecified_c",
                "dataset_config": dataset_cfg.__dict__,
                "true_c": true_c,
                "given_n_clusters": args.n_clusters,
                "seed": args.seed,
                "federation_every": args.federation_every,
                "fpt_active_concepts": fpt_active_concepts,
                "rows": rows,
                "failures": failures,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    fieldnames = [
        "method", "canonical_method", "status", "true_c", "given_n_clusters",
        "seed", "final_accuracy", "accuracy_auc", "concept_re_id_accuracy",
        "wrong_memory_reuse_rate", "assignment_entropy",
        "total_bytes", "wall_clock_s", "fpt_active_concepts",
    ]
    with open(results_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(results_dir / "failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print(f"\n=== Misspecified-C Summary (true={true_c}, given={args.n_clusters}) ===",
          flush=True)
    for row in rows:
        acc = float(row["final_accuracy"]) if row["final_accuracy"] is not None else -1
        print(f"  {row['method']:30s} final={acc:.4f}", flush=True)
    if fpt_active_concepts is not None:
        print(f"  FPT discovered C={fpt_active_concepts} (true C={true_c})", flush=True)
    print(f"  Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
