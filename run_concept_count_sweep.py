from __future__ import annotations

"""Concept-count sensitivity sweep on CIFAR-100 (disjoint labels).

Varies C ∈ {2, 4, 6, 8} via rho, runs all 26 baselines + FPT.
Cluster-aware baselines receive oracle n_clusters = true C.
Records FPT's eigengap-discovered concept count for verification.

Usage (local):
    python run_concept_count_sweep.py --seed 42 --rho 3.0

RunPod submission (4 rho × 3 seeds = 12 jobs):
    for rho in 6.0 3.0 2.0 1.5; do
      python runpod/submit_experiment.py --script run_concept_count_sweep.py \
        --seeds 42 43 44 --rho $rho \
        --out-file runpod_results_csweep_rho${rho}.json
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
    run_adaptive_fedavg_full,
    run_apfl_full,
    run_atp_full,
    run_cfl_full,
    run_compressed_fedavg_full,
    run_ditto_full,
    run_fedccfa_full,
    run_fedccfa_impl_full,
    run_feddrift_full,
    run_fedem_full,
    run_fedgwc_full,
    run_fedproto_full,
    run_fedprox_full,
    run_fedrc_full,
    run_fesem_full,
    run_flash_full,
    run_flux_full,
    run_flux_prior_full,
    run_hcfl_full,
    run_ifca_full,
    run_pfedme_full,
    run_scaffold_full,
    run_tracked_summary_full,
)
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_local_only,
    run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import (
    canonical_method_name,
    dedupe_method_names,
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
    model_type: str = "linear",
    n_clusters: int = 4,
    concept_discovery: str = "ot",
) -> dict[str, Callable[[], object]]:
    """Build method dict.  Cluster-aware baselines use *n_clusters*."""
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
        "LocalOnly": lambda: run_local_only(exp_cfg, dataset=dataset),
        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=fpt_lr, n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
        "FedAvg-FPTTrain": lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=fpt_lr, n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
        "Oracle": lambda: run_oracle_baseline(
            exp_cfg, dataset=dataset, lr=fpt_lr, n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
        "FedProto": lambda: run_fedproto_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "pFedMe": lambda: run_pfedme_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "APFL": lambda: run_apfl_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        # Cluster-aware baselines — use oracle n_clusters.
        "FedEM": lambda: run_fedem_full(
            dataset, federation_every=federation_every,
            n_components=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FedCCFA": lambda: run_fedccfa_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "CFL": lambda: run_cfl_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FeSEM": lambda: run_fesem_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FedRC": lambda: run_fedrc_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "TrackedSummary": lambda: run_tracked_summary_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "Flash": lambda: run_flash_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FedDrift": lambda: run_feddrift_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "IFCA": lambda: run_ifca_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "ATP": lambda: run_atp_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FLUX": lambda: run_flux_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FLUX-prior": lambda: run_flux_prior_full(
            dataset, federation_every=federation_every,
            n_clusters=n_clusters,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "CompressedFedAvg": lambda: run_compressed_fedavg_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FedProx": lambda: run_fedprox_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "FedCCFA-Impl": lambda: run_fedccfa_impl_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "Ditto": lambda: run_ditto_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "SCAFFOLD": lambda: run_scaffold_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "Adaptive-FedAvg": lambda: run_adaptive_fedavg_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "HCFL": lambda: run_hcfl_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
        "FedGWC": lambda: run_fedgwc_full(
            dataset, federation_every=federation_every,
            lr=fpt_lr, n_epochs=fpt_epochs,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concept-count sensitivity sweep on CIFAR-100 (disjoint labels)",
    )
    parser.add_argument("--results-dir", default="tmp/concept_count_sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--rho", type=float, default=3.0,
                        help="Recurrence period (controls C: C=max(2, round(T/rho)))")
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
                        choices=["none", "shared", "disjoint", "overlap"],
                        help="How concepts differ in label distribution")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    parser.add_argument("--model-type", choices=["linear", "small_cnn"], default="linear")
    parser.add_argument("--concept-discovery", choices=["gibbs", "ot"], default="ot",
                        help="Concept discovery method for FPT Phase A")
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # Derive true concept count from config.
    n_concepts = max(2, round(args.T / args.rho))
    print(f"Config: T={args.T}, rho={args.rho} -> true C={n_concepts}, "
          f"label_split={args.label_split}, seed={args.seed}", flush=True)

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
        model_type=args.model_type,
        n_clusters=n_concepts,
        concept_discovery=args.concept_discovery,
    )
    method_names = list(methods.keys())
    method_names = dedupe_method_names(method_names)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    fpt_active_concepts: int | None = None

    for method_name in method_names:
        print(f"Running {method_name}...", flush=True)
        t0 = time.time()
        try:
            result = methods[method_name]()

            # Capture FPT's eigengap-discovered concept count.
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
                    "n_concepts": n_concepts,
                    "rho": args.rho,
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
                    "n_concepts": n_concepts,
                    "rho": args.rho,
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
                "experiment": "concept_count_sweep",
                "dataset_config": dataset_cfg.__dict__,
                "n_concepts": n_concepts,
                "rho": args.rho,
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
        "method", "canonical_method", "status", "n_concepts", "rho", "seed",
        "final_accuracy", "accuracy_auc", "concept_re_id_accuracy",
        "wrong_memory_reuse_rate", "assignment_entropy",
        "total_bytes", "wall_clock_s", "fpt_active_concepts",
    ]
    with open(results_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    with open(results_dir / "failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print(f"\n=== C={n_concepts} (rho={args.rho}) Summary ===", flush=True)
    for row in rows:
        acc = float(row["final_accuracy"]) if row["final_accuracy"] is not None else -1
        auc = float(row["accuracy_auc"]) if row["accuracy_auc"] is not None else -1
        print(f"  {row['method']:30s} final={acc:.4f}  auc={auc:.4f}", flush=True)
    if fpt_active_concepts is not None:
        print(f"  FPT discovered C={fpt_active_concepts} (true C={n_concepts})", flush=True)
    print(f"  Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
