from __future__ import annotations

"""Run a small CIFAR-100 recurrence smoke comparison across baselines.

This entrypoint is intended for fast sanity checks rather than paper-grade
reporting. By default it de-duplicates alias methods such as ``FeSEM`` when
they currently map to the same implementation as another baseline.
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
) -> dict[str, Callable[[], object]]:
    fpt_kwargs = {
        "auto_scale": fpt_mode == "auto",
        "similarity_calibration": fpt_mode == "calibrated",
        "model_signature_weight": (
            0.55 if fpt_mode in {"hybrid", "hybrid-labelwise", "hybrid-proto", "hybrid-proto-early", "hybrid-proto-firstagg", "hybrid-proto-subagg"} else 0.0
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
        "prototype_alignment_mix": 0.25 if fpt_mode in {"hybrid-proto", "hybrid-proto-early", "hybrid-proto-firstagg", "hybrid-proto-subagg"} else 0.0,
        "prototype_alignment_early_rounds": 1 if fpt_mode == "hybrid-proto-early" else 0,
        "prototype_alignment_early_mix": 0.4 if fpt_mode == "hybrid-proto-early" else 0.0,
        "prototype_prealign_early_rounds": 1 if fpt_mode == "hybrid-proto-firstagg" else 0,
        "prototype_prealign_early_mix": 0.3 if fpt_mode == "hybrid-proto-firstagg" else 0.0,
        "prototype_subgroup_early_rounds": 1 if fpt_mode == "hybrid-proto-subagg" else 0,
        "prototype_subgroup_early_mix": 0.3 if fpt_mode == "hybrid-proto-subagg" else 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }
    if fpt_mode in {"hybrid", "hybrid-proto", "hybrid-proto-early", "hybrid-proto-firstagg", "hybrid-proto-subagg"}:
        fpt_kwargs["similarity_calibration"] = True
    if fpt_mode == "update-ot":
        fpt_kwargs["similarity_calibration"] = True
    if fpt_mode == "labelwise":
        fpt_kwargs["similarity_calibration"] = True
    if fpt_mode == "hybrid-labelwise":
        fpt_kwargs["similarity_calibration"] = True
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
            **fpt_kwargs,
        ).run(dataset),
        "LocalOnly": lambda: run_local_only(exp_cfg, dataset=dataset),
        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=fpt_lr, n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
        "FedAvg-FPTTrain": lambda: run_fedavg_baseline(
            exp_cfg,
            dataset=dataset,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
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
        "FedEM": lambda: run_fedem_full(
            dataset, federation_every=federation_every,
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
            lr=fpt_lr, n_epochs=fpt_epochs, model_type=model_type,
        ),
        "FedRC": lambda: run_fedrc_full(
            dataset, federation_every=federation_every,
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
            dataset, federation_every=federation_every, n_clusters=4,
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
    parser = argparse.ArgumentParser(description="CIFAR-100 all-baselines smoke comparison")
    parser.add_argument("--results-dir", default="tmp/cifar100_all_baselines_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--rho", type=float, default=2.0)
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
        choices=["base", "auto", "calibrated", "hybrid", "update-ot", "labelwise", "hybrid-labelwise", "hybrid-proto", "hybrid-proto-early", "hybrid-proto-firstagg", "hybrid-proto-subagg"],
        default="base",
    )
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    parser.add_argument("--model-type", choices=["linear", "small_cnn"], default="linear")
    parser.add_argument("--include-aliases", action="store_true")
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=None,
        help="Dirichlet concentration for non-IID label distribution "
             "(e.g. 0.01, 0.1, 0.5, 1.0). None = balanced (default).",
    )
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
        model_type=args.model_type,
    )
    method_names = list(methods.keys())
    if not args.include_aliases:
        method_names = dedupe_method_names(method_names)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for method_name in method_names:
        print(f"Running {method_name}...", flush=True)
        t0 = time.time()
        try:
            result = methods[method_name]()
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
                    "final_accuracy": metrics.final_accuracy,
                    "accuracy_auc": metrics.accuracy_auc,
                    "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                    "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
                    "assignment_entropy": metrics.assignment_entropy,
                    "total_bytes": float(getattr(result, "total_bytes", 0.0) or 0.0),
                    "wall_clock_s": time.time() - t0,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "method": method_name,
                    "status": "failed",
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
                "dataset_config": dataset_cfg.__dict__,
                "federation_every": args.federation_every,
                "rows": rows,
                "failures": failures,
                "include_aliases": args.include_aliases,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(results_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "canonical_method",
                "status",
                "final_accuracy",
                "accuracy_auc",
                "concept_re_id_accuracy",
                "wrong_memory_reuse_rate",
                "assignment_entropy",
                "total_bytes",
                "wall_clock_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(results_dir / "failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    print("\n=== Summary ===", flush=True)
    for row in rows:
        print(
            f"{row['method']:22s} final={float(row['final_accuracy']):.4f} "
            f"auc={float(row['accuracy_auc']):.4f} bytes={float(row['total_bytes']):.0f}",
            flush=True,
        )
    print(f"Failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
