from __future__ import annotations

"""Omega calibration experiment: compare FPT-legacy vs FPT-calibrated on CIFAR-100.

This is the CRITICAL experiment from the auto-review loop Round 2.
Tests whether calibrated omega fixes the re-ID bottleneck and closes the accuracy gap.

Runs: FPT-legacy (omega=1.0), FPT-calibrated (auto-omega), FPT-omega-10, FPT-omega-20,
      FedAvg, CFL, Oracle — on CIFAR-100 disjoint, single seed for quick validation.
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import identity_metrics_valid
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
        predicted=np.asarray(
            getattr(result, "predicted_concept_matrix"), dtype=np.int32
        ),
        accuracy_curve=np.asarray(
            getattr(result, "accuracy_matrix"), dtype=np.float64
        ),
        total_bytes=total_bytes,
        method_name=method_name,
    )


def _build_fpt(dataset, *, omega: float, federation_every: int,
               lr: float, n_epochs: int, label: str):
    """Build a FedProTrack runner with specified omega."""
    return (label, lambda: FedProTrackRunner(
        config=TwoPhaseConfig(
            omega=omega,
            kappa=0.7,
            novelty_threshold=0.25,
            loss_novelty_threshold=0.02,
            sticky_dampening=1.0,
            sticky_posterior_gate=0.35,
            merge_threshold=0.80,
            min_count=5.0,
            max_concepts=6,
            merge_every=2,
            shrink_every=6,
        ),
        federation_every=federation_every,
        detector_name="ADWIN",
        seed=int(dataset.config.seed),
        lr=lr,
        n_epochs=n_epochs,
        soft_aggregation=True,
        blend_alpha=0.0,
        similarity_calibration=True,
    ).run(dataset))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Omega calibration experiment on CIFAR-100"
    )
    parser.add_argument("--results-dir", default="tmp/omega_calibration")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
    )
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--rho", type=float, default=7.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.85)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--samples-per-coarse-class", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--fpt-lr", type=float, default=0.05)
    parser.add_argument("--fpt-epochs", type=int, default=5)
    parser.add_argument("--label-split", default="disjoint")
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        print(
            f"\n{'='*70}\n"
            f"OMEGA CALIBRATION EXPERIMENT: seed={seed}\n"
            f"K={args.K}, T={args.T}, label_split={args.label_split}\n"
            f"{'='*70}",
            flush=True,
        )

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
            seed=seed,
            label_split=args.label_split,
            min_group_size=args.min_group_size,
        )

        print("Preparing feature cache ...", flush=True)
        prepare_cifar100_recurrence_feature_cache(dataset_cfg)

        print("Generating dataset ...", flush=True)
        dataset = generate_cifar100_recurrence_dataset(dataset_cfg)
        gt = dataset.concept_matrix

        exp_cfg = ExperimentConfig(
            generator_config=dataset.config,
            federation_every=args.federation_every,
        )

        fed = args.federation_every
        lr = args.fpt_lr
        ep = args.fpt_epochs

        # Build method list: key FPT variants + baselines
        methods: dict[str, object] = {}

        # Common TwoPhaseConfig kwargs for FPT variants
        _fpt_cfg_kw = dict(
            kappa=0.7,
            novelty_threshold=0.25,
            loss_novelty_threshold=0.02,
            sticky_dampening=1.0,
            sticky_posterior_gate=0.35,
            merge_threshold=0.80,
            min_count=5.0,
            max_concepts=6,
            merge_every=2,
            shrink_every=6,
        )

        # FPT-omega-1.0 (legacy baseline)
        methods["FPT-omega-1.0"] = lambda: FedProTrackRunner(
            config=TwoPhaseConfig(omega=1.0, **_fpt_cfg_kw),
            federation_every=fed,
            detector_name="ADWIN",
            seed=seed,
            lr=lr,
            n_epochs=ep,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        ).run(dataset)

        # FPT-omega-5.0 (best legacy omega)
        methods["FPT-omega-5.0"] = lambda: FedProTrackRunner(
            config=TwoPhaseConfig(omega=5.0, **_fpt_cfg_kw),
            federation_every=fed,
            detector_name="ADWIN",
            seed=seed,
            lr=lr,
            n_epochs=ep,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        ).run(dataset)

        # FPT-trust (trust-weighted centroid estimation, omega=2.0)
        # Tighter novelty/spawn settings to reduce over-spawning
        _trust_cfg_kw = {**_fpt_cfg_kw, "novelty_threshold": 0.15, "max_concepts": 5}
        methods["FPT-trust"] = lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                enable_trust_estimation=True,
                trust_buffer_size=5,
                trust_decay=0.7,
                trust_promotion_threshold=2,
                **_trust_cfg_kw,
            ),
            federation_every=fed,
            detector_name="ADWIN",
            seed=seed,
            lr=lr,
            n_epochs=ep,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
            enable_trust_estimation=True,
            trust_buffer_size=5,
            trust_decay=0.7,
            trust_promotion_threshold=2,
        ).run(dataset)

        # Baselines
        methods["FedAvg"] = lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=ep, seed=seed,
        )
        methods["CFL"] = lambda: run_cfl_full(
            dataset, federation_every=fed,
        )
        methods["Oracle"] = lambda: run_oracle_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=ep, seed=seed,
        )

        # Run all methods
        for method_name, method_fn in methods.items():
            print(f"\n--- {method_name} ---", flush=True)
            t0 = time.time()
            try:
                result = method_fn()
                elapsed = time.time() - t0
                log = _make_log(method_name, result, gt)
                # FPT-omega-* and FPT-trust-* variants are identity-capable
                is_fpt = method_name.startswith(("FPT-omega-", "FPT-trust"))
                id_capable = is_fpt or identity_metrics_valid(method_name)

                metrics_obj = compute_all_metrics(
                    log, identity_capable=id_capable,
                )
                metrics = metrics_obj.to_dict()

                fa = metrics.get("final_accuracy") or 0.0
                ma = metrics.get("accuracy_auc") or 0.0
                ri = metrics.get("concept_re_id_accuracy") or 0.0

                row = {
                    "seed": seed,
                    "method": method_name,
                    "final_acc": round(float(fa), 4),
                    "mean_acc": round(float(ma), 4),
                    "re_id": round(float(ri), 4),
                    "total_bytes": int(log.total_bytes or 0),
                    "elapsed_s": round(elapsed, 1),
                    "trust_accept_rate": None,
                    "trust_mean_trust": None,
                }

                if not id_capable:
                    row["re_id"] = "N/A"

                # Extract trust diagnostics for FPT-trust variants
                diags = getattr(result, "phase_a_round_diagnostics", None)
                if diags and method_name.startswith("FPT-trust"):
                    accept_rates = [
                        d["trust_accept_rate"]
                        for d in diags
                        if d.get("trust_accept_rate") is not None
                    ]
                    mean_trusts = [
                        d["trust_mean_trust"]
                        for d in diags
                        if d.get("trust_mean_trust") is not None
                    ]
                    row["trust_accept_rate"] = (
                        round(float(np.mean(accept_rates)), 4)
                        if accept_rates else None
                    )
                    row["trust_mean_trust"] = (
                        round(float(np.mean(mean_trusts)), 4)
                        if mean_trusts else None
                    )

                print(
                    f"  final_acc={row['final_acc']}, mean_acc={row['mean_acc']}, "
                    f"re_id={row['re_id']}, bytes={row['total_bytes']}, "
                    f"time={row['elapsed_s']}s",
                    end="",
                    flush=True,
                )
                if "trust_accept_rate" in row:
                    print(
                        f", trust_accept={row['trust_accept_rate']}, "
                        f"trust_mean={row['trust_mean_trust']}",
                        end="",
                    )
                print(flush=True)
                all_rows.append(row)

            except Exception:
                traceback.print_exc()
                all_rows.append({
                    "seed": seed,
                    "method": method_name,
                    "final_acc": "ERROR",
                    "mean_acc": "ERROR",
                    "re_id": "ERROR",
                    "total_bytes": "ERROR",
                    "elapsed_s": round(time.time() - t0, 1),
                })

    # Save results
    csv_path = results_dir / "omega_sweep_results.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved: {csv_path}")

    json_path = results_dir / "omega_sweep_results.json"
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"JSON saved: {json_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("OMEGA CALIBRATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'FinalAcc':>10} {'MeanAcc':>10} {'Re-ID':>10} {'Bytes':>12}")
    print("-" * 65)
    for row in sorted(all_rows, key=lambda r: -float(r.get("final_acc", 0))
                      if isinstance(r.get("final_acc"), (int, float)) else -999):
        fa = row.get("final_acc", "?")
        ma = row.get("mean_acc", "?")
        ri = row.get("re_id", "?")
        tb = row.get("total_bytes", "?")
        print(f"{row['method']:<20} {fa:>10} {ma:>10} {ri:>10} {tb:>12}")


if __name__ == "__main__":
    main()
