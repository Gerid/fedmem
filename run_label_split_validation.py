from __future__ import annotations

"""Validate that concept-specific label distributions make concept tracking
accuracy-relevant on CIFAR-100.

Runs FedProTrack, CFL, IFCA, FedAvg, and Oracle on two dataset variants:
  1. label_split="none"   — all concepts share 20 coarse classes (style-only)
  2. label_split="disjoint" — each concept gets 5 non-overlapping classes

If the hypothesis is correct, Oracle and FedProTrack should dramatically
outperform CFL/FedAvg on the disjoint split, while being roughly equal on
the style-only split.
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full, run_ifca_full
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


def _build_methods(dataset, exp_cfg, *, federation_every, fpt_lr, fpt_epochs):
    return {
        "FedProTrack": lambda: FedProTrackRunner(
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
        ).run(dataset),
        "CFL": lambda: run_cfl_full(
            dataset, federation_every=federation_every
        ),
        "IFCA": lambda: run_ifca_full(
            dataset, federation_every=federation_every, n_clusters=4
        ),
        "FedAvg": lambda: run_fedavg_baseline(exp_cfg, dataset=dataset),
        "Oracle": lambda: run_oracle_baseline(exp_cfg, dataset=dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label-split validation: style-only vs disjoint"
    )
    parser.add_argument("--results-dir", default="tmp/label_split_validation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.9)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument("--samples-per-coarse-class", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--fpt-lr", type=float, default=0.05)
    parser.add_argument("--fpt-epochs", type=int, default=5)
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    splits = ["none", "disjoint"]
    all_rows: list[dict[str, object]] = []

    for label_split in splits:
        for seed in args.seeds:
            print(
                f"\n{'='*60}\n"
                f"label_split={label_split}, seed={seed}\n"
                f"{'='*60}",
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
                label_split=label_split,
            )

            print("Preparing feature cache...", flush=True)
            prepare_cifar100_recurrence_feature_cache(dataset_cfg)
            dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

            n_concepts_actual = int(dataset.concept_matrix.max()) + 1
            print(
                f"  Dataset: K={args.K}, T={args.T}, "
                f"n_concepts={n_concepts_actual}, "
                f"label_split={label_split}",
                flush=True,
            )

            # Print label distribution per concept for verification.
            for c_id in range(n_concepts_actual):
                cells = list(zip(*np.where(dataset.concept_matrix == c_id)))
                if cells:
                    k, t = cells[0]
                    _, y = dataset.data[(k, t)]
                    unique_labels = np.unique(y)
                    print(
                        f"  Concept {c_id}: "
                        f"{len(unique_labels)} classes, "
                        f"labels={unique_labels[:8]}{'...' if len(unique_labels) > 8 else ''}",
                        flush=True,
                    )

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
            )

            for method_name, run_fn in methods.items():
                print(f"  Running {method_name}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    result = run_fn()
                    log = _make_log(method_name, result, dataset.concept_matrix)
                    metrics = compute_all_metrics(
                        log,
                        identity_capable=identity_metrics_valid(method_name),
                    )
                    elapsed = time.time() - t0
                    fa = getattr(metrics, "final_accuracy", None) or 0.0
                    reid = getattr(metrics, "concept_re_id_accuracy", None) or 0.0
                    ent = getattr(metrics, "assignment_entropy", None) or 0.0
                    row = {
                        "label_split": label_split,
                        "seed": seed,
                        "method": method_name,
                        "final_acc": round(fa, 4),
                        "re_id": round(reid, 4),
                        "entropy": round(ent, 4),
                        "elapsed_s": round(elapsed, 1),
                    }
                    all_rows.append(row)
                    print(
                        f"acc={row['final_acc']:.3f}  "
                        f"re_id={row['re_id']:.3f}  "
                        f"({elapsed:.1f}s)",
                        flush=True,
                    )
                except Exception:
                    print(f"FAILED", flush=True)
                    traceback.print_exc()

    # Write CSV
    csv_path = results_dir / "label_split_comparison.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved to {csv_path}", flush=True)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Mean final_acc across seeds")
    print("=" * 70)
    print(f"{'Method':<20} {'style-only (none)':<20} {'disjoint':<20} {'delta':<10}")
    print("-" * 70)

    method_names = list(dict.fromkeys(r["method"] for r in all_rows))
    for method in method_names:
        accs_none = [
            r["final_acc"]
            for r in all_rows
            if r["method"] == method and r["label_split"] == "none"
        ]
        accs_disj = [
            r["final_acc"]
            for r in all_rows
            if r["method"] == method and r["label_split"] == "disjoint"
        ]
        mean_none = np.mean(accs_none) if accs_none else float("nan")
        mean_disj = np.mean(accs_disj) if accs_disj else float("nan")
        delta_val = mean_disj - mean_none
        print(
            f"{method:<20} {mean_none:<20.4f} {mean_disj:<20.4f} {delta_val:+.4f}"
        )

    # Print re-ID summary for identity-capable methods
    print("\n" + "=" * 70)
    print("SUMMARY: Mean re_id across seeds (identity-capable only)")
    print("=" * 70)
    print(f"{'Method':<20} {'style-only (none)':<20} {'disjoint':<20} {'delta':<10}")
    print("-" * 70)

    for method in method_names:
        if not identity_metrics_valid(method):
            continue
        reids_none = [
            r["re_id"]
            for r in all_rows
            if r["method"] == method and r["label_split"] == "none"
        ]
        reids_disj = [
            r["re_id"]
            for r in all_rows
            if r["method"] == method and r["label_split"] == "disjoint"
        ]
        mean_none = np.mean(reids_none) if reids_none else float("nan")
        mean_disj = np.mean(reids_disj) if reids_disj else float("nan")
        delta_val = mean_disj - mean_none
        print(
            f"{method:<20} {mean_none:<20.4f} {mean_disj:<20.4f} {delta_val:+.4f}"
        )

    # Save summary JSON
    summary = {
        "label_splits": splits,
        "seeds": args.seeds,
        "K": args.K,
        "T": args.T,
        "n_samples": args.n_samples,
        "rows": all_rows,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON saved to {results_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
