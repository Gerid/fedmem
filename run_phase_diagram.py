from __future__ import annotations

"""Phase diagram sweep: accuracy vs label heterogeneity level.

Varies label heterogeneity across three levels to map FedProTrack's
boundary conditions:

  - label_split="shared"     (no heterogeneity -- all concepts share 20 classes)
  - label_split="overlapping" (partial -- each concept gets 10/20 classes, ~50% overlap)
  - label_split="disjoint"   (full -- each concept gets 5 non-overlapping classes)

For each level, runs FedProTrack, CFL, IFCA, FedAvg, and Oracle with 3 seeds.
Produces the data for a "boundary condition phase diagram" figure showing
where concept identity tracking becomes accuracy-critical.

Key insight: as label heterogeneity increases, methods that can correctly
identify recurring concepts (FedProTrack, IFCA, Oracle) should separate
from methods that cannot (FedAvg, CFL).
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
    """Convert a method result into a unified ExperimentLog."""
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


def _build_methods(
    dataset,
    exp_cfg: ExperimentConfig,
    *,
    federation_every: int,
    fpt_lr: float,
    fpt_epochs: int,
) -> dict[str, object]:
    """Build the callable map for all methods under test."""
    n_concepts_true = int(dataset.concept_matrix.max()) + 1

    return {
        "FedProTrack": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
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
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        ).run(dataset),
        "CFL": lambda: run_cfl_full(
            dataset, federation_every=federation_every
        ),
        "IFCA": lambda: run_ifca_full(
            dataset,
            federation_every=federation_every,
            n_clusters=n_concepts_true,
        ),
        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg,
            dataset=dataset,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
        "Oracle": lambda: run_oracle_baseline(
            exp_cfg,
            dataset=dataset,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            seed=int(dataset.config.seed),
        ),
    }


# Map from user-facing label_split names to internal config values.
_SPLIT_CONFIGS = {
    "shared": {"label_split": "none"},
    "overlapping": {"label_split": "overlap", "n_classes_per_concept": 10},
    "disjoint": {"label_split": "disjoint"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase diagram sweep: accuracy vs label heterogeneity"
    )
    parser.add_argument("--results-dir", default="tmp/phase_diagram")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
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
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["shared", "overlapping", "disjoint"]
    all_rows: list[dict[str, object]] = []

    for split_name in split_names:
        split_cfg = _SPLIT_CONFIGS[split_name]
        for seed in args.seeds:
            print(
                f"\n{'='*70}\n"
                f"label_split={split_name}, seed={seed}\n"
                f"{'='*70}",
                flush=True,
            )

            dataset_kw = dict(
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
                min_group_size=args.min_group_size,
                **split_cfg,
            )
            dataset_cfg = CIFAR100RecurrenceConfig(**dataset_kw)

            print("Preparing feature cache...", flush=True)
            prepare_cifar100_recurrence_feature_cache(dataset_cfg)
            dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

            n_concepts_actual = int(dataset.concept_matrix.max()) + 1
            print(
                f"  Dataset: K={args.K}, T={args.T}, "
                f"n_concepts={n_concepts_actual}, "
                f"label_split={split_name}",
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
                        f"labels={unique_labels[:8].tolist()}"
                        f"{'...' if len(unique_labels) > 8 else ''}",
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
                    auc = getattr(metrics, "accuracy_auc", None) or 0.0
                    total_bytes = getattr(log, "total_bytes", None) or 0.0

                    row = {
                        "label_split": split_name,
                        "seed": seed,
                        "method": method_name,
                        "final_acc": round(float(fa), 4),
                        "re_id": round(float(reid), 4),
                        "entropy": round(float(ent), 4),
                        "auc": round(float(auc), 4),
                        "total_bytes": int(total_bytes),
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
                    print("FAILED", flush=True)
                    traceback.print_exc()

    # ---- Save CSV ----
    csv_path = results_dir / "phase_diagram_results.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved to {csv_path}", flush=True)

    # ---- Print summary tables ----
    method_names = list(dict.fromkeys(r["method"] for r in all_rows))

    # Table 1: Final accuracy across heterogeneity levels.
    print("\n" + "=" * 80)
    print("PHASE DIAGRAM: Mean final_acc across seeds")
    print("=" * 80)
    print(f"{'Method':<15} {'shared':<14} {'overlapping':<14} {'disjoint':<14} {'disj-shared':<12}")
    print("-" * 80)
    summary_rows = []
    for method in method_names:
        vals = {}
        for split in split_names:
            accs = [
                r["final_acc"]
                for r in all_rows
                if r["method"] == method and r["label_split"] == split
            ]
            vals[split] = float(np.mean(accs)) if accs else float("nan")

        delta_val = vals["disjoint"] - vals["shared"]
        print(
            f"{method:<15} "
            f"{vals['shared']:<14.4f} "
            f"{vals['overlapping']:<14.4f} "
            f"{vals['disjoint']:<14.4f} "
            f"{delta_val:+.4f}"
        )
        summary_rows.append({
            "method": method,
            "shared_acc": round(vals["shared"], 4),
            "overlapping_acc": round(vals["overlapping"], 4),
            "disjoint_acc": round(vals["disjoint"], 4),
            "delta_disj_shared": round(delta_val, 4),
        })

    # Table 2: Re-ID accuracy (identity-capable methods only).
    print("\n" + "=" * 80)
    print("PHASE DIAGRAM: Mean re_id across seeds (identity-capable)")
    print("=" * 80)
    print(f"{'Method':<15} {'shared':<14} {'overlapping':<14} {'disjoint':<14}")
    print("-" * 80)
    for method in method_names:
        if not identity_metrics_valid(method):
            continue
        vals = {}
        for split in split_names:
            reids = [
                r["re_id"]
                for r in all_rows
                if r["method"] == method and r["label_split"] == split
            ]
            vals[split] = float(np.mean(reids)) if reids else float("nan")
        print(
            f"{method:<15} "
            f"{vals['shared']:<14.4f} "
            f"{vals['overlapping']:<14.4f} "
            f"{vals['disjoint']:<14.4f}"
        )

    # Table 3: Communication bytes.
    print("\n" + "=" * 80)
    print("PHASE DIAGRAM: Mean total_bytes across seeds")
    print("=" * 80)
    print(f"{'Method':<15} {'shared':<14} {'overlapping':<14} {'disjoint':<14}")
    print("-" * 80)
    for method in method_names:
        vals = {}
        for split in split_names:
            bytess = [
                r["total_bytes"]
                for r in all_rows
                if r["method"] == method and r["label_split"] == split
            ]
            vals[split] = float(np.mean(bytess)) if bytess else float("nan")
        print(
            f"{method:<15} "
            f"{vals['shared']:<14.0f} "
            f"{vals['overlapping']:<14.0f} "
            f"{vals['disjoint']:<14.0f}"
        )

    # ---- Save summary JSON ----
    summary = {
        "experiment": "phase_diagram",
        "split_names": split_names,
        "K": args.K,
        "T": args.T,
        "n_samples": args.n_samples,
        "rho": args.rho,
        "alpha": args.alpha,
        "delta": args.delta,
        "seeds": args.seeds,
        "federation_every": args.federation_every,
        "fpt_lr": args.fpt_lr,
        "fpt_epochs": args.fpt_epochs,
        "summary": summary_rows,
        "raw_rows": all_rows,
    }
    json_path = results_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON saved to {json_path}", flush=True)


if __name__ == "__main__":
    main()
