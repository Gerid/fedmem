from __future__ import annotations

"""FedProTrack advantage experiment on CIFAR-100 with favorable conditions.

Designs a concept matrix with K=10 clients, C=4 concepts, T=30 time steps,
and disjoint label distributions across concepts. This scenario is engineered
to showcase FedProTrack's strengths:

  1. **Label heterogeneity**: disjoint 5-class partitions per concept make
     concept identity accuracy-critical.
  2. **Temporal horizon**: T=30 gives the Gibbs posterior enough observations
     to converge (T >= 20).
  3. **Data sufficiency**: n_samples=800 yields >= 25 samples/class/concept.
  4. **Low singleton ratio**: rho=7.5 -> 4 concepts shared across 10 clients,
     so most concepts are observed by multiple clients simultaneously.

Runs: FedProTrack (tuned), CFL, IFCA, FedAvg, Oracle.
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
    """Build the callable map for all methods under test.

    Returns
    -------
    dict[str, callable]
        Mapping from method name to a zero-argument callable that returns the
        method result.
    """
    n_concepts_true = int(dataset.concept_matrix.max()) + 1

    return {
        # ---- FedProTrack with tuned hyperparameters for this scenario ----
        "FedProTrack": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                # Tuned thresholds (from Phase 4 validation):
                novelty_threshold=0.25,
                loss_novelty_threshold=0.02,   # conservative spawning
                sticky_dampening=1.0,           # less suppression
                sticky_posterior_gate=0.35,
                merge_threshold=0.80,           # moderate merge to reduce over-spawning
                min_count=5.0,
                max_concepts=6,                 # cap at true(4) + 2 buffer
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
            # Use similarity calibration for real-data fingerprints.
            similarity_calibration=True,
        ).run(dataset),
        # ---- Baselines ----
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


def _print_concept_matrix_stats(
    concept_matrix: np.ndarray,
) -> None:
    """Print diagnostic statistics about the concept matrix."""
    K, T = concept_matrix.shape
    n_concepts = int(concept_matrix.max()) + 1
    print(f"  Concept matrix: K={K}, T={T}, n_concepts={n_concepts}")

    # Singleton ratio: fraction of (concept, t) groups with only 1 client.
    singleton_count = 0
    total_groups = 0
    for t in range(T):
        concepts_at_t = concept_matrix[:, t]
        for c in range(n_concepts):
            clients_with_c = np.sum(concepts_at_t == c)
            if clients_with_c > 0:
                total_groups += 1
                if clients_with_c == 1:
                    singleton_count += 1

    singleton_ratio = singleton_count / max(total_groups, 1)
    print(f"  Singleton ratio: {singleton_ratio:.3f} ({singleton_count}/{total_groups})")

    # Recurrence: how many times each concept appears across all clients.
    for c in range(n_concepts):
        occurrences = int(np.sum(concept_matrix == c))
        n_clients_ever = len(set(
            k for k in range(K) if np.any(concept_matrix[k] == c)
        ))
        print(f"  Concept {c}: {occurrences} cell-occurrences across {n_clients_ever} clients")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedProTrack advantage experiment on CIFAR-100"
    )
    parser.add_argument("--results-dir", default="tmp/fpt_advantage")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
    )
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=800)
    # rho=7.5 -> n_concepts = round(30/7.5) = 4
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
            f"seed={seed}, K={args.K}, T={args.T}, "
            f"label_split={args.label_split}, rho={args.rho}\n"
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

        print("Preparing feature cache...", flush=True)
        prepare_cifar100_recurrence_feature_cache(dataset_cfg)
        dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

        _print_concept_matrix_stats(dataset.concept_matrix)

        # Print label distribution per concept for verification.
        n_concepts_actual = int(dataset.concept_matrix.max()) + 1
        for c_id in range(n_concepts_actual):
            cells = list(zip(*np.where(dataset.concept_matrix == c_id)))
            if cells:
                k, t = cells[0]
                _, y = dataset.data[(k, t)]
                unique_labels = np.unique(y)
                print(
                    f"  Concept {c_id} labels: {unique_labels.tolist()}",
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

                # Lifecycle counters (FedProTrack only).
                spawned = getattr(result, "spawned_concepts", None) or 0
                active = getattr(result, "active_concepts", None) or 0

                row = {
                    "seed": seed,
                    "method": method_name,
                    "final_acc": round(float(fa), 4),
                    "re_id": round(float(reid), 4),
                    "entropy": round(float(ent), 4),
                    "auc": round(float(auc), 4),
                    "total_bytes": int(total_bytes),
                    "spawned_concepts": spawned,
                    "active_concepts": active,
                    "elapsed_s": round(elapsed, 1),
                }
                all_rows.append(row)
                print(
                    f"acc={row['final_acc']:.3f}  "
                    f"re_id={row['re_id']:.3f}  "
                    f"auc={row['auc']:.3f}  "
                    f"bytes={row['total_bytes']}  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
            except Exception:
                print("FAILED", flush=True)
                traceback.print_exc()

    # ---- Save CSV ----
    csv_path = results_dir / "fpt_advantage_results.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved to {csv_path}", flush=True)

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("SUMMARY: Mean metrics across seeds")
    print("=" * 80)
    print(
        f"{'Method':<15} {'FinalAcc':<12} {'Re-ID':<12} "
        f"{'AUC':<12} {'Entropy':<12} {'Bytes':<14} {'Spawned':<10}"
    )
    print("-" * 80)

    method_names = list(dict.fromkeys(r["method"] for r in all_rows))
    summary_rows = []
    for method in method_names:
        method_rows = [r for r in all_rows if r["method"] == method]
        accs = [r["final_acc"] for r in method_rows]
        reids = [r["re_id"] for r in method_rows]
        aucs = [r["auc"] for r in method_rows]
        ents = [r["entropy"] for r in method_rows]
        bytess = [r["total_bytes"] for r in method_rows]
        spawned = [r["spawned_concepts"] for r in method_rows]

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_reid = float(np.mean(reids))
        mean_auc = float(np.mean(aucs))
        mean_ent = float(np.mean(ents))
        mean_bytes = float(np.mean(bytess))
        mean_spawned = float(np.mean(spawned))

        print(
            f"{method:<15} "
            f"{mean_acc:.4f}+/-{std_acc:.3f} "
            f"{mean_reid:<12.4f} "
            f"{mean_auc:<12.4f} "
            f"{mean_ent:<12.4f} "
            f"{mean_bytes:<14.0f} "
            f"{mean_spawned:<10.1f}"
        )
        summary_rows.append({
            "method": method,
            "mean_final_acc": round(mean_acc, 4),
            "std_final_acc": round(std_acc, 4),
            "mean_re_id": round(mean_reid, 4),
            "mean_auc": round(mean_auc, 4),
            "mean_entropy": round(mean_ent, 4),
            "mean_total_bytes": int(mean_bytes),
            "mean_spawned": round(mean_spawned, 1),
        })

    # ---- Save summary JSON ----
    summary = {
        "experiment": "fpt_advantage",
        "label_split": args.label_split,
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
