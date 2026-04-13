from __future__ import annotations

"""Rerun specific baselines on all flagship configs.

Usage:
    python run_baseline_patch.py --seed 42 --methods FedDrift FedEM
    python run_baseline_patch.py --seed 42 --methods FedDrift FedEM --dataset fmow
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import (
    run_fedccfa_full,
    run_feddrift_full,
    run_fedem_full,
    run_fedrc_full,
)
from fedprotrack.experiment.baselines import run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import (
    canonical_method_name,
    identity_metrics_valid,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def _make_log(method_name, result, ground_truth):
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


# Flagship CIFAR-100 configs (matching paper Table 1)
CIFAR100_CONFIGS = [
    {"tag": "K20_rho17", "K": 20, "T": 100, "rho": 17.0},
    {"tag": "K20_rho25", "K": 20, "T": 100, "rho": 25.0},
    {"tag": "K20_rho33", "K": 20, "T": 100, "rho": 33.0},
    {"tag": "K40_rho25", "K": 40, "T": 100, "rho": 25.0},
    {"tag": "K40_rho33", "K": 40, "T": 100, "rho": 33.0},
]

METHOD_MAP = {
    "FedDrift": lambda ds, fe, lr, ep: run_feddrift_full(
        ds, federation_every=fe, lr=lr, n_epochs=ep,
    ),
    "FedEM": lambda ds, fe, lr, ep: run_fedem_full(
        ds, federation_every=fe, lr=lr, n_epochs=ep,
    ),
    "FedRC": lambda ds, fe, lr, ep: run_fedrc_full(
        ds, federation_every=fe, n_clusters=4, lr=lr, n_epochs=ep,
    ),
    "FedCCFA": lambda ds, fe, lr, ep: run_fedccfa_full(
        ds, federation_every=fe, lr=lr, n_epochs=ep,
    ),
}


def run_cifar100_patch(seed, methods, results_dir, args):
    rows = []
    for cfg_spec in CIFAR100_CONFIGS:
        tag = cfg_spec["tag"]
        cfg = CIFAR100RecurrenceConfig(
            K=cfg_spec["K"], T=cfg_spec["T"], rho=cfg_spec["rho"],
            n_samples=200, alpha=0.75, delta=0.85,
            n_features=128, samples_per_coarse_class=120,
            batch_size=256, n_workers=args.n_workers,
            data_root=args.data_root,
            feature_cache_dir=args.feature_cache_dir,
            feature_seed=2718, seed=seed,
            label_split="disjoint",
        )
        print(f"\n=== {tag} seed={seed} ===", flush=True)
        prepare_cifar100_recurrence_feature_cache(cfg)
        ds = generate_cifar100_recurrence_dataset(cfg)
        exp = ExperimentConfig(generator_config=ds.config, federation_every=2)

        for method_name in methods:
            fn = METHOD_MAP.get(method_name)
            if fn is None:
                print(f"  Unknown method {method_name}, skipping")
                continue
            print(f"  Running {method_name}...", flush=True)
            t0 = time.time()
            try:
                result = fn(ds, 2, 0.01, 5)
                log = _make_log(method_name, result, ds.concept_matrix)
                metrics = compute_all_metrics(
                    log, identity_capable=identity_metrics_valid(method_name),
                )
                row = {
                    "config": tag, "method": method_name, "seed": seed,
                    "status": "ok",
                    "final_accuracy": metrics.final_accuracy,
                    "accuracy_auc": metrics.accuracy_auc,
                    "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                    "total_bytes": float(getattr(result, "total_bytes", 0) or 0),
                    "wall_clock_s": time.time() - t0,
                }
                print(f"    {method_name}: final={metrics.final_accuracy:.4f} "
                      f"({time.time()-t0:.1f}s)", flush=True)
            except Exception as exc:
                row = {
                    "config": tag, "method": method_name, "seed": seed,
                    "status": "failed", "error": str(exc),
                    "wall_clock_s": time.time() - t0,
                    "traceback": traceback.format_exc(),
                }
                print(f"    {method_name}: FAILED — {exc}", flush=True)
            rows.append(row)

    return rows


def run_fmow_patch(seed, methods, results_dir, args):
    """Patch FedDrift/FedEM on fMoW."""
    try:
        from fedprotrack.real_data.fmow import (
            FMoWConfig,
            generate_fmow_dataset,
        )
    except ImportError:
        print("fMoW module not available, skipping")
        return []

    cfg = FMoWConfig(
        K=20, T=100, rho=25.0, n_samples=200,
        n_features=128, seed=seed,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
    )
    print(f"\n=== fMoW seed={seed} ===", flush=True)
    ds = generate_fmow_dataset(cfg)
    exp = ExperimentConfig(generator_config=ds.config, federation_every=2)

    rows = []
    for method_name in methods:
        fn = METHOD_MAP.get(method_name)
        if fn is None:
            continue
        print(f"  Running {method_name}...", flush=True)
        t0 = time.time()
        try:
            result = fn(ds, 2, 0.01, 5)
            log = _make_log(method_name, result, ds.concept_matrix)
            metrics = compute_all_metrics(
                log, identity_capable=identity_metrics_valid(method_name),
            )
            row = {
                "config": "fmow", "method": method_name, "seed": seed,
                "status": "ok",
                "final_accuracy": metrics.final_accuracy,
                "accuracy_auc": metrics.accuracy_auc,
                "wall_clock_s": time.time() - t0,
            }
            print(f"    {method_name}: final={metrics.final_accuracy:.4f}", flush=True)
        except Exception as exc:
            row = {
                "config": "fmow", "method": method_name, "seed": seed,
                "status": "failed", "error": str(exc),
                "wall_clock_s": time.time() - t0,
            }
            print(f"    {method_name}: FAILED — {exc}", flush=True)
        rows.append(row)
    return rows


def run_fmnist_patch(seed, methods, results_dir, args):
    """Patch FedDrift/FedEM on Fashion-MNIST."""
    try:
        from fedprotrack.real_data.fmnist_recurrence import (
            FMNISTRecurrenceConfig,
            generate_fmnist_recurrence_dataset,
            prepare_fmnist_recurrence_feature_cache,
        )
    except ImportError:
        print("FMNIST module not available, skipping")
        return []

    cfg = FMNISTRecurrenceConfig(
        K=20, T=100, rho=25.0, n_samples=200,
        n_features=128, samples_per_coarse_class=120,
        seed=seed, n_workers=args.n_workers,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
    )
    print(f"\n=== F-MNIST seed={seed} ===", flush=True)
    prepare_fmnist_recurrence_feature_cache(cfg)
    ds = generate_fmnist_recurrence_dataset(cfg)
    exp = ExperimentConfig(generator_config=ds.config, federation_every=2)

    rows = []
    for method_name in methods:
        fn = METHOD_MAP.get(method_name)
        if fn is None:
            continue
        print(f"  Running {method_name}...", flush=True)
        t0 = time.time()
        try:
            result = fn(ds, 2, 0.01, 5)
            log = _make_log(method_name, result, ds.concept_matrix)
            metrics = compute_all_metrics(
                log, identity_capable=identity_metrics_valid(method_name),
            )
            row = {
                "config": "fmnist", "method": method_name, "seed": seed,
                "status": "ok",
                "final_accuracy": metrics.final_accuracy,
                "accuracy_auc": metrics.accuracy_auc,
                "wall_clock_s": time.time() - t0,
            }
            print(f"    {method_name}: final={metrics.final_accuracy:.4f}", flush=True)
        except Exception as exc:
            row = {
                "config": "fmnist", "method": method_name, "seed": seed,
                "status": "failed", "error": str(exc),
                "wall_clock_s": time.time() - t0,
            }
        rows.append(row)
    return rows


def run_cifar10_patch(seed, methods, results_dir, args):
    """Patch FedDrift/FedEM on CIFAR-10."""
    try:
        from fedprotrack.real_data.cifar10_recurrence import (
            CIFAR10RecurrenceConfig,
            generate_cifar10_recurrence_dataset,
            prepare_cifar10_recurrence_feature_cache,
        )
    except ImportError:
        print("CIFAR-10 module not available, skipping")
        return []

    cfg = CIFAR10RecurrenceConfig(
        K=20, T=100, rho=25.0, n_samples=200,
        n_features=128, samples_per_coarse_class=120,
        seed=seed, n_workers=args.n_workers,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
    )
    print(f"\n=== CIFAR-10 seed={seed} ===", flush=True)
    prepare_cifar10_recurrence_feature_cache(cfg)
    ds = generate_cifar10_recurrence_dataset(cfg)
    exp = ExperimentConfig(generator_config=ds.config, federation_every=2)

    rows = []
    for method_name in methods:
        fn = METHOD_MAP.get(method_name)
        if fn is None:
            continue
        print(f"  Running {method_name}...", flush=True)
        t0 = time.time()
        try:
            result = fn(ds, 2, 0.01, 5)
            log = _make_log(method_name, result, ds.concept_matrix)
            metrics = compute_all_metrics(
                log, identity_capable=identity_metrics_valid(method_name),
            )
            row = {
                "config": "cifar10", "method": method_name, "seed": seed,
                "status": "ok",
                "final_accuracy": metrics.final_accuracy,
                "accuracy_auc": metrics.accuracy_auc,
                "wall_clock_s": time.time() - t0,
            }
            print(f"    {method_name}: final={metrics.final_accuracy:.4f}", flush=True)
        except Exception as exc:
            row = {
                "config": "cifar10", "method": method_name, "seed": seed,
                "status": "failed", "error": str(exc),
                "wall_clock_s": time.time() - t0,
            }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Rerun specific baselines on flagship configs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=["FedDrift", "FedEM", "FedRC", "FedCCFA"])
    parser.add_argument("--dataset", default="all",
                        choices=["all", "cifar100", "fmow", "fmnist", "cifar10"])
    parser.add_argument("--results-dir", default="tmp/baseline_patch")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--n-workers", type=int, default=0)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    if args.dataset in ("all", "cifar100"):
        all_rows.extend(run_cifar100_patch(args.seed, args.methods, results_dir, args))

    if args.dataset in ("all", "fmow"):
        all_rows.extend(run_fmow_patch(args.seed, args.methods, results_dir, args))

    if args.dataset in ("all", "fmnist"):
        all_rows.extend(run_fmnist_patch(args.seed, args.methods, results_dir, args))

    if args.dataset in ("all", "cifar10"):
        all_rows.extend(run_cifar10_patch(args.seed, args.methods, results_dir, args))

    # Save
    with open(results_dir / "results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    if all_rows:
        with open(results_dir / "results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\n=== Done: {len(all_rows)} results saved to {results_dir} ===")
    ok = sum(1 for r in all_rows if r.get("status") == "ok")
    print(f"    {ok}/{len(all_rows)} succeeded")


if __name__ == "__main__":
    main()
