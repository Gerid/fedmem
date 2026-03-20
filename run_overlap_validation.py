from __future__ import annotations

"""Validate FedProTrack on the overlap label-split variant.

Each of 4 concepts gets 10/20 coarse classes with 50% overlap between
neighbors.  This creates a regime where concept identity matters for
accuracy but clustering is ambiguous due to partial label overlap.
"""

import argparse
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


def _build_methods(dataset, exp_cfg, *, federation_every, fpt_lr, fpt_epochs):
    return {
        "FedProTrack": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0, kappa=0.7, novelty_threshold=0.25,
                loss_novelty_threshold=0.15, sticky_dampening=1.5,
                sticky_posterior_gate=0.35, merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, int(dataset.concept_matrix.max()) + 3),
                merge_every=2, shrink_every=6,
            ),
            federation_every=federation_every, detector_name="ADWIN",
            seed=int(dataset.config.seed), lr=fpt_lr, n_epochs=fpt_epochs,
            soft_aggregation=True, blend_alpha=0.0,
        ).run(dataset),
        "CFL": lambda: run_cfl_full(dataset, federation_every=federation_every),
        "IFCA": lambda: run_ifca_full(dataset, federation_every=federation_every, n_clusters=4),
        "FedAvg": lambda: run_fedavg_baseline(exp_cfg, dataset=dataset),
        "Oracle": lambda: run_oracle_baseline(exp_cfg, dataset=dataset),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--rho", type=float, default=3.0)
    p.add_argument("--n-classes-per-concept", type=int, default=10)
    p.add_argument("--federation-every", type=int, default=2)
    p.add_argument("--fpt-lr", type=float, default=0.05)
    p.add_argument("--fpt-epochs", type=int, default=5)
    args = p.parse_args()

    splits = ["none", "overlap", "disjoint"]
    all_rows = []

    for label_split in splits:
        for seed in args.seeds:
            print(f"\n{'='*60}\nlabel_split={label_split}, seed={seed}\n{'='*60}", flush=True)
            kw = dict(
                K=args.K, T=args.T, n_samples=args.n_samples, rho=args.rho,
                alpha=0.75, delta=0.9, n_features=64,
                samples_per_coarse_class=30, batch_size=128, n_workers=0,
                data_root=".cifar100_cache", feature_cache_dir=".feature_cache",
                feature_seed=2718, seed=seed, label_split=label_split,
            )
            if label_split == "overlap":
                kw["n_classes_per_concept"] = args.n_classes_per_concept
            cfg = CIFAR100RecurrenceConfig(**kw)
            prepare_cifar100_recurrence_feature_cache(cfg)
            ds = generate_cifar100_recurrence_dataset(cfg)
            n_c = int(ds.concept_matrix.max()) + 1
            print(f"  n_concepts={n_c}, label_split={label_split}", flush=True)
            exp_cfg = ExperimentConfig(generator_config=ds.config, federation_every=args.federation_every)
            methods = _build_methods(ds, exp_cfg, federation_every=args.federation_every, fpt_lr=args.fpt_lr, fpt_epochs=args.fpt_epochs)
            for name, fn in methods.items():
                print(f"  {name}...", end=" ", flush=True)
                t0 = time.time()
                try:
                    res = fn()
                    log = _make_log(name, res, ds.concept_matrix)
                    m = compute_all_metrics(log, identity_capable=identity_metrics_valid(name))
                    fa = getattr(m, "final_accuracy", None) or 0.0
                    reid = getattr(m, "concept_re_id_accuracy", None) or 0.0
                    elapsed = time.time() - t0
                    all_rows.append({"split": label_split, "seed": seed, "method": name, "acc": round(fa, 4), "reid": round(reid, 4), "t": round(elapsed, 1)})
                    print(f"acc={fa:.3f}  reid={reid:.3f}  ({elapsed:.1f}s)", flush=True)
                except Exception:
                    print("FAILED", flush=True)
                    traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY: Mean final_acc across seeds")
    print("=" * 80)
    print(f"{'Method':<15} {'none':<12} {'overlap':<12} {'disjoint':<12}")
    print("-" * 80)
    method_names = list(dict.fromkeys(r["method"] for r in all_rows))
    for method in method_names:
        vals = {}
        for split in splits:
            accs = [r["acc"] for r in all_rows if r["method"] == method and r["split"] == split]
            vals[split] = np.mean(accs) if accs else float("nan")
        print(f"{method:<15} {vals['none']:<12.4f} {vals['overlap']:<12.4f} {vals['disjoint']:<12.4f}")

    print("\nSUMMARY: Mean re_id across seeds")
    print("-" * 80)
    print(f"{'Method':<15} {'none':<12} {'overlap':<12} {'disjoint':<12}")
    print("-" * 80)
    for method in method_names:
        if not identity_metrics_valid(method):
            continue
        vals = {}
        for split in splits:
            reids = [r["reid"] for r in all_rows if r["method"] == method and r["split"] == split]
            vals[split] = np.mean(reids) if reids else float("nan")
        print(f"{method:<15} {vals['none']:<12.4f} {vals['overlap']:<12.4f} {vals['disjoint']:<12.4f}")


if __name__ == "__main__":
    main()
