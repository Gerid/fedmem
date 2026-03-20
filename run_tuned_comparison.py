from __future__ import annotations

"""Compare tuned FedProTrack (max_concepts=4) vs baselines on disjoint."""

import time
import traceback
import numpy as np

from fedprotrack.baselines.runners import run_cfl_full, run_ifca_full
from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
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


def main():
    seeds = [42, 43, 44, 45, 46]
    label_split = "disjoint"
    fed_every = 2

    all_rows = []

    for seed in seeds:
        print(f"\n{'='*60}\nseed={seed}\n{'='*60}", flush=True)
        cfg = CIFAR100RecurrenceConfig(
            K=4, T=12, n_samples=200, rho=3.0, alpha=0.75, delta=0.9,
            n_features=64, samples_per_coarse_class=30, batch_size=128,
            n_workers=0, data_root=".cifar100_cache",
            feature_cache_dir=".feature_cache", feature_seed=2718,
            seed=seed, label_split=label_split,
        )
        prepare_cifar100_recurrence_feature_cache(cfg)
        ds = generate_cifar100_recurrence_dataset(cfg)
        n_true = int(ds.concept_matrix.max()) + 1
        exp_cfg = ExperimentConfig(generator_config=ds.config, federation_every=fed_every)

        methods = {
            "FPT-default(max7)": lambda: FedProTrackRunner(
                config=TwoPhaseConfig(
                    omega=2.0, kappa=0.7, novelty_threshold=0.25,
                    loss_novelty_threshold=0.15, sticky_dampening=1.5,
                    sticky_posterior_gate=0.35, merge_threshold=0.85,
                    min_count=5.0, max_concepts=max(6, n_true + 3),
                    merge_every=2, shrink_every=6,
                ),
                federation_every=fed_every, detector_name="ADWIN",
                seed=seed, lr=0.05, n_epochs=5,
                soft_aggregation=True, blend_alpha=0.0,
            ).run(ds),
            "FPT-tuned(max4)": lambda: FedProTrackRunner(
                config=TwoPhaseConfig(
                    omega=2.0, kappa=0.7, novelty_threshold=0.25,
                    loss_novelty_threshold=0.15, sticky_dampening=1.5,
                    sticky_posterior_gate=0.35, merge_threshold=0.85,
                    min_count=5.0, max_concepts=4,
                    merge_every=2, shrink_every=6,
                ),
                federation_every=fed_every, detector_name="ADWIN",
                seed=seed, lr=0.05, n_epochs=5,
                soft_aggregation=True, blend_alpha=0.0,
            ).run(ds),
            "FPT-tuned(max5,merge50)": lambda: FedProTrackRunner(
                config=TwoPhaseConfig(
                    omega=2.0, kappa=0.7, novelty_threshold=0.25,
                    loss_novelty_threshold=0.35, sticky_dampening=2.5,
                    sticky_posterior_gate=0.35, merge_threshold=0.50,
                    min_count=5.0, max_concepts=5,
                    merge_every=2, shrink_every=6,
                ),
                federation_every=fed_every, detector_name="ADWIN",
                seed=seed, lr=0.05, n_epochs=5,
                soft_aggregation=True, blend_alpha=0.0,
            ).run(ds),
            "CFL": lambda: run_cfl_full(ds, federation_every=fed_every),
            "IFCA": lambda: run_ifca_full(ds, federation_every=fed_every, n_clusters=4),
            "FedAvg": lambda: run_fedavg_baseline(exp_cfg, dataset=ds),
            "Oracle": lambda: run_oracle_baseline(exp_cfg, dataset=ds),
        }

        for name, fn in methods.items():
            print(f"  {name}...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = fn()
                log = _make_log(name, result, ds.concept_matrix)
                m = compute_all_metrics(log, identity_capable=identity_metrics_valid(
                    "FedProTrack" if name.startswith("FPT") else name
                ))
                fa = getattr(m, "final_accuracy", None) or 0.0
                reid = getattr(m, "concept_re_id_accuracy", None) or 0.0
                elapsed = time.time() - t0
                all_rows.append({"seed": seed, "method": name, "acc": round(fa, 4), "reid": round(reid, 4)})
                print(f"acc={fa:.3f}  reid={reid:.3f}  ({elapsed:.1f}s)", flush=True)
            except Exception:
                print("FAILED", flush=True)
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("SUMMARY: Mean across 5 seeds on disjoint label split")
    print("=" * 70)
    print(f"{'Method':<25} {'Acc':>8} {'ReID':>8}")
    print("-" * 45)
    method_names = list(dict.fromkeys(r["method"] for r in all_rows))
    for method in method_names:
        accs = [r["acc"] for r in all_rows if r["method"] == method]
        reids = [r["reid"] for r in all_rows if r["method"] == method]
        print(f"{method:<25} {np.mean(accs):>8.4f} {np.mean(reids):>8.4f}")


if __name__ == "__main__":
    main()
