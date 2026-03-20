from __future__ import annotations

"""Quick validation: adapter vs linear after zero-init fix.

Runs seed=42, T=10, K=4, n_samples=400 to verify adapter at least matches linear.
Also tests lower lr=0.01 for adapter.
"""

import time
from pathlib import Path

import numpy as np

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

SEED = 42
K = 4
T = 10
N_SAMPLES = 400
N_FEATURES = 64
SAMPLES_PER_COARSE_CLASS = 30
FEDERATION_EVERY = 2


def _build_fpt_config(dataset) -> TwoPhaseConfig:
    return TwoPhaseConfig(
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
    )


def _common_fpt_kwargs() -> dict:
    return {
        "auto_scale": False,
        "update_ot_weight": 0.0,
        "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0,
        "labelwise_proto_dim": 4,
        "prototype_alignment_early_rounds": 0,
        "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0,
        "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0,
        "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }


def run_variant(name: str, dataset, lr: float, model_type: str) -> dict:
    cfg = _build_fpt_config(dataset)
    common = _common_fpt_kwargs()
    t0 = time.time()
    result = FedProTrackRunner(
        config=cfg,
        federation_every=FEDERATION_EVERY,
        detector_name="ADWIN",
        seed=SEED,
        lr=lr,
        n_epochs=5,
        soft_aggregation=True,
        blend_alpha=0.0,
        model_type=model_type,
        hidden_dim=64,
        adapter_dim=16,
        similarity_calibration=False,
        model_signature_weight=0.0,
        model_signature_dim=8,
        prototype_alignment_mix=0.0,
        **common,
    ).run(dataset)
    log = result.to_experiment_log()
    metrics = compute_all_metrics(log, identity_capable=True)
    elapsed = time.time() - t0
    return {
        "name": name,
        "final_acc": metrics.final_accuracy,
        "re_id": metrics.concept_re_id_accuracy,
        "auc": metrics.accuracy_auc,
        "bytes": float(result.total_bytes),
        "elapsed": elapsed,
    }


def main() -> None:
    print("Preparing CIFAR-100 feature cache...", flush=True)
    cache_cfg = CIFAR100RecurrenceConfig(
        K=K, T=T, n_samples=N_SAMPLES, seed=SEED,
        n_features=N_FEATURES,
        samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS,
    )
    prepare_cifar100_recurrence_feature_cache(cache_cfg)

    dataset_cfg = CIFAR100RecurrenceConfig(
        K=K, T=T, n_samples=N_SAMPLES,
        rho=2.0, alpha=0.75, delta=0.9,
        n_features=N_FEATURES,
        samples_per_coarse_class=SAMPLES_PER_COARSE_CLASS,
        batch_size=128, n_workers=0, seed=SEED,
    )
    dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

    variants = [
        ("FPT-linear (lr=0.05)", 0.05, "linear"),
        ("FPT-adapter (lr=0.05)", 0.05, "feature_adapter"),
        ("FPT-adapter (lr=0.01)", 0.01, "feature_adapter"),
    ]

    results = []
    for name, lr, mtype in variants:
        print(f"\nRunning: {name}...", flush=True)
        row = run_variant(name, dataset, lr, mtype)
        results.append(row)
        print(
            f"  -> acc={row['final_acc']:.4f}  re-ID={row['re_id']:.4f}  "
            f"bytes={row['bytes']:.0f}  ({row['elapsed']:.1f}s)",
            flush=True,
        )

    print("\n=== Validation Summary ===", flush=True)
    print(f"{'Variant':30s}  {'FinalAcc':>10s}  {'Re-ID':>10s}  {'Bytes':>8s}", flush=True)
    print("-" * 65, flush=True)
    for r in results:
        reid_str = f"{r['re_id']:.4f}" if r["re_id"] is not None else "N/A"
        print(
            f"{r['name']:30s}  {r['final_acc']:10.4f}  {reid_str:>10s}  {r['bytes']:8.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
