from __future__ import annotations

"""Diagnose posterior entropy collapse v2: use exact experiment parameters."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

SEED = 42
K = 4
T = 20
N_SAMPLES = 400
N_FEATURES = 64

# Prepare dataset
cache_cfg = CIFAR100RecurrenceConfig(
    K=K, T=T, n_samples=N_SAMPLES, seed=SEED,
    n_features=N_FEATURES, samples_per_coarse_class=30,
)
prepare_cifar100_recurrence_feature_cache(cache_cfg)

dataset_cfg = CIFAR100RecurrenceConfig(
    K=K, T=T, n_samples=N_SAMPLES,
    rho=2.0, alpha=0.75, delta=0.9,
    n_features=N_FEATURES, samples_per_coarse_class=30,
    batch_size=128, n_workers=0, seed=SEED,
)
dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

# Exact config from run_h2h_lr_sweep.py
cfg = TwoPhaseConfig(
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

common = {
    "auto_scale": False,
    "similarity_calibration": False,
    "model_signature_weight": 0.0,
    "model_signature_dim": 8,
    "update_ot_weight": 0.0,
    "update_ot_dim": 4,
    "labelwise_proto_weight": 0.0,
    "labelwise_proto_dim": 4,
    "prototype_alignment_mix": 0.0,
    "prototype_alignment_early_rounds": 0,
    "prototype_alignment_early_mix": 0.0,
    "prototype_prealign_early_rounds": 0,
    "prototype_prealign_early_mix": 0.0,
    "prototype_subgroup_early_rounds": 0,
    "prototype_subgroup_early_mix": 0.0,
    "prototype_subgroup_min_clients": 3,
    "prototype_subgroup_similarity_gate": 0.8,
}


def run_and_diagnose(label: str, model_type: str, fed_every: int, lr: float, n_epochs: int):
    print(f"\n{'=' * 80}")
    print(f"{label}: model={model_type}, fed_every={fed_every}, lr={lr}, epochs={n_epochs}")
    print(f"{'=' * 80}")

    runner = FedProTrackRunner(
        config=cfg,
        federation_every=fed_every,
        detector_name="ADWIN",
        seed=SEED,
        lr=lr,
        n_epochs=n_epochs,
        soft_aggregation=True,
        blend_alpha=0.0,
        model_type=model_type,
        hidden_dim=64,
        adapter_dim=16,
        **common,
    )
    result = runner.run(dataset)

    log = result.to_experiment_log()
    metrics = compute_all_metrics(log, identity_capable=True)

    print(f"  final_accuracy:   {metrics.final_accuracy:.4f}")
    print(f"  re_id_accuracy:   {metrics.concept_re_id_accuracy:.4f}")
    print(f"  assignment_entropy: {metrics.assignment_entropy:.6f}")
    print(f"  total_bytes:      {result.total_bytes:.0f}")
    print(f"  spawned_concepts: {result.spawned_concepts}")
    print(f"  active_concepts:  {result.active_concepts}")

    # Analyze soft_assignments
    sa = result.soft_assignments
    if sa is not None:
        K_, T_, C = sa.shape
        print(f"\n  soft_assignments shape: ({K_}, {T_}, {C})")

        # Count cells with any nonzero posterior
        nonzero_cells = np.any(sa > 1e-10, axis=2)
        n_nonzero = int(nonzero_cells.sum())
        n_total = K_ * T_
        print(f"  Cells with nonzero posterior: {n_nonzero}/{n_total}")

        # Per-cell entropy
        eps = 1e-12
        p = np.clip(sa, eps, None)
        H = -np.sum(p * np.log(p), axis=-1)  # (K, T)

        # Separate fed vs non-fed
        fed_mask = nonzero_cells
        nonfed_mask = ~nonzero_cells

        H_fed = float(H[fed_mask].mean()) if fed_mask.any() else 0.0
        H_nonfed = float(H[nonfed_mask].mean()) if nonfed_mask.any() else 0.0
        H_overall = float(H.mean())

        print(f"  H(fed cells):    {H_fed:.6f} (over {int(fed_mask.sum())} cells)")
        print(f"  H(non-fed cells): {H_nonfed:.6f} (over {int(nonfed_mask.sum())} cells)")
        print(f"  H(overall):       {H_overall:.6f}")
        print(f"  Dilution formula: {int(fed_mask.sum())} * {H_fed:.4f} / {n_total} = "
              f"{int(fed_mask.sum()) * H_fed / n_total:.6f}")

    # Now test: what if we use the SAME federation_every for both?
    return metrics.assignment_entropy


# Run exact experiment configs
ent_adapter = run_and_diagnose(
    "ADAPTER (exact h2h params)",
    model_type="feature_adapter", fed_every=5, lr=0.05, n_epochs=30
)
ent_linear = run_and_diagnose(
    "LINEAR (exact h2h params)",
    model_type="linear", fed_every=2, lr=0.05, n_epochs=5
)

print(f"\n{'=' * 80}")
print(f"COMPARISON: adapter entropy={ent_adapter:.6f}, linear entropy={ent_linear:.6f}")
print(f"{'=' * 80}")

# Now the KEY test: run both with the SAME federation_every
print(f"\n\n{'#' * 80}")
print(f"CONTROL: both with federation_every=5")
print(f"{'#' * 80}")

ent_adapter_ctrl = run_and_diagnose(
    "ADAPTER (fed=5)",
    model_type="feature_adapter", fed_every=5, lr=0.05, n_epochs=30
)
ent_linear_ctrl = run_and_diagnose(
    "LINEAR (fed=5)",
    model_type="linear", fed_every=5, lr=0.05, n_epochs=5
)

print(f"\n{'=' * 80}")
print(f"CONTROL: adapter={ent_adapter_ctrl:.6f}, linear={ent_linear_ctrl:.6f}")
print(f"{'=' * 80}")

print(f"\n\n{'#' * 80}")
print(f"CONTROL 2: both with federation_every=2")
print(f"{'#' * 80}")

ent_adapter_ctrl2 = run_and_diagnose(
    "ADAPTER (fed=2)",
    model_type="feature_adapter", fed_every=2, lr=0.05, n_epochs=30
)
ent_linear_ctrl2 = run_and_diagnose(
    "LINEAR (fed=2)",
    model_type="linear", fed_every=2, lr=0.05, n_epochs=5
)

print(f"\n{'=' * 80}")
print(f"CONTROL 2: adapter={ent_adapter_ctrl2:.6f}, linear={ent_linear_ctrl2:.6f}")
print(f"{'=' * 80}")

print("\n\nDONE.")
