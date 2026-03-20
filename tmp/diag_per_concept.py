from __future__ import annotations

"""Per-concept accuracy diagnostic: where does FPT lose to Oracle/CFL?"""

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.experiment.baselines import run_oracle_baseline, run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

cfg = CIFAR100RecurrenceConfig(
    K=10, T=30, n_samples=800, rho=7.5, alpha=0.5, delta=0.85,
    n_features=128, samples_per_coarse_class=120, batch_size=256,
    n_workers=0, seed=42, label_split="disjoint", min_group_size=2,
)
prepare_cifar100_recurrence_feature_cache(cfg)
dataset = generate_cifar100_recurrence_dataset(cfg)
gt = dataset.concept_matrix

exp_cfg = ExperimentConfig(generator_config=dataset.config, federation_every=2)

# Run methods
fpt_runner = FedProTrackRunner(
    config=TwoPhaseConfig(
        omega=2.0, kappa=0.7,
        novelty_threshold=0.25, loss_novelty_threshold=0.02,
        sticky_dampening=1.0, sticky_posterior_gate=0.35,
        merge_threshold=0.80, min_count=5.0, max_concepts=6,
        merge_every=2, shrink_every=6,
    ),
    federation_every=2, detector_name="ADWIN",
    seed=42, lr=0.05, n_epochs=5,
    soft_aggregation=True, blend_alpha=0.0,
    similarity_calibration=True,
)

print("Running FedProTrack...", flush=True)
fpt_result = fpt_runner.run(dataset)
print("Running Oracle...", flush=True)
oracle_result = run_oracle_baseline(exp_cfg, dataset=dataset, lr=0.05, n_epochs=5, seed=42)
print("Running CFL...", flush=True)
cfl_result = run_cfl_full(dataset, federation_every=2)
print("Running FedAvg...", flush=True)
fedavg_result = run_fedavg_baseline(exp_cfg, dataset=dataset, lr=0.05, n_epochs=5, seed=42)

# Per-concept accuracy
results = {
    "FedProTrack": fpt_result.accuracy_matrix,
    "Oracle": oracle_result.accuracy_matrix,
    "CFL": cfl_result.accuracy_matrix,
    "FedAvg": fedavg_result.accuracy_matrix,
}

n_concepts = int(gt.max()) + 1
K, T = gt.shape

print(f"\n{'Method':<15}", end="")
for c in range(n_concepts):
    print(f"  C{c:d}_acc", end="")
print("  Overall")
print("-" * (15 + 9 * (n_concepts + 1)))

for method, acc_mat in results.items():
    print(f"{method:<15}", end="")
    for c in range(n_concepts):
        mask = gt == c
        if mask.any():
            c_acc = float(acc_mat[mask].mean())
        else:
            c_acc = float("nan")
        print(f"  {c_acc:.4f}", end="")
    print(f"  {float(acc_mat.mean()):.4f}")

# Per time-step accuracy for FPT vs Oracle (last 10 steps)
print("\n\nPer-step accuracy (last 10 steps):")
print(f"{'Step':<6}", end="")
for method in results:
    print(f"  {method:<12}", end="")
print()
for t in range(max(0, T-10), T):
    print(f"t={t:<4}", end="")
    for method, acc_mat in results.items():
        step_acc = float(acc_mat[:, t].mean())
        print(f"  {step_acc:<12.4f}", end="")
    print()

# FedProTrack predicted vs true concept matrix
print("\n\nFPT predicted concept matrix (first 5 clients, last 10 steps):")
fpt_pred = fpt_result.predicted_concept_matrix
for k in range(min(5, K)):
    print(f"  client {k}: true={gt[k, -10:].tolist()}  pred={fpt_pred[k, -10:].tolist()}")
