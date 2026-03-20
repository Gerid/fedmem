from __future__ import annotations

from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiment.baselines import run_oracle_baseline, run_fedavg_baseline
from fedprotrack.baselines.runners import run_cfl_full
import numpy as np

cfg = CIFAR100RecurrenceConfig(
    K=4, T=20, n_samples=400, n_features=64,
    samples_per_coarse_class=30, seed=42,
)
prepare_cifar100_recurrence_feature_cache(cfg)
ds = generate_cifar100_recurrence_dataset(cfg)
exp = ExperimentConfig(generator_config=ds.config, federation_every=2)

print(f"Unique concepts: {np.unique(ds.concept_matrix)}")
print(f"Concept matrix:\n{ds.concept_matrix}")
print()

# Oracle fixed (n_epochs=5, lr=0.05)
r = run_oracle_baseline(exp, dataset=ds, lr=0.05, n_epochs=5)
print(f"Oracle (ep=5, lr=0.05):  final={r.final_accuracy:.4f}  mean={r.mean_accuracy:.4f}")

# Oracle fixed (n_epochs=5, lr=0.1)
r2 = run_oracle_baseline(exp, dataset=ds, lr=0.1, n_epochs=5)
print(f"Oracle (ep=5, lr=0.1):   final={r2.final_accuracy:.4f}  mean={r2.mean_accuracy:.4f}")

# Oracle old behavior (n_epochs=1)
r3 = run_oracle_baseline(exp, dataset=ds, lr=0.1, n_epochs=1)
print(f"Oracle (ep=1, lr=0.1):   final={r3.final_accuracy:.4f}  mean={r3.mean_accuracy:.4f}")

# CFL
r_cfl = run_cfl_full(ds, federation_every=2)
print(f"CFL:                     final={float(np.mean(r_cfl.accuracy_matrix[:,-1])):.4f}")

# FedAvg
r_avg = run_fedavg_baseline(exp, dataset=ds)
print(f"FedAvg (ep=1):           final={r_avg.final_accuracy:.4f}")

# FedAvg with matched training
r_avg2 = run_fedavg_baseline(exp, dataset=ds, lr=0.05, n_epochs=5)
print(f"FedAvg (ep=5, lr=0.05):  final={r_avg2.final_accuracy:.4f}")
