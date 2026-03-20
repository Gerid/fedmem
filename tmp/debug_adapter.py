from __future__ import annotations

"""Debug adapter training: check what happens at each stage."""

import numpy as np
import torch

from fedprotrack.models.torch_feature_adapter import TorchFeatureAdapterClassifier
from fedprotrack.models.torch_linear import TorchLinearClassifier
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

# Get the number of classes
n_classes = max(int(v) for _, y in dataset.data.values() for v in np.unique(y)) + 1
print(f"n_classes = {n_classes}")
print(f"n_features = {N_FEATURES}")

# Get first client's data
X, y = dataset.data[(0, 0)]
mid = len(X) // 2
X_train, y_train = X[mid:], y[mid:]
X_val, y_val = X[:mid], y[:mid]
print(f"X_train.shape = {X_train.shape}, y_train unique = {np.unique(y_train)}")

# Test linear model
linear = TorchLinearClassifier(n_features=N_FEATURES, n_classes=n_classes, lr=0.05, n_epochs=5, seed=SEED)
linear.fit(X_train, y_train)
preds_linear = linear.predict(X_val)
acc_linear = float(np.mean(preds_linear == y_val))
print(f"\nLinear acc after fit: {acc_linear:.4f}")
print(f"Linear params: coef shape={linear.get_params()['coef'].shape}, "
      f"intercept shape={linear.get_params()['intercept'].shape}")
print(f"Linear param sizes: coef={linear.get_params()['coef'].size}, intercept={linear.get_params()['intercept'].size}")

# Test adapter model
adapter = TorchFeatureAdapterClassifier(
    n_features=N_FEATURES, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16,
    lr=0.05, n_epochs=5, seed=SEED,
)
# Check initial predictions before any training
preds_pre = adapter.predict(X_val)
acc_pre = float(np.mean(preds_pre == y_val))
print(f"\nAdapter acc BEFORE fit (unfitted, zeros): {acc_pre:.4f}")

adapter.fit(X_train, y_train)
preds_adapter = adapter.predict(X_val)
acc_adapter = float(np.mean(preds_adapter == y_val))
print(f"Adapter acc after fit: {acc_adapter:.4f}")

# Check adapter params
params = adapter.get_params()
print(f"\nAdapter param keys: {list(params.keys())}")
for k, v in params.items():
    print(f"  {k}: shape={v.shape}, norm={np.linalg.norm(v):.4f}, mean={v.mean():.6f}")

# Also test: does the adapter params round-trip correctly through aggregation?
# Simulate what happens: get_params -> aggregate (just identity) -> set_params
saved = adapter.get_params()
adapter2 = TorchFeatureAdapterClassifier(
    n_features=N_FEATURES, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16,
    lr=0.05, n_epochs=5, seed=SEED + 1,
)
adapter2.set_params(saved)
preds_rt = adapter2.predict(X_val)
acc_rt = float(np.mean(preds_rt == y_val))
print(f"\nAdapter acc after round-trip set_params: {acc_rt:.4f}")

# Check: what does the adapter do WITHOUT federation (just repeated fit)?
adapter3 = TorchFeatureAdapterClassifier(
    n_features=N_FEATURES, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16,
    lr=0.05, n_epochs=5, seed=SEED,
)
for t in range(5):
    X_t, y_t = dataset.data[(0, t)]
    mid = len(X_t) // 2
    adapter3.fit(X_t[mid:], y_t[mid:])
    preds = adapter3.predict(X_t[:mid])
    acc = float(np.mean(preds == y_t[:mid]))
    print(f"  Adapter standalone step {t}: acc={acc:.4f}")

# Key question: what happens when we do set_params then predict on a DIFFERENT slot?
# The get_params returns expert.0.* keys, but after Phase A routing, the concept ID may differ
print("\n--- Slot routing test ---")
adapter4 = TorchFeatureAdapterClassifier(
    n_features=N_FEATURES, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16,
    lr=0.05, n_epochs=5, seed=SEED,
)
adapter4.fit(X_train, y_train, slot_id=0)
# Now get params for slot 0
p0 = adapter4.get_params(slot_id=0)
print(f"Slot 0 param keys: {list(p0.keys())}")

# What happens if we call predict with slot_id=0?
preds_s0 = adapter4.predict(X_val, slot_id=0)
acc_s0 = float(np.mean(preds_s0 == y_val))
print(f"Predict with slot_id=0: acc={acc_s0:.4f}")

# What happens if we call predict with slot_id=1 (uninitialized)?
preds_s1 = adapter4.predict(X_val, slot_id=1)
acc_s1 = float(np.mean(preds_s1 == y_val))
print(f"Predict with slot_id=1 (fresh): acc={acc_s1:.4f}")

# Now simulate what the runner does after aggregation:
# It creates params with expert.<concept_id>.* keys and calls set_params
# If concept_id != 0, the adapter needs to handle that
concept_id = 3  # Simulate a concept ID of 3
remapped = {}
for k, v in p0.items():
    if k.startswith("shared."):
        remapped[k] = v.copy()
    elif k.startswith("expert.0."):
        new_key = k.replace("expert.0.", f"expert.{concept_id}.")
        remapped[new_key] = v.copy()
    else:
        remapped[k] = v.copy()

print(f"\nRemapped keys: {list(remapped.keys())}")
adapter5 = TorchFeatureAdapterClassifier(
    n_features=N_FEATURES, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16,
    lr=0.05, n_epochs=5, seed=SEED,
)
adapter5.set_params(remapped)
preds_c3 = adapter5.predict(X_val, slot_id=concept_id)
acc_c3 = float(np.mean(preds_c3 == y_val))
print(f"Predict with slot_id={concept_id} after set_params: acc={acc_c3:.4f}")
