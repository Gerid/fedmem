from __future__ import annotations
"""Diagnostic 4: Confirm the trunk averaging + slow recovery is the root cause.
Also test potential fixes: lower lr, more epochs, zero-init adapter."""

import sys
sys.path.insert(0, r"E:\fedprotrack\.claude\worktrees\elegant-poitras")

import numpy as np
import torch
import torch.nn as nn
from fedprotrack.models.torch_feature_adapter import TorchFeatureAdapterClassifier

np.random.seed(42)

n_features = 64
n_classes = 20

means = np.random.randn(n_classes, n_features).astype(np.float32) * 0.3
def make_data(n, classes_active):
    y = np.random.choice(classes_active, n).astype(np.int64)
    X = means[y] + np.random.randn(n, n_features).astype(np.float32) * 0.5
    return X, y

# Test: what if we zero-init the adapter up projection?
print("=== Fix 1: Zero-init adapter.up.weight ===")
m = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42, device=torch.device("cpu"))
with torch.no_grad():
    for slot_id, (adapter, head) in m._experts.items():
        adapter.up.weight.zero_()
        adapter.up.bias.zero_()

classes = list(range(5))
for step in range(10):
    X_train, y_train = make_data(100, classes)
    X_test, y_test = make_data(200, classes)
    m.fit(X_train, y_train, slot_id=0)
    acc = np.mean(m.predict(X_test) == y_test)
    if step < 5 or step % 5 == 4:
        print(f"  step={step}: acc={acc:.3f}")

# Test: what if we use a lower learning rate?
print("\n=== Fix 2: lr=0.01 instead of 0.05 ===")
m2 = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.01, n_epochs=5, seed=42, device=torch.device("cpu"))
for step in range(10):
    X_train, y_train = make_data(100, classes)
    X_test, y_test = make_data(200, classes)
    m2.fit(X_train, y_train, slot_id=0)
    acc = np.mean(m2.predict(X_test) == y_test)
    if step < 5 or step % 5 == 4:
        print(f"  step={step}: acc={acc:.3f}")

# Test: what if we use more epochs?
print("\n=== Fix 3: n_epochs=20 instead of 5 ===")
m3 = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=20, seed=42, device=torch.device("cpu"))
for step in range(10):
    X_train, y_train = make_data(100, classes)
    X_test, y_test = make_data(200, classes)
    m3.fit(X_train, y_train, slot_id=0)
    acc = np.mean(m3.predict(X_test) == y_test)
    if step < 5 or step % 5 == 4:
        print(f"  step={step}: acc={acc:.3f}")

# Test: lr=0.01, epochs=20
print("\n=== Fix 4: lr=0.01, n_epochs=20 ===")
m4 = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.01, n_epochs=20, seed=42, device=torch.device("cpu"))
for step in range(10):
    X_train, y_train = make_data(100, classes)
    X_test, y_test = make_data(200, classes)
    m4.fit(X_train, y_train, slot_id=0)
    acc = np.mean(m4.predict(X_test) == y_test)
    if step < 5 or step % 5 == 4:
        print(f"  step={step}: acc={acc:.3f}")

# Reference: linear model with same schedule
print("\n=== Reference: Linear model, lr=0.05, epochs=5 ===")
from fedprotrack.models.torch_linear import TorchLinearClassifier
m_ref = TorchLinearClassifier(n_features, n_classes, lr=0.05, n_epochs=5, seed=42, device=torch.device("cpu"))
for step in range(10):
    X_train, y_train = make_data(100, classes)
    X_test, y_test = make_data(200, classes)
    m_ref.fit(X_train, y_train)
    acc = np.mean(m_ref.predict(X_test) == y_test)
    if step < 5 or step % 5 == 4:
        print(f"  step={step}: acc={acc:.3f}")

# Test: freeze trunk, only train adapter + head
print("\n=== Fix 5: Freeze trunk (update_shared=False) ===")
m5 = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42, device=torch.device("cpu"))
for step in range(10):
    X_train, y_train = make_data(100, classes)
    X_test, y_test = make_data(200, classes)
    m5.fit(X_train, y_train, slot_id=0, update_shared=False)
    acc = np.mean(m5.predict(X_test) == y_test)
    if step < 5 or step % 5 == 4:
        print(f"  step={step}: acc={acc:.3f}")
