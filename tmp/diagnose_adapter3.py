from __future__ import annotations
"""Diagnostic 3: The core issue — adapter convergence speed vs drift frequency."""

import sys
sys.path.insert(0, r"E:\fedprotrack\.claude\worktrees\elegant-poitras")

import numpy as np
import torch
from fedprotrack.models.torch_feature_adapter import TorchFeatureAdapterClassifier
from fedprotrack.models.torch_linear import TorchLinearClassifier

np.random.seed(42)

n_features = 64
n_classes = 20
n_train = 100  # half of 200 samples (train split)

# Create class-conditional Gaussian data
means = np.random.randn(n_classes, n_features).astype(np.float32) * 0.3
def make_data(n, classes_active):
    y = np.random.choice(classes_active, n).astype(np.int64)
    X = means[y] + np.random.randn(n, n_features).astype(np.float32) * 0.5
    return X, y

# Simulate: how many fit() calls needed to reach decent accuracy?
print("=== Convergence speed: single client, repeated fit() calls ===")
print("  (Each fit = 5 epochs on 100 samples, lr=0.05)")
classes = list(range(5))  # concept = 5 active classes

for model_type in ["linear", "adapter"]:
    if model_type == "linear":
        m = TorchLinearClassifier(n_features, n_classes, lr=0.05, n_epochs=5, seed=42, device=torch.device("cpu"))
    else:
        m = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42, device=torch.device("cpu"))

    for step in range(20):
        X_train, y_train = make_data(n_train, classes)
        X_test, y_test = make_data(200, classes)

        if model_type == "linear":
            m.fit(X_train, y_train)
        else:
            m.fit(X_train, y_train, slot_id=0)

        acc = np.mean(m.predict(X_test) == y_test)
        if step < 5 or step % 5 == 4:
            print(f"  {model_type:8s} step={step:2d}: acc={acc:.3f}")

# Now simulate with drift every ~6 steps and federation every 2
print("\n=== CIFAR-100-like scenario: drift + federation ===")
print("  K=4, T=20, federation_every=2, concept changes at t=0,7,14")

# 3 concept phases, each with different active classes
concept_schedule = [
    (0, 6, [0,1,2,3,4]),
    (7, 13, [5,6,7,8,9]),
    (14, 19, [10,11,12,13,14]),
]

for model_type in ["linear", "adapter"]:
    total_correct = 0
    total_samples = 0

    # 4 clients, all same concept for simplicity
    models = []
    for k in range(4):
        if model_type == "linear":
            models.append(TorchLinearClassifier(n_features, n_classes, lr=0.05, n_epochs=5, seed=42+k, device=torch.device("cpu")))
        else:
            models.append(TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42+k, device=torch.device("cpu")))

    for t in range(20):
        # Find active classes for this timestep
        active = None
        for t_start, t_end, classes in concept_schedule:
            if t_start <= t <= t_end:
                active = classes
                break
        if active is None:
            active = list(range(15, 20))

        step_accs = []
        for k in range(4):
            X, y = make_data(200, active)
            X_test, y_test = X[:100], y[:100]
            X_train, y_train = X[100:], y[100:]

            # Predict first
            preds = models[k].predict(X_test)
            acc = np.mean(preds == y_test)
            step_accs.append(acc)

            # Train
            if model_type == "linear":
                models[k].fit(X_train, y_train)
            else:
                models[k].fit(X_train, y_train, slot_id=0)

        # Federate every 2 steps (concept-aware = all same concept)
        if (t + 1) % 2 == 0:
            all_params = [m.get_params() for m in models]
            avg = {}
            for key in all_params[0]:
                avg[key] = np.mean([p[key] for p in all_params], axis=0)
            for m in models:
                m.set_params(avg)

        mean_acc = np.mean(step_accs)
        total_correct += sum(step_accs) * 100
        total_samples += 4 * 100

        marker = ""
        if t in [0, 7, 14]:
            marker = " <<< CONCEPT DRIFT"
        if (t + 1) % 2 == 0:
            marker += " [fed]"
        print(f"  {model_type:8s} t={t:2d}: mean_acc={mean_acc:.3f}{marker}")

    overall = total_correct / total_samples
    print(f"  {model_type:8s} OVERALL: {overall:.3f}\n")
