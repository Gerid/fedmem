from __future__ import annotations
"""Diagnostic 2: test adapter vs linear on structured CIFAR-like data,
and simulate the federation+aggregation loop to find the failure mode."""

import sys
sys.path.insert(0, r"E:\fedprotrack\.claude\worktrees\elegant-poitras")

import numpy as np
import torch
from fedprotrack.models.torch_feature_adapter import TorchFeatureAdapterClassifier
from fedprotrack.models.torch_linear import TorchLinearClassifier

np.random.seed(42)
torch.manual_seed(42)

n_features = 64
n_classes = 20
n_train = 200  # per client per timestep

# Create structured data: class-conditional Gaussian
means = np.random.randn(n_classes, n_features).astype(np.float32) * 0.5
def make_data(n, classes_active=None):
    if classes_active is None:
        classes_active = list(range(n_classes))
    y = np.random.choice(classes_active, n).astype(np.int64)
    X = means[y] + np.random.randn(n, n_features).astype(np.float32) * 0.3
    return X, y

# Test 1: single client, no federation
print("=== Test 1: Single-client accuracy (no federation) ===")
X_train, y_train = make_data(n_train)
X_test, y_test = make_data(200)

for lr in [0.01, 0.05, 0.1]:
    for epochs in [1, 5, 10]:
        linear = TorchLinearClassifier(n_features, n_classes, lr=lr, n_epochs=epochs, seed=42, device=torch.device("cpu"))
        adapter = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=lr, n_epochs=epochs, seed=42, device=torch.device("cpu"))

        linear.fit(X_train, y_train)
        adapter.fit(X_train, y_train)

        la = np.mean(linear.predict(X_test) == y_test)
        aa = np.mean(adapter.predict(X_test) == y_test)
        print(f"  lr={lr}, epochs={epochs}: linear={la:.3f}, adapter={aa:.3f}, delta={aa-la:+.3f}")

# Test 2: simulate federated rounds (the real scenario)
print("\n=== Test 2: Federated simulation (4 clients, 10 rounds) ===")
K = 4
T = 10

# Different concept for each client (simplified)
client_concepts = [list(range(0, 5)), list(range(5, 10)), list(range(10, 15)), list(range(15, 20))]

for model_type in ["linear", "adapter"]:
    models = []
    for k in range(K):
        if model_type == "linear":
            m = TorchLinearClassifier(n_features, n_classes, lr=0.05, n_epochs=5, seed=42+k, device=torch.device("cpu"))
        else:
            m = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42+k, device=torch.device("cpu"))
        models.append(m)

    for t in range(T):
        accs = []
        for k in range(K):
            X_k, y_k = make_data(n_train, client_concepts[k])
            mid = len(X_k) // 2
            X_test_k, y_test_k = X_k[:mid], y_k[:mid]
            X_train_k, y_train_k = X_k[mid:], y_k[mid:]

            # Predict
            preds = models[k].predict(X_test_k)
            acc = np.mean(preds == y_test_k)
            accs.append(acc)

            # Train
            if model_type == "linear":
                models[k].fit(X_train_k, y_train_k)
            else:
                models[k].fit(X_train_k, y_train_k, slot_id=0)

        # Federation: FedAvg all clients (simulating concept-unaware aggregation)
        if (t + 1) % 2 == 0:
            all_params = [m.get_params() for m in models]
            # Average
            avg = {}
            for key in all_params[0]:
                avg[key] = np.mean([p[key] for p in all_params], axis=0)
            for m in models:
                m.set_params(avg)

        mean_acc = np.mean(accs)
        if t == 0 or t == T-1 or (t+1) % 2 == 0:
            print(f"  {model_type:8s} t={t}: mean_acc={mean_acc:.3f} {'[federated]' if (t+1)%2==0 else ''}")

# Test 3: concept-aware aggregation (the FedProTrack way)
print("\n=== Test 3: Concept-aware aggregation (correct grouping) ===")
# Now each client keeps its own concept, only same-concept clients aggregate
for model_type in ["linear", "adapter"]:
    models = []
    for k in range(K):
        if model_type == "linear":
            m = TorchLinearClassifier(n_features, n_classes, lr=0.05, n_epochs=5, seed=42+k, device=torch.device("cpu"))
        else:
            m = TorchFeatureAdapterClassifier(n_features, n_classes, hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42+k, device=torch.device("cpu"))
        models.append(m)

    for t in range(T):
        accs = []
        for k in range(K):
            X_k, y_k = make_data(n_train, client_concepts[k])
            mid = len(X_k) // 2
            X_test_k, y_test_k = X_k[:mid], y_k[:mid]
            X_train_k, y_train_k = X_k[mid:], y_k[mid:]

            preds = models[k].predict(X_test_k)
            acc = np.mean(preds == y_test_k)
            accs.append(acc)

            if model_type == "linear":
                models[k].fit(X_train_k, y_train_k)
            else:
                models[k].fit(X_train_k, y_train_k, slot_id=0)

        # No aggregation since each client has a unique concept
        mean_acc = np.mean(accs)
        if t == 0 or t == T-1:
            print(f"  {model_type:8s} t={t}: mean_acc={mean_acc:.3f}")
