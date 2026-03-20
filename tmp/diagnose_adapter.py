from __future__ import annotations
"""Diagnostic: check adapter initialization, gradient magnitude, and overfitting."""

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
n_samples = 200  # training set size after split

# Generate random data (mimics CIFAR-100 features)
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = np.random.randint(0, n_classes, n_samples).astype(np.int64)

# 1. Check adapter initialization
print("=== Adapter Initialization ===")
adapter_model = TorchFeatureAdapterClassifier(
    n_features=n_features, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42,
    device=torch.device("cpu"),
)
adapter_block = adapter_model._experts[0][0]
up_w = adapter_block.up.weight.data
down_w = adapter_block.down.weight.data
print(f"  adapter.up.weight: mean={up_w.mean():.4f}, std={up_w.std():.4f}, max_abs={up_w.abs().max():.4f}")
print(f"  adapter.down.weight: mean={down_w.mean():.4f}, std={down_w.std():.4f}, max_abs={down_w.abs().max():.4f}")

# Check: what does the adapter residual do at init?
test_input = torch.randn(10, 64)
with torch.no_grad():
    trunk_out = adapter_model._trunk(torch.randn(10, n_features))
    residual = adapter_block(trunk_out) - trunk_out
    print(f"  Residual at init: mean_abs={residual.abs().mean():.4f}, max_abs={residual.abs().max():.4f}")
    print(f"  Trunk output: mean_abs={trunk_out.abs().mean():.4f}")
    print(f"  Residual/trunk ratio: {residual.abs().mean() / trunk_out.abs().mean():.4f}")

# 2. Prediction before training
print("\n=== Prediction Before Training ===")
preds_adapter = adapter_model.predict(X)
print(f"  Adapter predictions before fit: unique={np.unique(preds_adapter)}")
acc_before = np.mean(preds_adapter == y)
print(f"  Adapter accuracy before fit: {acc_before:.4f} (random={1/n_classes:.4f})")

linear_model = TorchLinearClassifier(
    n_features=n_features, n_classes=n_classes,
    lr=0.05, n_epochs=5, seed=42, device=torch.device("cpu"),
)
preds_linear = linear_model.predict(X)
print(f"  Linear predictions before fit: unique count={len(np.unique(preds_linear))}")

# 3. Train and compare
print("\n=== Training (5 epochs on 200 samples) ===")
adapter_model2 = TorchFeatureAdapterClassifier(
    n_features=n_features, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=42,
    device=torch.device("cpu"),
)
linear_model2 = TorchLinearClassifier(
    n_features=n_features, n_classes=n_classes,
    lr=0.05, n_epochs=5, seed=42, device=torch.device("cpu"),
)

adapter_model2.fit(X, y)
linear_model2.fit(X, y)

preds_a = adapter_model2.predict(X)
preds_l = linear_model2.predict(X)
train_acc_a = np.mean(preds_a == y)
train_acc_l = np.mean(preds_l == y)
print(f"  Adapter train accuracy: {train_acc_a:.4f}")
print(f"  Linear train accuracy: {train_acc_l:.4f}")

# Test on fresh data
X_test = np.random.randn(200, n_features).astype(np.float32)
y_test = np.random.randint(0, n_classes, 200).astype(np.int64)
test_acc_a = np.mean(adapter_model2.predict(X_test) == y_test)
test_acc_l = np.mean(linear_model2.predict(X_test) == y_test)
print(f"  Adapter test accuracy: {test_acc_a:.4f}")
print(f"  Linear test accuracy: {test_acc_l:.4f}")

# 4. Check what happens after set_params (aggregation round)
print("\n=== Aggregation Simulation ===")
params_before = adapter_model2.get_params()
print(f"  Param keys: {list(params_before.keys())}")

# Simulate: create another adapter, train it, then average
adapter_model3 = TorchFeatureAdapterClassifier(
    n_features=n_features, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=5, seed=43,
    device=torch.device("cpu"),
)
X2 = np.random.randn(n_samples, n_features).astype(np.float32)
y2 = np.random.randint(0, n_classes, n_samples).astype(np.int64)
adapter_model3.fit(X2, y2)
params_other = adapter_model3.get_params()

# FedAvg: average the params
avg_params = {
    k: (params_before[k] + params_other[k]) / 2.0
    for k in params_before
}
adapter_model2.set_params(avg_params)
avg_acc = np.mean(adapter_model2.predict(X) == y)
print(f"  Adapter accuracy AFTER averaging 2 clients: {avg_acc:.4f}")
print(f"  (was {train_acc_a:.4f} before averaging)")

# 5. Key issue: the slot_id mismatch during get_params/set_params
print("\n=== Slot ID in Aggregation ===")
# get_params always exports the _active_slot's expert
# If two clients have different concept IDs (slot_ids), their expert
# params have DIFFERENT keys (expert.0.* vs expert.1.*)
# FedAvg only averages params with the SAME key
# So expert params from slot 0 and slot 1 will NOT be averaged together
# BUT: trunk (shared.*) WILL be averaged across concepts
# This is correct behavior for namespaced aggregation

# However, the issue is: after set_params, the model may have expert
# weights for a slot_id it was never assigned to, and the optimizer
# gets rebuilt for ALL existing slots, including trunk
print(f"  Slot IDs before set_params: {list(adapter_model2._experts.keys())}")

# 6. Check gradient explosion potential
print("\n=== Gradient Analysis ===")
adapter_grad = TorchFeatureAdapterClassifier(
    n_features=n_features, n_classes=n_classes,
    hidden_dim=64, adapter_dim=16, lr=0.05, n_epochs=1, seed=42,
    device=torch.device("cpu"),
)
X_t = torch.tensor(X[:10], dtype=torch.float32)
y_t = torch.tensor(y[:10], dtype=torch.long)
adapter_grad._trunk.train()
a, h = adapter_grad._experts[0]
a.train(); h.train()
logits = adapter_grad._forward_slot(X_t, 0)
loss = torch.nn.functional.cross_entropy(logits, y_t)
loss.backward()

print(f"  Loss value: {loss.item():.4f}")
for name, param in adapter_grad._trunk.named_parameters():
    if param.grad is not None:
        print(f"  trunk.{name} grad: mean_abs={param.grad.abs().mean():.6f}, max_abs={param.grad.abs().max():.6f}")
for name, param in a.named_parameters():
    if param.grad is not None:
        print(f"  adapter.{name} grad: mean_abs={param.grad.abs().mean():.6f}, max_abs={param.grad.abs().max():.6f}")
for name, param in h.named_parameters():
    if param.grad is not None:
        print(f"  head.{name} grad: mean_abs={param.grad.abs().mean():.6f}, max_abs={param.grad.abs().max():.6f}")

# Step size: lr * grad magnitude
lr = 0.05
for name, param in adapter_grad._trunk.named_parameters():
    if param.grad is not None:
        step = lr * param.grad.abs().max()
        w_mag = param.data.abs().max()
        print(f"  trunk.{name}: max step/max weight = {step/w_mag:.4f}")

print("\n=== Summary ===")
print("The adapter model has 5.8x more params than linear (7588 vs 1300)")
print("Training regime: 200 samples, 5 epochs, lr=0.05")
print("Params-to-samples ratio: 37.9 (adapter) vs 6.5 (linear)")
print("The adapter residual connection starts non-zero (Kaiming init)")
print("After aggregation, all clients' TRUNK weights are averaged (shared.*)")
print("But each client's EXPERT weights go to different slots")
