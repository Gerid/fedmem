from __future__ import annotations

"""Debug: trace what params flow through get_params -> aggregation -> set_params."""

import numpy as np
from fedprotrack.models.torch_feature_adapter import TorchFeatureAdapterClassifier
from fedprotrack.federation.aggregator import (
    has_namespaced_params, split_param_namespaces, merge_param_namespaces,
)

N_FEATURES = 64
N_CLASSES = 20

# Create and train two adapters
a1 = TorchFeatureAdapterClassifier(N_FEATURES, N_CLASSES, 64, 16, lr=0.05, n_epochs=5, seed=42)
a2 = TorchFeatureAdapterClassifier(N_FEATURES, N_CLASSES, 64, 16, lr=0.05, n_epochs=43)

rng = np.random.default_rng(42)
X = rng.standard_normal((200, N_FEATURES)).astype(np.float32)
y = rng.integers(0, N_CLASSES, size=200)
a1.fit(X, y, slot_id=0)
a2.fit(X, y, slot_id=0)

# Get params
p1 = a1.get_params(slot_id=0)
p2 = a2.get_params(slot_id=0)

print("=== Param keys from get_params ===")
for k in p1:
    print(f"  {k}: shape={p1[k].shape}")

print(f"\nhas_namespaced_params(p1) = {has_namespaced_params(p1)}")

shared1, expert1, other1 = split_param_namespaces(p1)
print(f"\nShared keys: {list(shared1.keys())}")
print(f"Expert keys: {list(expert1.keys())}")
print(f"Other keys: {list(other1.keys())}")

# Simulate aggregation: average shared, average expert
from fedprotrack.federation.aggregator import FedAvgAggregator
agg = FedAvgAggregator()

shared_agg = agg.aggregate([shared1, split_param_namespaces(p2)[0]])
expert_agg = agg.aggregate([expert1, split_param_namespaces(p2)[1]])

merged = merge_param_namespaces(shared=shared_agg, expert=expert_agg, other=None)
print(f"\nMerged (aggregated) keys: {list(merged.keys())}")
for k in merged:
    print(f"  {k}: shape={merged[k].shape}")

# Apply to a fresh model
a3 = TorchFeatureAdapterClassifier(N_FEATURES, N_CLASSES, 64, 16, lr=0.05, n_epochs=5, seed=99)
a3.set_params(merged)
preds = a3.predict(X, slot_id=0)
acc = float(np.mean(preds == y))
print(f"\nAcc after set_params(aggregated): {acc:.4f}")

# Check: what if we directly average the original params?
p_avg = {k: 0.5 * p1[k] + 0.5 * p2[k] for k in p1}
a4 = TorchFeatureAdapterClassifier(N_FEATURES, N_CLASSES, 64, 16, lr=0.05, n_epochs=5, seed=99)
a4.set_params(p_avg)
preds_avg = a4.predict(X, slot_id=0)
acc_avg = float(np.mean(preds_avg == y))
print(f"Acc after set_params(direct avg): {acc_avg:.4f}")

# Check consistency
for k in merged:
    if k in p_avg:
        diff = np.abs(merged[k] - p_avg[k]).max()
        if diff > 1e-6:
            print(f"  DIFF in {k}: max_abs_diff = {diff:.6f}")
    else:
        print(f"  Key {k} in merged but not in p_avg")
for k in p_avg:
    if k not in merged:
        print(f"  Key {k} in p_avg but not in merged")
