from __future__ import annotations
# Quick analysis of adapter vs linear parameter counts and data regime

n_features = 64
n_classes = 20
hidden_dim = 64
adapter_dim = 16

# Linear model: just coef + intercept
linear_params = n_features * n_classes + n_classes
print(f"Linear model params: {linear_params}")

# Adapter model (1 slot):
trunk = n_features * hidden_dim + hidden_dim  # 4160
adapter = hidden_dim * adapter_dim + adapter_dim + adapter_dim * hidden_dim + hidden_dim  # 2128
head = hidden_dim * n_classes + n_classes  # 1300
adapter_total = trunk + adapter + head
print(f"Adapter model params (1 slot): {adapter_total}")
print(f"  trunk: {trunk}, adapter: {adapter}, head: {head}")
print(f"Ratio: {adapter_total / linear_params:.1f}x more params")

# Bytes: 4 bytes per float32 param
print(f"\nLinear bytes per upload: {linear_params * 4}")
print(f"Adapter bytes per upload: {adapter_total * 4}")

# Data regime
# n_samples=400 means samples_per_coarse_class=30 * coarse_classes_per_concept
# The runner splits data: X[:mid] for test, X[mid:] for train
# So training samples = n_samples / 2 = 200 per (client, timestep)
print(f"\nTraining samples per (client, step): ~200")
print(f"Params-to-samples ratio (linear): {linear_params / 200:.1f}")
print(f"Params-to-samples ratio (adapter): {adapter_total / 200:.1f}")

# With lr=0.05, n_epochs=5 on SGD:
# Linear: 1300 params, 200 samples, 5 epochs = manageable
# Adapter: 7588 params, 200 samples, 5 epochs = severely overparameterized
# BUT: the optimizer includes trunk params for EVERY slot!
# Line 78: params = list(self._trunk.parameters()) + list(adapter.parameters()) + list(head.parameters())
# So trunk gradients get accumulated from multiple slots if multi-slot training

# Key insight: lr=0.05 with SGD on a 7588-param model with only 200 samples
# and 5 full epochs is likely causing severe overfitting or gradient explosion

# The adapter has a residual connection: x + up(relu(down(x)))
# With random init, up*down is not near zero, so the residual adds significant noise
# at initialization. This is different from standard adapter init where up is near-zero.
print(f"\nAdapter init issue: up.weight is NOT initialized to near-zero")
print("Standard adapter practice: init up.weight ~ 0 so residual starts as identity")
print("PyTorch default: Kaiming uniform init -> non-trivial initial adapter output")
