from __future__ import annotations

"""Debug adapter WITHOUT federation: pure local training to see ceiling."""

import numpy as np

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
    K=K, T=T, n_samples=N_SAMPLES, rho=2.0, alpha=0.75, delta=0.9,
    n_features=N_FEATURES, samples_per_coarse_class=30,
    batch_size=128, n_workers=0, seed=SEED,
)
dataset = generate_cifar100_recurrence_dataset(dataset_cfg)
n_classes = max(int(v) for _, y in dataset.data.values() for v in np.unique(y)) + 1

# Simulate local-only for client 0
for name, mk_model in [
    ("linear lr=0.05 ep=5", lambda: TorchLinearClassifier(N_FEATURES, n_classes, lr=0.05, n_epochs=5, seed=SEED)),
    ("linear lr=0.05 ep=20", lambda: TorchLinearClassifier(N_FEATURES, n_classes, lr=0.05, n_epochs=20, seed=SEED)),
    ("adapter lr=0.05 ep=5", lambda: TorchFeatureAdapterClassifier(N_FEATURES, n_classes, 64, 16, lr=0.05, n_epochs=5, seed=SEED)),
    ("adapter lr=0.05 ep=20", lambda: TorchFeatureAdapterClassifier(N_FEATURES, n_classes, 64, 16, lr=0.05, n_epochs=20, seed=SEED)),
    ("adapter lr=0.05 ep=50", lambda: TorchFeatureAdapterClassifier(N_FEATURES, n_classes, 64, 16, lr=0.05, n_epochs=50, seed=SEED)),
]:
    model = mk_model()
    accs = []
    for t in range(T):
        X, y = dataset.data[(0, t)]
        mid = len(X) // 2
        model.fit(X[mid:], y[mid:])
        preds = model.predict(X[:mid])
        accs.append(float(np.mean(preds == y[:mid])))
    print(f"{name:30s}  final={accs[-1]:.4f}  mean={np.mean(accs):.4f}  per-step: {' '.join(f'{a:.3f}' for a in accs)}")
