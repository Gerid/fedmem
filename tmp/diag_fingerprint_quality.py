from __future__ import annotations

"""Diagnostic: fingerprint quality -- can we distinguish concepts from fingerprints?"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

# Build fingerprints: mean + std of features for each (k, t)
fingerprints = []
labels = []
for k in range(cfg.K):
    for t in range(cfg.T):
        X, y = dataset.data[(k, t)]
        fp = np.concatenate([X.mean(axis=0), X.std(axis=0)])
        fingerprints.append(fp)
        labels.append(gt[k, t])

fingerprints = np.array(fingerprints)
labels = np.array(labels)

# KNN classifier on fingerprints
# Use leave-one-out style: train on first 80%, test on last 20%
n = len(labels)
split = int(0.8 * n)
idx = np.arange(n)
rng = np.random.default_rng(42)
rng.shuffle(idx)

X_train, y_train = fingerprints[idx[:split]], labels[idx[:split]]
X_test, y_test = fingerprints[idx[split:]], labels[idx[split:]]

for k_nn in [1, 3, 5, 10]:
    knn = KNeighborsClassifier(n_neighbors=k_nn)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"KNN (k={k_nn:2d}) fingerprint -> concept accuracy: {acc:.3f}")

# Also check: cosine similarity within vs across concepts
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(fingerprints)
within_sims = []
across_sims = []
for i in range(n):
    for j in range(i+1, n):
        s = sim_matrix[i, j]
        if labels[i] == labels[j]:
            within_sims.append(s)
        else:
            across_sims.append(s)

print(f"\nWithin-concept cosine similarity: {np.mean(within_sims):.4f} +/- {np.std(within_sims):.4f}")
print(f"Across-concept cosine similarity: {np.mean(across_sims):.4f} +/- {np.std(across_sims):.4f}")
print(f"Separation: {np.mean(within_sims) - np.mean(across_sims):.4f}")

# Per-concept analysis
n_concepts = int(gt.max()) + 1
for c in range(n_concepts):
    c_mask = labels == c
    c_fps = fingerprints[c_mask]
    within = cosine_similarity(c_fps).mean()
    # Similarity to other concepts
    other_fps = fingerprints[~c_mask]
    across = cosine_similarity(c_fps, other_fps).mean()
    print(f"  Concept {c}: within={within:.4f}  across={across:.4f}  gap={within-across:.4f}")
