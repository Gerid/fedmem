from __future__ import annotations

"""Diagnostic: what re-ID would a simple nearest-neighbor tracker achieve?

The KNN test shows concepts are perfectly separable in raw fingerprint space.
So the Gibbs posterior's 73% re-ID must be losing information somewhere.
Let's build a trivial tracker and see what re-ID it achieves.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from fedprotrack.metrics.concept_metrics import concept_re_id_accuracy
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
K, T = gt.shape

# Build fingerprints for each (k, t)
fps = {}
for k in range(K):
    for t in range(T):
        X, y = dataset.data[(k, t)]
        fps[(k, t)] = np.concatenate([X.mean(axis=0), X.std(axis=0)])

# Simple nearest-neighbor tracker:
# At t=0, assign each client to a new concept (one per unique fingerprint cluster).
# At t>0, for each client, find the most similar known concept fingerprint.
# If similarity > threshold, assign to that concept. Otherwise, create new.

# First, collect all concept fingerprints (using ground truth to establish prototypes)
concept_prototypes = {}  # concept_id -> list of fingerprints
for k in range(K):
    for t in range(T):
        c = gt[k, t]
        if c not in concept_prototypes:
            concept_prototypes[c] = []
        concept_prototypes[c].append(fps[(k, t)])

for c in concept_prototypes:
    concept_prototypes[c] = np.mean(concept_prototypes[c], axis=0)

# Now do online tracking WITHOUT ground truth
# Memory bank: concept_id -> prototype fingerprint (running mean)
memory = {}
predicted = np.zeros((K, T), dtype=np.int32)
next_concept_id = 0

for t in range(T):
    for k in range(K):
        fp = fps[(k, t)]

        if not memory:
            # First assignment
            memory[next_concept_id] = fp.copy()
            predicted[k, t] = next_concept_id
            next_concept_id += 1
            continue

        # Find most similar concept in memory
        best_sim = -1
        best_c = -1
        for c_id, proto in memory.items():
            sim = float(cosine_similarity(fp.reshape(1, -1), proto.reshape(1, -1))[0, 0])
            if sim > best_sim:
                best_sim = sim
                best_c = c_id

        if best_sim > 0.95:
            predicted[k, t] = best_c
            # Update prototype with running mean
            n = 1  # simplified
            memory[best_c] = 0.95 * memory[best_c] + 0.05 * fp
        else:
            memory[next_concept_id] = fp.copy()
            predicted[k, t] = next_concept_id
            next_concept_id += 1

reid, _, _ = concept_re_id_accuracy(gt, predicted)
n_pred_concepts = int(predicted.max()) + 1
print(f"Simple NN tracker: re-ID={reid:.3f}, predicted concepts={n_pred_concepts}")
print(f"FedProTrack Gibbs posterior: re-ID=0.723")
print(f"Improvement potential: {reid - 0.723:+.3f}")

# Also try with the "ideal" approach: KNN on all fingerprints
from sklearn.neighbors import KNeighborsClassifier

all_fps = np.array([fps[(k, t)] for k in range(K) for t in range(T)])
all_labels = np.array([gt[k, t] for k in range(K) for t in range(T)])

# Leave-one-out KNN
predicted_knn = np.zeros((K, T), dtype=np.int32)
for k in range(K):
    for t in range(T):
        # Train on all OTHER (k', t') pairs
        mask = np.ones(K * T, dtype=bool)
        idx = k * T + t
        mask[idx] = False
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(all_fps[mask], all_labels[mask])
        predicted_knn[k, t] = knn.predict(all_fps[idx:idx+1])[0]

reid_knn, _, _ = concept_re_id_accuracy(gt, predicted_knn)
print(f"\nLeave-one-out KNN: re-ID={reid_knn:.3f}")
