from __future__ import annotations

"""Compute a quantitative B_j^2 proxy on CIFAR-100 to bridge theory and real data.

B_j^2 = ||W_j* - W_bar*||_F^2 measures how far each concept's optimal multiclass
linear head is from the global average. On CIFAR-100 with ResNet-18 features,
we estimate this by:
  1. Training per-concept one-vs-rest linear heads on ground-truth concept groups
  2. Computing the Frobenius distance between each concept head matrix and their average
  3. Plugging into the theory's crossover formula: SNR = K*n*B_j^2/(sigma^2*d)
  4. Running actual Oracle vs FedAvg to verify the prediction empirically

This gives a quantitative test: theory predicts Oracle > FedAvg iff SNR > C-1.
"""

import json
from pathlib import Path

import numpy as np

from fedprotrack.real_data.cifar100_recurrence import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
)


def _ovr_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit one-vs-rest OLS regression heads: W = (X^T X + lambda I)^{-1} X^T Y_onehot.

    Returns W of shape (d, n_classes) where each column is a class head.
    """
    X64 = X.astype(np.float64)
    y_int = y.astype(np.int64)
    d = X64.shape[1]
    classes = sorted(set(y_int.tolist()))
    n_classes = len(classes)

    # Build one-hot Y matrix
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y_onehot = np.zeros((len(X64), n_classes), dtype=np.float64)
    for i, yi in enumerate(y_int):
        Y_onehot[i, class_to_idx[int(yi)]] = 1.0

    # OVR OLS: W = (X^T X + lambda I)^{-1} X^T Y
    reg = 1e-4 * np.eye(d, dtype=np.float64)
    W = np.linalg.solve(X64.T @ X64 + reg, X64.T @ Y_onehot)
    return W  # (d, n_classes)


def _ovr_predict(X: np.ndarray, W: np.ndarray, classes: list[int]) -> np.ndarray:
    """Predict class labels from OVR head matrix."""
    scores = X.astype(np.float64) @ W  # (n, n_classes)
    pred_idx = np.argmax(scores, axis=1)
    return np.array([classes[i] for i in pred_idx])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="CIFAR-100 B_j^2 multiclass proxy")
    parser.add_argument("--results-dir", default="tmp/cifar100_bj_proxy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--data-root", default=None, help="CIFAR-100 data root")
    parser.add_argument("--feature-cache-dir", default=None, help="Feature cache dir")
    parser.add_argument("--n-workers", type=int, default=0, help="(unused)")
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K = 10
    T = 30
    n_concepts = 4
    seeds = args.seeds
    label_split = "disjoint"

    all_results = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")

        cfg = CIFAR100RecurrenceConfig(
            K=K, T=T, n_samples=500,
            rho=T / n_concepts, alpha=0.5,
            label_split=label_split, seed=seed,
        )
        dataset = generate_cifar100_recurrence_dataset(cfg)
        data = dataset.data
        concept_matrix = dataset.concept_matrix

        X0, _ = data[(0, 0)]
        d = X0.shape[1]
        print(f"Feature dim d={d}")

        unique_concepts = sorted(set(int(v) for v in np.unique(concept_matrix)))
        C = len(unique_concepts)

        # ---- Head-space B_j^2: fit per-concept OVR heads ----
        concept_heads: dict[int, np.ndarray] = {}
        concept_classes: dict[int, list[int]] = {}
        concept_n_samples: dict[int, int] = {}
        concept_residual_vars: dict[int, float] = {}

        # Collect all classes for global head
        all_classes_set: set[int] = set()
        for k in range(K):
            for t in range(T):
                _, y = data[(k, t)]
                all_classes_set.update(y.astype(int).tolist())
        all_classes = sorted(all_classes_set)
        n_total_classes = len(all_classes)
        print(f"Total classes across all concepts: {n_total_classes}")

        for cid in unique_concepts:
            all_X = []
            all_y = []
            for k in range(K):
                for t in range(T):
                    if int(concept_matrix[k, t]) == cid:
                        X, y = data[(k, t)]
                        all_X.append(X)
                        all_y.append(y)
            X_c = np.concatenate(all_X).astype(np.float64)
            y_c = np.concatenate(all_y).astype(np.int64)
            concept_n_samples[cid] = len(X_c)
            concept_classes[cid] = sorted(set(y_c.tolist()))

            # Fit OVR head for this concept's classes
            W_c = _ovr_fit(X_c, y_c)
            # Embed into full class space: pad to (d, n_total_classes)
            W_full = np.zeros((d, n_total_classes), dtype=np.float64)
            for local_idx, cls in enumerate(concept_classes[cid]):
                global_idx = all_classes.index(cls)
                W_full[:, global_idx] = W_c[:, local_idx]
            concept_heads[cid] = W_full

            # Residual variance
            scores = X_c @ W_c
            class_to_local = {c: i for i, c in enumerate(concept_classes[cid])}
            target_scores = np.array([scores[i, class_to_local[int(y_c[i])]] for i in range(len(y_c))])
            residuals = 1.0 - target_scores  # residual from correct-class score
            concept_residual_vars[cid] = float(np.mean(residuals ** 2))

            n_local_classes = len(concept_classes[cid])
            print(f"  Concept {cid}: {len(X_c)} samples, {n_local_classes} classes, "
                  f"head Frobenius norm={np.linalg.norm(W_full, 'fro'):.4f}")

        # Compute B_j^2 in head-matrix Frobenius space: ||W_j - W_bar||_F^2
        W_bar = np.mean([concept_heads[c] for c in unique_concepts], axis=0)

        B_j_sq_list = []
        for cid in unique_concepts:
            diff = concept_heads[cid] - W_bar
            B_j_sq = float(np.sum(diff ** 2))  # Frobenius norm squared
            B_j_sq_list.append(B_j_sq)
            print(f"  Concept {cid}: B_j^2 (head-matrix Frobenius) = {B_j_sq:.6f}")

        B_j_sq_mean = float(np.mean(B_j_sq_list))
        B_j_sq_min = float(np.min(B_j_sq_list))

        # Pairwise concept separation
        pairwise_dists = []
        for i, ci in enumerate(unique_concepts):
            for j, cj in enumerate(unique_concepts):
                if i < j:
                    dist = float(np.linalg.norm(concept_heads[ci] - concept_heads[cj], 'fro'))
                    pairwise_dists.append(dist)
        delta_min = min(pairwise_dists) if pairwise_dists else 0.0
        delta_mean = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0

        sigma_sq = float(np.mean([concept_residual_vars[c] for c in unique_concepts]))

        # ---- Empirical Oracle vs FedAvg (classification accuracy) ----
        # Global OVR head
        X_all = np.concatenate([data[(k, t)][0] for k in range(K) for t in range(T)]).astype(np.float64)
        y_all = np.concatenate([data[(k, t)][1] for k in range(K) for t in range(T)]).astype(np.int64)
        W_global = _ovr_fit(X_all, y_all)

        test_start = 3 * T // 4
        fedavg_correct = 0
        oracle_correct = 0
        total_test = 0

        for k in range(K):
            for t in range(test_start, T):
                X, y = data[(k, t)]
                X64 = X.astype(np.float64)
                y_int = y.astype(np.int64)
                cid = int(concept_matrix[k, t])

                # FedAvg: use global head
                pred_global = _ovr_predict(X64, W_global, all_classes)
                fedavg_correct += int(np.sum(pred_global == y_int))

                # Oracle: use concept-specific head
                W_c = concept_heads[cid]
                pred_oracle = _ovr_predict(X64, W_c, all_classes)
                oracle_correct += int(np.sum(pred_oracle == y_int))

                total_test += len(y_int)

        fedavg_acc = fedavg_correct / max(total_test, 1)
        oracle_acc = oracle_correct / max(total_test, 1)
        empirical_oracle_wins = oracle_acc > fedavg_acc

        # Compute SNR
        n_per_client = concept_n_samples[unique_concepts[0]] // max(1, K * T // C)
        n_train = max(1, n_per_client // 2)
        # Effective d for multiclass: d * n_classes
        d_eff = d * n_total_classes
        snr = K * n_train * B_j_sq_mean / (sigma_sq * d_eff)

        print(f"\n  Results:")
        print(f"    d = {d}, n_classes = {n_total_classes}, d_eff = {d_eff}")
        print(f"    C = {C}, K = {K}, n_train = {n_train}")
        print(f"    B_j^2 (mean, Frobenius) = {B_j_sq_mean:.6f}")
        print(f"    B_j^2 (min) = {B_j_sq_min:.6f}")
        print(f"    Delta (min pairwise, Frobenius) = {delta_min:.4f}")
        print(f"    sigma^2 (residual) = {sigma_sq:.6f}")
        print(f"    SNR_concept = {snr:.2f}")
        print(f"    Crossover threshold (C-1) = {C-1}")
        print(f"    Theory prediction: {'Oracle wins' if snr > C-1 else 'Global wins'}")
        print(f"    Empirical: FedAvg acc={fedavg_acc:.4f}, Oracle acc={oracle_acc:.4f}")
        print(f"    Empirical Oracle wins: {empirical_oracle_wins}")
        print(f"    Theory-experiment match: {'YES' if (snr > C-1) == empirical_oracle_wins else 'NO'}")

        all_results.append({
            "seed": seed,
            "d": d, "n_classes": n_total_classes, "d_eff": d_eff,
            "C": C, "K": K, "n_train": n_train,
            "B_j_sq_mean": round(B_j_sq_mean, 6),
            "B_j_sq_min": round(B_j_sq_min, 6),
            "B_j_sq_per_concept": {str(c): round(B_j_sq_list[i], 6) for i, c in enumerate(unique_concepts)},
            "delta_min": round(delta_min, 4),
            "delta_mean": round(delta_mean, 4),
            "sigma_sq": round(sigma_sq, 6),
            "SNR_concept": round(snr, 4),
            "theory_oracle_wins": bool(snr > (C - 1)),
            "fedavg_acc": round(fedavg_acc, 4),
            "oracle_acc": round(oracle_acc, 4),
            "empirical_oracle_wins": bool(empirical_oracle_wins),
            "theory_experiment_match": bool((snr > (C - 1)) == empirical_oracle_wins),
        })

    # Save
    with open(out_dir / "bj_proxy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'bj_proxy_results.json'}")


if __name__ == "__main__":
    main()
