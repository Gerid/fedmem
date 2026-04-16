from __future__ import annotations

"""Low-SNR CIFAR-100 bridge: overlapping label splits to test the OTHER side of the crossover.

When concepts share most of their classes, B_j^2 is small, SNR < C-1,
and the theory predicts FedAvg >= Oracle. This complements the disjoint-label
high-SNR bridge in run_cifar100_bj_proxy.py.
"""

import json
from pathlib import Path

import numpy as np

from fedprotrack.real_data.cifar100_recurrence import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
)


def _ovr_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit one-vs-rest OLS regression heads."""
    X64 = X.astype(np.float64)
    y_int = y.astype(np.int64)
    d = X64.shape[1]
    classes = sorted(set(y_int.tolist()))
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y_onehot = np.zeros((len(X64), n_classes), dtype=np.float64)
    for i, yi in enumerate(y_int):
        Y_onehot[i, class_to_idx[int(yi)]] = 1.0
    reg = 1e-4 * np.eye(d, dtype=np.float64)
    W = np.linalg.solve(X64.T @ X64 + reg, X64.T @ Y_onehot)
    return W


def _ovr_predict(X: np.ndarray, W: np.ndarray, classes: list[int]) -> np.ndarray:
    """Predict class labels from OVR head matrix."""
    scores = X.astype(np.float64) @ W
    pred_idx = np.argmax(scores, axis=1)
    return np.array([classes[i] for i in pred_idx])


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Low-SNR CIFAR-100 B_j^2 proxy (overlapping labels)")
    parser.add_argument("--results-dir", default="tmp/cifar100_bj_proxy_lowsnr")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K = 10
    T = 30
    n_concepts = 4
    seeds = args.seeds

    # Use "shared" label split: all concepts see ALL 20 coarse classes.
    # Concepts differ only by client sampling order, not by label content.
    # This makes B_j^2 ≈ 0 → SNR << C-1 → theory predicts FedAvg >= Oracle.
    label_split = "shared"

    all_results = []

    for seed in seeds:
        print(f"\n=== Seed {seed} (shared labels → low SNR) ===")

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

        unique_concepts = sorted(set(int(v) for v in np.unique(concept_matrix)))
        C = len(unique_concepts)

        # Collect all classes
        all_classes_set: set[int] = set()
        for k in range(K):
            for t in range(T):
                _, y = data[(k, t)]
                all_classes_set.update(y.astype(int).tolist())
        all_classes = sorted(all_classes_set)
        n_total_classes = len(all_classes)

        # Fit per-concept OVR heads
        concept_heads: dict[int, np.ndarray] = {}
        concept_classes: dict[int, list[int]] = {}
        concept_n_samples: dict[int, int] = {}
        concept_residual_vars: dict[int, float] = {}

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

            W_c = _ovr_fit(X_c, y_c)
            # Embed into full class space
            W_full = np.zeros((d, n_total_classes), dtype=np.float64)
            for local_idx, cls in enumerate(concept_classes[cid]):
                global_idx = all_classes.index(cls)
                W_full[:, global_idx] = W_c[:, local_idx]
            concept_heads[cid] = W_full

            # Residual variance
            scores = X_c @ W_c
            class_to_local = {c: i for i, c in enumerate(concept_classes[cid])}
            target_scores = np.array([scores[i, class_to_local[int(y_c[i])]] for i in range(len(y_c))])
            residuals = 1.0 - target_scores
            concept_residual_vars[cid] = float(np.mean(residuals ** 2))

            n_local_classes = len(concept_classes[cid])
            print(f"  Concept {cid}: {len(X_c)} samples, {n_local_classes} classes")

        # Compute B_j^2
        W_bar = np.mean([concept_heads[c] for c in unique_concepts], axis=0)
        B_j_sq_list = []
        for cid in unique_concepts:
            diff = concept_heads[cid] - W_bar
            B_j_sq = float(np.sum(diff ** 2))
            B_j_sq_list.append(B_j_sq)
            print(f"  Concept {cid}: B_j^2 = {B_j_sq:.6f}")

        B_j_sq_mean = float(np.mean(B_j_sq_list))
        sigma_sq = float(np.mean([concept_residual_vars[c] for c in unique_concepts]))

        # Empirical Oracle vs FedAvg
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

                pred_global = _ovr_predict(X64, W_global, all_classes)
                fedavg_correct += int(np.sum(pred_global == y_int))

                W_c = concept_heads[cid]
                pred_oracle = _ovr_predict(X64, W_c, all_classes)
                oracle_correct += int(np.sum(pred_oracle == y_int))

                total_test += len(y_int)

        fedavg_acc = fedavg_correct / max(total_test, 1)
        oracle_acc = oracle_correct / max(total_test, 1)
        empirical_oracle_wins = oracle_acc > fedavg_acc

        n_per_client = concept_n_samples[unique_concepts[0]] // max(1, K * T // C)
        n_train = max(1, n_per_client // 2)
        d_eff = d * n_total_classes
        snr = K * n_train * B_j_sq_mean / (sigma_sq * d_eff)
        theory_oracle_wins = snr > (C - 1)

        print(f"\n  Results:")
        print(f"    B_j^2 (mean) = {B_j_sq_mean:.6f}")
        print(f"    sigma^2 = {sigma_sq:.6f}")
        print(f"    SNR = {snr:.4f}, threshold C-1 = {C-1}")
        print(f"    Theory: {'Oracle' if theory_oracle_wins else 'FedAvg'}")
        print(f"    Empirical: FedAvg={fedavg_acc:.4f}, Oracle={oracle_acc:.4f}")
        print(f"    Match: {(theory_oracle_wins == empirical_oracle_wins)}")

        all_results.append({
            "seed": seed, "label_split": label_split,
            "d": d, "n_classes": n_total_classes, "d_eff": d_eff,
            "C": C, "K": K, "n_train": n_train,
            "B_j_sq_mean": round(B_j_sq_mean, 6),
            "sigma_sq": round(sigma_sq, 6),
            "SNR_concept": round(snr, 4),
            "theory_oracle_wins": bool(theory_oracle_wins),
            "fedavg_acc": round(fedavg_acc, 4),
            "oracle_acc": round(oracle_acc, 4),
            "empirical_oracle_wins": bool(empirical_oracle_wins),
            "theory_experiment_match": bool(theory_oracle_wins == empirical_oracle_wins),
        })

    with open(out_dir / "bj_proxy_lowsnr_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'bj_proxy_lowsnr_results.json'}")


if __name__ == "__main__":
    main()
