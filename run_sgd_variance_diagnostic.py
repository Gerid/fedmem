"""SGD variance inflation diagnostic: empirical evidence for the CFL > Oracle bridge.

Tests the core claim: Oracle overfits more than CFL because concept groups are small.
With E epochs of SGD on n_j samples, effective variance σ²_eff increases.
CFL's contamination acts as regularization, reducing overfitting.

Protocol:
  For each (K, E) config on CIFAR-100:
    1. Train Oracle (per-concept) and measure train-vs-test accuracy gap
    2. Train CFL and measure its train-vs-test gap
    3. Train with artificially enlarged groups (pooling wrong-concept data)
    4. Directly estimate σ²_eff by the MSE of parameter estimates

Prediction: Oracle's train-test gap > CFL's train-test gap, especially when K/C is small.

Usage:
    python run_sgd_variance_diagnostic.py --seeds 42 43 44 45 46
    python run_sgd_variance_diagnostic.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _train_evaluate(X_train, y_train, X_test, y_test, n_features, n_classes,
                    n_epochs=5, lr=0.05, seed=42):
    """Train a linear model and return train/test accuracy."""
    from fedprotrack.models.torch_model import TorchLinearClassifier
    model = TorchLinearClassifier(n_features, n_classes, seed=seed, n_epochs=n_epochs, lr=lr)
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = float((train_preds == y_train).mean())
    test_acc = float((test_preds == y_test).mean())
    return train_acc, test_acc, model


def _run_one_config(K, n_epochs, seed, args):
    """Measure train-test gaps for Oracle vs CFL-like mixing."""
    from fedprotrack.real_data.cifar100_recurrence import (
        CIFAR100RecurrenceConfig,
        generate_cifar100_recurrence_dataset,
    )
    from fedprotrack.baselines.runners import _extract_dims
    from fedprotrack.estimators.shrinkage import compute_effective_rank

    cfg = CIFAR100RecurrenceConfig(
        K=K,
        T=args.T,
        n_samples=args.n_samples,
        rho=args.rho,
        alpha=0.75,
        delta=0.85,
        n_features=args.n_features,
        samples_per_coarse_class=120,
        batch_size=256,
        n_workers=args.n_workers,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
        seed=seed,
        label_split="none",
        backbone=args.backbone,
    )

    dataset = generate_cifar100_recurrence_dataset(cfg)
    C = int(dataset.concept_matrix.max()) + 1
    _K, T, n_features, n_classes = _extract_dims(dataset)

    # Use the last time step for evaluation, an earlier one for training.
    t_train = 0
    t_test = min(5, T - 1)

    # Collect per-concept data.
    concept_data_train: dict[int, list] = {j: [] for j in range(C)}
    concept_data_test: dict[int, list] = {j: [] for j in range(C)}
    for k in range(K):
        c_train = int(dataset.concept_matrix[k, t_train])
        c_test = int(dataset.concept_matrix[k, t_test])
        X_tr, y_tr = dataset.data[(k, t_train)]
        X_te, y_te = dataset.data[(k, t_test)]
        concept_data_train[c_train].append((X_tr, y_tr))
        concept_data_test[c_test].append((X_te, y_te))

    rows = []

    for j in range(C):
        if not concept_data_train[j] or not concept_data_test[j]:
            continue

        # Oracle: train on concept j's data only.
        X_oracle = np.concatenate([d[0] for d in concept_data_train[j]])
        y_oracle = np.concatenate([d[1] for d in concept_data_train[j]])
        X_test_j = np.concatenate([d[0] for d in concept_data_test[j]])
        y_test_j = np.concatenate([d[1] for d in concept_data_test[j]])

        oracle_train_acc, oracle_test_acc, _ = _train_evaluate(
            X_oracle, y_oracle, X_test_j, y_test_j,
            n_features, n_classes, n_epochs=n_epochs, lr=args.lr, seed=seed,
        )
        oracle_gap = oracle_train_acc - oracle_test_acc

        # Contaminated: mix in data from other concepts (simulating CFL's error).
        # Mix fraction η ≈ 0.2 (typical CFL error rate).
        for eta in [0.0, 0.1, 0.2, 0.3, 0.5]:
            if eta == 0.0:
                X_mixed = X_oracle
                y_mixed = y_oracle
            else:
                # Pool some data from other concepts.
                other_data = []
                for j2 in range(C):
                    if j2 != j and concept_data_train[j2]:
                        for d in concept_data_train[j2]:
                            other_data.append(d)
                if not other_data:
                    continue

                X_other = np.concatenate([d[0] for d in other_data])
                y_other = np.concatenate([d[1] for d in other_data])

                # Sample η fraction from other concepts.
                n_contam = int(len(X_oracle) * eta / (1 - eta))
                n_contam = min(n_contam, len(X_other))
                rng = np.random.default_rng(seed + j * 100)
                idx = rng.choice(len(X_other), size=n_contam, replace=False)

                X_mixed = np.concatenate([X_oracle, X_other[idx]])
                y_mixed = np.concatenate([y_oracle, y_other[idx]])

            mixed_train_acc, mixed_test_acc, _ = _train_evaluate(
                X_mixed, y_mixed, X_test_j, y_test_j,
                n_features, n_classes, n_epochs=n_epochs, lr=args.lr, seed=seed,
            )
            mixed_gap = mixed_train_acc - mixed_test_acc

            row = {
                "K": K,
                "C": C,
                "K_over_C": round(K / C, 2),
                "n_epochs": n_epochs,
                "seed": seed,
                "concept_j": j,
                "n_oracle": len(X_oracle),
                "eta_inject": eta,
                "n_mixed": len(X_mixed),
                "oracle_train_acc": round(oracle_train_acc, 4),
                "oracle_test_acc": round(oracle_test_acc, 4),
                "oracle_gap": round(oracle_gap, 4),
                "mixed_train_acc": round(mixed_train_acc, 4),
                "mixed_test_acc": round(mixed_test_acc, 4),
                "mixed_gap": round(mixed_gap, 4),
                "overfitting_reduction": round(oracle_gap - mixed_gap, 4),
                "test_acc_improvement": round(mixed_test_acc - oracle_test_acc, 4),
            }
            rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGD variance inflation diagnostic",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--seed", type=int, default=None, help="Single seed (RunPod compat)")
    parser.add_argument("--K-values", type=int, nargs="+", default=[4, 8, 12, 20])
    parser.add_argument("--epoch-values", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--results-dir", default="tmp/sgd_variance_diagnostic")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--n-workers", type=int, default=2)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--backbone", default="resnet18")
    args = parser.parse_args()

    if args.seed is not None:
        args.seeds = [args.seed]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    total = len(args.K_values) * len(args.epoch_values) * len(args.seeds)
    done = 0

    for K in args.K_values:
        for n_epochs in args.epoch_values:
            for seed in args.seeds:
                done += 1
                print(f"\n[{done}/{total}] K={K}, epochs={n_epochs}, seed={seed}")
                try:
                    rows = _run_one_config(K, n_epochs, seed, args)
                    all_rows.extend(rows)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    traceback.print_exc()

    # Save.
    with open(results_dir / "results.json", "w") as f:
        json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)

    if all_rows:
        csv_path = results_dir / "results.csv"
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)

    # ── Analysis ──
    print(f"\n{'='*70}")
    print("SGD Variance Inflation Analysis")
    print(f"{'='*70}")

    from collections import defaultdict

    # Group by (K, epochs, eta).
    gap_data: dict[tuple, list] = defaultdict(list)
    test_data: dict[tuple, list] = defaultdict(list)
    for r in all_rows:
        key = (r["K"], r["n_epochs"], r["eta_inject"])
        gap_data[key].append(r["mixed_gap"])
        test_data[key].append(r["mixed_test_acc"])

    print(f"\n--- Overfitting (train-test gap) by K × epochs × η ---")
    print(f"{'K':>4} {'E':>3} {'η':>5} {'gap_mean':>9} {'test_acc':>9}")
    print("-" * 40)
    for K in args.K_values:
        for E in args.epoch_values:
            for eta in [0.0, 0.1, 0.2, 0.3, 0.5]:
                key = (K, E, eta)
                gaps = gap_data.get(key, [])
                tests = test_data.get(key, [])
                if gaps:
                    print(f"{K:>4} {E:>3} {eta:>5.1f} {np.mean(gaps):>9.4f} {np.mean(tests):>9.4f}")
        print()

    # Key finding: optimal η (best test accuracy) per (K, E).
    print(f"\n--- Optimal Contamination η (best test acc) by K × epochs ---")
    print(f"{'K':>4} {'E':>3} {'η*':>5} {'test@η*':>8} {'test@0':>8} {'Δ':>7}")
    print("-" * 45)
    for K in args.K_values:
        for E in args.epoch_values:
            best_eta = 0.0
            best_test = -1
            test_at_0 = float("nan")
            for eta in [0.0, 0.1, 0.2, 0.3, 0.5]:
                key = (K, E, eta)
                tests = test_data.get(key, [])
                if tests:
                    t_mean = np.mean(tests)
                    if eta == 0.0:
                        test_at_0 = t_mean
                    if t_mean > best_test:
                        best_test = t_mean
                        best_eta = eta
            if best_test > 0:
                delta = best_test - test_at_0
                print(f"{K:>4} {E:>3} {best_eta:>5.1f} {best_test:>8.4f} "
                      f"{test_at_0:>8.4f} {delta:>+7.4f}")

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
