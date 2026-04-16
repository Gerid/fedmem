from __future__ import annotations

"""DRCT d_eff ablation: which effective dimension gives the best shrinkage?

Five-arm comparison on CIFAR-100 frozen features (Oracle concept assignments):
  1. No-shrink       — pure concept-level aggregation (λ=0)
  2. Fixed-d         — λ calibrated with ambient d=128
  3. Feature-r_Σ     — λ calibrated with feature effective rank
  4. DRCT-r^G        — λ calibrated with gradient effective rank (proposed)
  5. Empirical-Bayes — λ from between-concept variance (Proposition 1)

Each arm uses Oracle concept assignments to isolate the shrinkage effect
from the concept identification effect.

Usage:
    python run_neurips_drct_ablation.py --seed 42
    python run_neurips_drct_ablation.py --seed 42 --K 12 --T 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")
# GPU_THRESHOLD=0 removed: linear models stay on CPU (faster for <8192 params)


# ---------------------------------------------------------------------------
# Gradient covariance effective rank (from drct_falsification_test.py)
# ---------------------------------------------------------------------------

def _participation_ratio(eigs: np.ndarray) -> float:
    """Participation ratio = (tr M)^2 / tr(M^2)."""
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = eigs[eigs > 0.0]
    if eigs.size == 0:
        return 1.0
    num = float(eigs.sum()) ** 2
    den = float((eigs ** 2).sum())
    return num / den if den > 1e-30 else 1.0


def compute_gradient_effective_rank(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute r^G from last-layer gradient covariance via Gram trick.

    Parameters
    ----------
    model : TorchLinearClassifier
        Trained linear classifier with .model attribute (nn.Linear).
    X : np.ndarray
        Feature matrix, shape (n_samples, d).
    y : np.ndarray
        Labels, shape (n_samples,).

    Returns
    -------
    float
        Gradient effective rank r^G.
    """
    import torch
    import torch.nn.functional as F

    device = model.device
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).long().to(device)

    model._linear.eval()
    with torch.no_grad():
        logits = model._linear(X_t)
        p = torch.softmax(logits, dim=1)
        n_classes = logits.shape[1]
        e_y = F.one_hot(y_t, num_classes=n_classes).float()
        residual = (p - e_y).cpu().numpy().astype(np.float64)

    features = X.astype(np.float64)
    N = features.shape[0]

    # Gram matrix: K_ij = (1/N) * <r_i, r_j> * <z_i, z_j>
    RRT = residual @ residual.T
    ZZT = features @ features.T
    K = (RRT * ZZT) / float(N)
    eigs = np.linalg.eigvalsh(K)
    return _participation_ratio(eigs)


def compute_feature_effective_rank(X: np.ndarray) -> float:
    """Compute r_Σ from feature covariance via Gram trick.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, d).

    Returns
    -------
    float
        Feature effective rank r_Σ.
    """
    X = X.astype(np.float64)
    X_centered = X - X.mean(axis=0, keepdims=True)
    N = X.shape[0]
    K = (X_centered @ X_centered.T) / float(N)
    eigs = np.linalg.eigvalsh(K)
    return _participation_ratio(eigs)


# ---------------------------------------------------------------------------
# Shrinkage arms
# ---------------------------------------------------------------------------

def compute_shrinkage_lambda(
    sigma2: float,
    sigma_B2: float,
    n_concept: float,
    d_eff: float,
) -> float:
    """λ = σ² d_eff / n_concept / (σ² d_eff / n_concept + σ_B²)."""
    if n_concept <= 0 or d_eff <= 0:
        return 0.5
    variance_term = sigma2 * d_eff / n_concept
    denom = variance_term + sigma_B2
    if denom < 1e-30:
        return 0.5
    return float(np.clip(variance_term / denom, 0.0, 1.0))


def _flatten_params(params: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a param dict into a single 1-D vector."""
    return np.concatenate([v.ravel() for v in params.values()])


def _average_param_dicts(
    param_list: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Element-wise average of a list of param dicts."""
    out: dict[str, np.ndarray] = {}
    for key in param_list[0]:
        out[key] = np.mean([p[key] for p in param_list], axis=0)
    return out


def _shrink_param_dicts(
    concept_params: dict[str, np.ndarray],
    global_params: dict[str, np.ndarray],
    lam: float,
) -> dict[str, np.ndarray]:
    """Shrink concept params toward global: (1-λ)*concept + λ*global."""
    out: dict[str, np.ndarray] = {}
    for key in concept_params:
        out[key] = (1 - lam) * concept_params[key] + lam * global_params[key]
    return out


def estimate_sigma_B2(
    concept_weights: dict[int, dict[str, np.ndarray]],
    sigma2: float,
    n_concept: float,
) -> float:
    """Estimate between-concept variance from concept-level weight averages."""
    if len(concept_weights) < 2:
        return 0.0
    flat = [_flatten_params(w) for w in concept_weights.values()]
    stacked = np.stack(flat)
    global_mean = stacked.mean(axis=0)
    d = stacked.shape[1]
    C = len(concept_weights)
    spread = np.sum((stacked - global_mean) ** 2) / ((C - 1) * d)
    correction = sigma2 / n_concept if n_concept > 0 else 0.0
    return float(max(spread - correction, 0.0))


def estimate_sigma2_from_residuals(
    concept_uploads: dict[int, list[dict[str, np.ndarray]]],
    concept_means: dict[int, dict[str, np.ndarray]],
) -> float:
    """Estimate noise variance from within-concept model dispersion."""
    total_var = 0.0
    total_dim = 0
    total_count = 0
    for cid, uploads in concept_uploads.items():
        if len(uploads) < 2:
            continue
        mean_w = _flatten_params(concept_means[cid])
        for w in uploads:
            diff = _flatten_params(w) - mean_w
            total_var += np.sum(diff ** 2)
            total_dim += diff.size
        total_count += len(uploads) - 1
    if total_count == 0 or total_dim == 0:
        return 1e-4
    return float(total_var / total_dim)


# ---------------------------------------------------------------------------
# Oracle shrinkage experiment
# ---------------------------------------------------------------------------

def run_oracle_shrinkage_arm(
    dataset,
    *,
    arm_name: str,
    d_eff_source: str,
    lr: float,
    n_epochs: int,
    federation_every: int,
    seed: int,
) -> dict:
    """Run Oracle with a specific shrinkage calibration.

    Parameters
    ----------
    dataset : DriftDataset
    arm_name : str
        Human-readable arm label.
    d_eff_source : str
        One of "none", "fixed_d", "feature_r_sigma", "gradient_r_g",
        "empirical_bayes".
    lr, n_epochs, federation_every, seed : various

    Returns
    -------
    dict
        Row with arm_name, final_accuracy, accuracy_auc, lambda values, etc.
    """
    from fedprotrack.models import TorchLinearClassifier

    concept_matrix = dataset.concept_matrix
    K, T = concept_matrix.shape
    n_concepts = int(concept_matrix.max()) + 1

    # Infer dimensions
    X0, y0 = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    all_labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    acc_matrix = np.zeros((K, T), dtype=np.float64)
    concept_models: dict[int, TorchLinearClassifier] = {}
    lambda_history: list[float] = []
    r_g_history: list[float] = []
    r_sigma_history: list[float] = []

    for t in range(T):
        # Collect per-concept local models
        concept_uploads: dict[int, list[dict[str, np.ndarray]]] = {}
        concept_data_X: dict[int, list[np.ndarray]] = {}
        concept_data_y: dict[int, list[np.ndarray]] = {}

        for k in range(K):
            X, y = dataset.data[(k, t)]
            cid = int(concept_matrix[k, t])
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            # Init from concept model if available
            model = TorchLinearClassifier(
                n_features=n_features, n_classes=n_classes,
                lr=lr, n_epochs=n_epochs, seed=seed + k * T + t,
            )
            if cid in concept_models:
                model.set_params(concept_models[cid].get_params())
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            acc_matrix[k, t] = float(np.mean(y_pred == y_test))

            # Collect weights
            w = model.get_params()
            concept_uploads.setdefault(cid, []).append(w)
            concept_data_X.setdefault(cid, []).append(X)
            concept_data_y.setdefault(cid, []).append(y)

        # Federation step
        if (t + 1) % federation_every == 0 and t < T - 1:
            # Concept-level averages
            concept_means: dict[int, dict[str, np.ndarray]] = {}
            for cid, uploads in concept_uploads.items():
                concept_means[cid] = _average_param_dicts(uploads)

            # Global average
            all_uploads = [w for ws in concept_uploads.values() for w in ws]
            global_mean = _average_param_dicts(all_uploads)

            if len(concept_means) < 2 or d_eff_source == "none":
                # No shrinkage — pure concept-level
                for cid, params in concept_means.items():
                    m = TorchLinearClassifier(
                        n_features=n_features, n_classes=n_classes,
                        lr=lr, n_epochs=1, seed=seed,
                    )
                    m.set_params(params)
                    concept_models[cid] = m
                lambda_history.append(0.0)
                continue

            # Estimate noise and between-concept variance
            sigma2 = estimate_sigma2_from_residuals(
                concept_uploads, concept_means,
            )
            n_per_concept = K / n_concepts * (X.shape[0] // 2)

            # Compute d_eff based on arm
            if d_eff_source == "fixed_d":
                d_eff = float(n_features)

            elif d_eff_source == "feature_r_sigma":
                # Pool features from all clients this round
                all_X_round = np.concatenate(
                    [X for Xs in concept_data_X.values() for X in Xs]
                )
                d_eff = compute_feature_effective_rank(all_X_round)
                r_sigma_history.append(d_eff)

            elif d_eff_source == "gradient_r_g":
                # Compute r^G from a representative concept model
                r_g_vals = []
                for cid in concept_means:
                    if cid in concept_data_X and concept_data_X[cid]:
                        # Use concept model + concept data
                        m_tmp = TorchLinearClassifier(
                            n_features=n_features, n_classes=n_classes,
                            lr=lr, n_epochs=1, seed=seed,
                        )
                        m_tmp.set_params(concept_means[cid])
                        X_c = np.concatenate(concept_data_X[cid])
                        y_c = np.concatenate(concept_data_y[cid])
                        # Subsample for efficiency
                        if len(y_c) > 512:
                            rng = np.random.default_rng(seed + t)
                            idx = rng.choice(len(y_c), 512, replace=False)
                            X_c, y_c = X_c[idx], y_c[idx]
                        r_g = compute_gradient_effective_rank(m_tmp, X_c, y_c)
                        r_g_vals.append(r_g)
                d_eff = float(np.mean(r_g_vals)) if r_g_vals else float(n_features)
                r_g_history.append(d_eff)

                # Also compute r_sigma for logging
                all_X_round = np.concatenate(
                    [X for Xs in concept_data_X.values() for X in Xs]
                )
                r_sigma_history.append(
                    compute_feature_effective_rank(all_X_round)
                )

            elif d_eff_source == "empirical_bayes":
                # Proposition 1: λ from σ_B² directly, no d_eff
                sigma_B2 = estimate_sigma_B2(
                    concept_means, sigma2, n_per_concept,
                )
                if sigma_B2 < 1e-12:
                    lam = 0.5
                else:
                    var_term = sigma2 / n_per_concept if n_per_concept > 0 else 1e-4
                    lam = float(np.clip(var_term / (var_term + sigma_B2), 0, 1))

                for cid in concept_means:
                    w_shrunk = _shrink_param_dicts(concept_means[cid], global_mean, lam)
                    m = TorchLinearClassifier(
                        n_features=n_features, n_classes=n_classes,
                        lr=lr, n_epochs=1, seed=seed,
                    )
                    m.set_params(w_shrunk)
                    concept_models[cid] = m
                lambda_history.append(lam)
                continue
            else:
                d_eff = float(n_features)

            # Compute λ using d_eff
            sigma_B2 = estimate_sigma_B2(
                concept_means, sigma2, n_per_concept,
            )
            lam = compute_shrinkage_lambda(
                sigma2, sigma_B2, n_per_concept, d_eff,
            )

            # Apply shrinkage
            for cid in concept_means:
                w_shrunk = _shrink_param_dicts(concept_means[cid], global_mean, lam)
                m = TorchLinearClassifier(
                    n_features=n_features, n_classes=n_classes,
                    lr=lr, n_epochs=1, seed=seed,
                )
                m.set_params(w_shrunk)
                concept_models[cid] = m
            lambda_history.append(lam)

    final_acc = float(acc_matrix[:, -1].mean())
    # AUC: mean accuracy over all rounds
    trapz = getattr(np, "trapezoid", None) or np.trapz
    mean_curve = acc_matrix.mean(axis=0)
    auc = float(trapz(mean_curve) / max(T - 1, 1))

    return {
        "arm": arm_name,
        "d_eff_source": d_eff_source,
        "final_accuracy": round(final_acc, 5),
        "accuracy_auc": round(auc, 5),
        "mean_lambda": round(float(np.mean(lambda_history)), 5) if lambda_history else 0.0,
        "mean_r_g": round(float(np.mean(r_g_history)), 2) if r_g_history else None,
        "mean_r_sigma": round(float(np.mean(r_sigma_history)), 2) if r_sigma_history else None,
        "d": n_features,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ARMS = [
    ("No-shrink", "none"),
    ("Fixed-d", "fixed_d"),
    ("Feature-r_Σ", "feature_r_sigma"),
    ("DRCT-r^G", "gradient_r_g"),
    ("Empirical-Bayes", "empirical_bayes"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DRCT d_eff ablation: 5-arm shrinkage comparison"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results_drct_ablation")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--rho", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.85)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from fedprotrack.real_data.cifar100_recurrence import (
        CIFAR100RecurrenceConfig,
        generate_cifar100_recurrence_dataset,
        prepare_cifar100_recurrence_feature_cache,
    )

    seed = args.seed
    print(f"=== DRCT Shrinkage Ablation | seed={seed} ===")
    print(f"    K={args.K}, T={args.T}, d={args.n_features}")

    cfg = CIFAR100RecurrenceConfig(
        K=args.K,
        T=args.T,
        n_samples=400,
        rho=args.rho,
        alpha=args.alpha,
        delta=args.delta,
        n_features=args.n_features,
        samples_per_coarse_class=120,
        batch_size=256,
        n_workers=args.n_workers,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
        feature_seed=2718,
        seed=seed,
    )
    print("Warming feature cache...", flush=True)
    prepare_cifar100_recurrence_feature_cache(cfg)

    dataset = generate_cifar100_recurrence_dataset(cfg)
    n_concepts = int(dataset.concept_matrix.max()) + 1
    print(f"    n_concepts={n_concepts}")

    # Also run FedAvg and Oracle baselines for reference
    from fedprotrack.experiment.baselines import (
        run_fedavg_baseline,
        run_oracle_baseline,
    )
    from fedprotrack.experiment.runner import ExperimentConfig

    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=args.federation_every,
    )

    print("\n--- Running baselines ---")
    t0 = time.time()
    fedavg_res = run_fedavg_baseline(
        exp_cfg, dataset=dataset, lr=args.lr, n_epochs=args.n_epochs, seed=seed,
    )
    fedavg_acc = float(np.asarray(fedavg_res.accuracy_matrix)[:, -1].mean())
    print(f"  FedAvg:  {fedavg_acc:.4f} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    oracle_res = run_oracle_baseline(
        exp_cfg, dataset=dataset, lr=args.lr, n_epochs=args.n_epochs, seed=seed,
    )
    oracle_acc = float(np.asarray(oracle_res.accuracy_matrix)[:, -1].mean())
    print(f"  Oracle:  {oracle_acc:.4f} ({time.time()-t0:.1f}s)")

    # Run 5-arm shrinkage comparison
    rows = [
        {"arm": "FedAvg", "d_eff_source": "global", "final_accuracy": round(fedavg_acc, 5),
         "accuracy_auc": None, "mean_lambda": 1.0, "mean_r_g": None, "mean_r_sigma": None, "d": args.n_features},
        {"arm": "Oracle", "d_eff_source": "oracle", "final_accuracy": round(oracle_acc, 5),
         "accuracy_auc": None, "mean_lambda": 0.0, "mean_r_g": None, "mean_r_sigma": None, "d": args.n_features},
    ]

    print("\n--- Running 5-arm shrinkage ablation ---")
    for arm_name, d_eff_source in ARMS:
        t0 = time.time()
        row = run_oracle_shrinkage_arm(
            dataset,
            arm_name=arm_name,
            d_eff_source=d_eff_source,
            lr=args.lr,
            n_epochs=args.n_epochs,
            federation_every=args.federation_every,
            seed=seed,
        )
        row["seed"] = seed
        elapsed = time.time() - t0
        rows.append(row)
        print(
            f"  {arm_name:20s}  acc={row['final_accuracy']:.4f}  "
            f"λ={row['mean_lambda']:.4f}  "
            f"r^G={row['mean_r_g']}  "
            f"r_Σ={row['mean_r_sigma']}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    # Save results
    out = {
        "seed": seed,
        "K": args.K,
        "T": args.T,
        "d": args.n_features,
        "n_concepts": n_concepts,
        "rows": rows,
    }
    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Arm':20s} {'Acc':>8s} {'λ':>8s} {'r^G':>8s} {'r_Σ':>8s}")
    print(f"{'-'*70}")
    for row in rows:
        r_g_str = f"{row['mean_r_g']:.1f}" if row['mean_r_g'] is not None else "--"
        r_s_str = f"{row['mean_r_sigma']:.1f}" if row['mean_r_sigma'] is not None else "--"
        print(
            f"{row['arm']:20s} {row['final_accuracy']:>8.4f} "
            f"{row['mean_lambda']:>8.4f} {r_g_str:>8s} {r_s_str:>8s}"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
