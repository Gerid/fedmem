from __future__ import annotations

"""Granularity crossover experiment: validate Theorem 1 and Theorem 2.

Uses a controlled Gaussian linear regression setting that matches the
theory framework exactly:
  - C concepts with concept-specific linear predictors w_j*
  - Concept separation Δ = min_{i≠j} ||w_i* - w_j*||
  - K clients, each assigned to one concept per round
  - Two aggregation strategies: global (FedAvg) vs concept-level (Oracle)

Sweeps (K, C, Δ, n_samples) and records which strategy wins at each point.
Compares empirical crossover to the theoretical prediction:
  concept-level wins iff  Kn·B_j² / (σ²d) > C - 1

Outputs to ``tmp/granularity_crossover/``.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data generation (matches theory framework Section 1)
# ---------------------------------------------------------------------------

def generate_gaussian_fl_data(
    K: int,
    T: int,
    C: int,
    d: int,
    delta: float,
    sigma: float,
    n_per_client: int,
    stability_tau: int | None,
    seed: int,
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Generate federated data from Gaussian linear model with concept structure.

    Parameters
    ----------
    K : int
        Number of clients.
    T : int
        Number of time steps.
    C : int
        Number of concepts.
    d : int
        Feature dimension.
    delta : float
        Concept separation (Euclidean distance between concept-specific predictors).
    sigma : float
        Observation noise std.
    n_per_client : int
        Samples per (client, time-step).
    stability_tau : int or None
        Concept stability period. If None, concepts are fixed (stationary).
    seed : int
        Random seed.

    Returns
    -------
    concept_matrix : np.ndarray of shape (K, T)
    data : dict mapping (k, t) -> (X, y) with X: (n, d), y: (n,)
    w_stars : list of np.ndarray, the true concept-specific predictors
    """
    rng = np.random.default_rng(seed)

    # Generate concept-specific predictors spread on a hypersphere
    # with pairwise distance ≈ delta
    w_stars = []
    if C == 1:
        w_stars = [rng.standard_normal(d)]
    else:
        # Place concepts evenly in d-dimensional space with pairwise distance ≈ delta.
        # For simplex vertices: dist = r * sqrt(2 * C / (C-1))
        # So r = delta / sqrt(2 * C / (C-1))
        r = delta / np.sqrt(2.0 * C / (C - 1))
        # Generate C random directions in R^d and orthogonalise
        raw = rng.standard_normal((C, d))
        raw -= raw.mean(axis=0)
        # QR gives orthonormal rows when C <= d
        Q, _ = np.linalg.qr(raw.T)  # Q: (d, C)
        for j in range(C):
            w_stars.append(Q[:, j] * r)  # each w_star is d-dimensional

    # Verify separation
    actual_seps = []
    for i in range(C):
        for j in range(i + 1, C):
            actual_seps.append(float(np.linalg.norm(w_stars[i] - w_stars[j])))

    # Generate concept matrix
    concept_matrix = np.zeros((K, T), dtype=np.int32)
    if stability_tau is None:
        # Stationary: each client gets a fixed concept
        for k in range(K):
            concept_matrix[k, :] = k % C
    else:
        # Piecewise stationary with period tau
        for k in range(K):
            t = 0
            current_concept = k % C
            while t < T:
                end = min(t + stability_tau, T)
                concept_matrix[k, t:end] = current_concept
                current_concept = (current_concept + 1) % C
                t = end

    # Generate data
    Sigma = np.eye(d)  # isotropic covariance for simplicity
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            cid = int(concept_matrix[k, t])
            w_star = w_stars[cid]
            sample_seed = seed + k * T + t + 10000
            local_rng = np.random.default_rng(sample_seed)
            X = local_rng.standard_normal((n_per_client, d))
            # True Gaussian regression: y = X w* + epsilon
            y = X @ w_star + sigma * local_rng.standard_normal(n_per_client)
            data[(k, t)] = (X.astype(np.float32), y.astype(np.float32))

    return concept_matrix, data, w_stars


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def _ols_fit(X: np.ndarray, y: np.ndarray, d: int,
             init_w: np.ndarray | None = None,
             init_b: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form OLS estimator matching the theory (Gaussian regression).

    Returns (w, b) where w is d-dimensional and b is scalar intercept.
    Uses ridge regularization with small lambda for numerical stability.
    """
    X64 = X.astype(np.float64)
    y64 = y.astype(np.float64)
    n = len(X64)
    if n == 0:
        w = init_w.copy() if init_w is not None else np.zeros(d, dtype=np.float64)
        b = init_b.copy() if init_b is not None else np.zeros(1, dtype=np.float64)
        return w, b
    # Augment X with intercept column
    X_aug = np.hstack([X64, np.ones((n, 1), dtype=np.float64)])
    # Ridge OLS: w = (X'X + lambda*I)^{-1} X'y
    reg = 1e-6 * np.eye(d + 1, dtype=np.float64)
    w_aug = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y64)
    return w_aug[:d], w_aug[d:d+1]


def _predict_regression(X: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Predict continuous regression output y_hat = w'x + b."""
    return X.astype(np.float64) @ w + b[0]


def _mse(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error (the theory's excess risk metric)."""
    y_hat = X.astype(np.float64) @ w + b[0]
    return float(np.mean((y_hat - y.astype(np.float64)) ** 2))


def run_fedavg(
    concept_matrix: np.ndarray,
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    d: int,
    federation_every: int,
    lr: float = 0.0,
    n_epochs: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Global FedAvg: all clients aggregate into one model via OLS.

    Returns (acc_matrix, mse_matrix).
    """
    K, T = concept_matrix.shape
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    mse_matrix = np.zeros((K, T), dtype=np.float64)
    global_w = np.zeros(d, dtype=np.float64)
    global_b = np.zeros(1, dtype=np.float64)

    for t in range(T):
        uploads_w = []
        uploads_b = []
        for k in range(K):
            X, y = data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            mse_val = _mse(X_test, y_test, global_w, global_b)
            # R^2 = 1 - MSE / Var(y); use as "accuracy" proxy
            y_var = float(np.var(y_test.astype(np.float64)))
            acc_matrix[k, t] = max(1.0 - mse_val / max(y_var, 1e-12), 0.0)
            mse_matrix[k, t] = mse_val

            w_local, b_local = _ols_fit(X_train, y_train, d)
            uploads_w.append(w_local)
            uploads_b.append(b_local)

        if (t + 1) % federation_every == 0 and t < T - 1:
            global_w = np.mean(uploads_w, axis=0)
            global_b = np.mean(uploads_b, axis=0)

    return acc_matrix, mse_matrix


def run_oracle_concept(
    concept_matrix: np.ndarray,
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    d: int,
    federation_every: int,
    lr: float = 0.0,
    n_epochs: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Oracle concept-level FedAvg: aggregate within true concept groups via OLS.

    Returns (acc_matrix, mse_matrix).
    """
    K, T = concept_matrix.shape
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    mse_matrix = np.zeros((K, T), dtype=np.float64)
    concept_w: dict[int, np.ndarray] = {}
    concept_b: dict[int, np.ndarray] = {}

    for t in range(T):
        uploads_by_concept: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for k in range(K):
            X, y = data[(k, t)]
            cid = int(concept_matrix[k, t])
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            w_init = concept_w.get(cid, np.zeros(d, dtype=np.float64))
            b_init = concept_b.get(cid, np.zeros(1, dtype=np.float64))

            mse_val = _mse(X_test, y_test, w_init, b_init)
            y_var = float(np.var(y_test.astype(np.float64)))
            acc_matrix[k, t] = max(1.0 - mse_val / max(y_var, 1e-12), 0.0)
            mse_matrix[k, t] = mse_val

            w_local, b_local = _ols_fit(X_train, y_train, d)
            uploads_by_concept.setdefault(cid, []).append((w_local, b_local))

        if (t + 1) % federation_every == 0 and t < T - 1:
            for cid, uploads in uploads_by_concept.items():
                ws = [u[0] for u in uploads]
                bs = [u[1] for u in uploads]
                concept_w[cid] = np.mean(ws, axis=0)
                concept_b[cid] = np.mean(bs, axis=0)

    return acc_matrix, mse_matrix


def run_shrinkage(
    concept_matrix: np.ndarray,
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    d: int,
    federation_every: int,
    lr: float = 0.0,
    n_epochs: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrinkage estimator (Theorem 3): interpolate concept-level toward global via OLS.

    Returns (acc_matrix, mse_matrix).
    """
    K, T = concept_matrix.shape
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    mse_matrix = np.zeros((K, T), dtype=np.float64)
    concept_w: dict[int, np.ndarray] = {}
    concept_b: dict[int, np.ndarray] = {}

    for t in range(T):
        uploads_by_concept: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for k in range(K):
            X, y = data[(k, t)]
            cid = int(concept_matrix[k, t])
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            w_init = concept_w.get(cid, np.zeros(d, dtype=np.float64))
            b_init = concept_b.get(cid, np.zeros(1, dtype=np.float64))

            mse_val = _mse(X_test, y_test, w_init, b_init)
            y_var = float(np.var(y_test.astype(np.float64)))
            acc_matrix[k, t] = max(1.0 - mse_val / max(y_var, 1e-12), 0.0)
            mse_matrix[k, t] = mse_val

            w_local, b_local = _ols_fit(X_train, y_train, d)
            uploads_by_concept.setdefault(cid, []).append((w_local, b_local))

        if (t + 1) % federation_every == 0 and t < T - 1:
            # First compute concept-level averages
            agg_w: dict[int, np.ndarray] = {}
            agg_b: dict[int, np.ndarray] = {}
            for cid, uploads in uploads_by_concept.items():
                ws = [u[0] for u in uploads]
                bs = [u[1] for u in uploads]
                agg_w[cid] = np.mean(ws, axis=0)
                agg_b[cid] = np.mean(bs, axis=0)

            if len(agg_w) < 2:
                concept_w.update(agg_w)
                concept_b.update(agg_b)
                continue

            # Compute global mean
            all_cids = list(agg_w.keys())
            global_w = np.mean([agg_w[c] for c in all_cids], axis=0)
            global_b = np.mean([agg_b[c] for c in all_cids], axis=0)

            # Estimate within-concept variance from client upload dispersion
            within_vars = []
            for cid2, uploads2 in uploads_by_concept.items():
                if len(uploads2) >= 2:
                    ws2 = np.array([u[0] for u in uploads2])
                    # Per-coordinate variance of client uploads within this concept
                    within_vars.append(float(np.mean(np.var(ws2, axis=0))))
            within_var_per_dim = np.mean(within_vars) if within_vars else 1e-6

            # Between-concept variance of concept-level estimates
            between_var_per_dim = float(np.mean(np.var(
                np.array([agg_w[c] for c in all_cids]), axis=0
            )))

            # Empirical Bayes: sigma_B^2 = between - within/n_per_group
            C_active = len(all_cids)
            n_per_group = K / max(C_active, 1)
            sigma_B_sq = max(between_var_per_dim - within_var_per_dim / max(n_per_group, 1), 0.0)
            noise_var = within_var_per_dim / max(n_per_group, 1)

            # Shrinkage coefficient: lambda = noise / (noise + signal)
            if sigma_B_sq + noise_var > 1e-12:
                lam = noise_var / (noise_var + sigma_B_sq)
            else:
                lam = 0.5

            for cid in all_cids:
                concept_w[cid] = (1 - lam) * agg_w[cid] + lam * global_w
                concept_b[cid] = (1 - lam) * agg_b[cid] + lam * global_b

    return acc_matrix, mse_matrix


# ---------------------------------------------------------------------------
# Practical baselines (no oracle concept labels)
# ---------------------------------------------------------------------------

def run_ifca(
    concept_matrix: np.ndarray,
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    d: int,
    federation_every: int,
    C_hat: int = 0,
    lr: float = 0.0,
    n_epochs: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """IFCA (Ghosh et al. 2020): clients choose best cluster head by loss.

    Each round, server maintains C_hat cluster models. Each client evaluates
    all cluster models on its data and selects the best one, then trains
    locally and uploads. Server averages uploads per cluster.

    Returns (acc_matrix, mse_matrix).
    """
    K, T = concept_matrix.shape
    if C_hat == 0:
        C_hat = len(set(int(v) for v in np.unique(concept_matrix)))
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    mse_matrix = np.zeros((K, T), dtype=np.float64)

    # Initialize cluster heads with small random perturbations
    rng = np.random.default_rng(0)
    cluster_w = [rng.standard_normal(d).astype(np.float64) * 0.01 for _ in range(C_hat)]
    cluster_b = [np.zeros(1, dtype=np.float64) for _ in range(C_hat)]

    for t in range(T):
        uploads_by_cluster: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for k in range(K):
            X, y = data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            # Select best cluster by MSE on local data
            best_c = 0
            best_loss = float("inf")
            for c_idx in range(C_hat):
                loss = _mse(X_train, y_train, cluster_w[c_idx], cluster_b[c_idx])
                if loss < best_loss:
                    best_loss = loss
                    best_c = c_idx

            # Evaluate
            mse_val = _mse(X_test, y_test, cluster_w[best_c], cluster_b[best_c])
            y_var = float(np.var(y_test.astype(np.float64)))
            acc_matrix[k, t] = max(1.0 - mse_val / max(y_var, 1e-12), 0.0)
            mse_matrix[k, t] = mse_val

            # Train locally and upload
            w_local, b_local = _ols_fit(X_train, y_train, d)
            uploads_by_cluster.setdefault(best_c, []).append((w_local, b_local))

        if (t + 1) % federation_every == 0 and t < T - 1:
            for c_idx, uploads in uploads_by_cluster.items():
                ws = [u[0] for u in uploads]
                bs = [u[1] for u in uploads]
                cluster_w[c_idx] = np.mean(ws, axis=0)
                cluster_b[c_idx] = np.mean(bs, axis=0)

    return acc_matrix, mse_matrix


def run_cfl(
    concept_matrix: np.ndarray,
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    d: int,
    federation_every: int,
    C_hat: int = 0,
    lr: float = 0.0,
    n_epochs: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """CFL (Sattler et al. 2021): cluster by cosine similarity of gradient updates.

    After each federation round, server clusters client updates by cosine
    similarity and averages within clusters. Each client receives its
    cluster-specific model (multi-cluster serving).

    Returns (acc_matrix, mse_matrix).
    """
    K, T = concept_matrix.shape
    if C_hat == 0:
        C_hat = len(set(int(v) for v in np.unique(concept_matrix)))
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    mse_matrix = np.zeros((K, T), dtype=np.float64)

    # Per-client model (initially zero)
    client_w = [np.zeros(d, dtype=np.float64) for _ in range(K)]
    client_b = [np.zeros(1, dtype=np.float64) for _ in range(K)]

    for t in range(T):
        uploads_w = []
        uploads_b = []
        for k in range(K):
            X, y = data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            mse_val = _mse(X_test, y_test, client_w[k], client_b[k])
            y_var = float(np.var(y_test.astype(np.float64)))
            acc_matrix[k, t] = max(1.0 - mse_val / max(y_var, 1e-12), 0.0)
            mse_matrix[k, t] = mse_val

            w_local, b_local = _ols_fit(X_train, y_train, d)
            uploads_w.append(w_local)
            uploads_b.append(b_local)

        if (t + 1) % federation_every == 0 and t < T - 1:
            # Compute gradient updates relative to mean
            mean_w = np.mean(uploads_w, axis=0)
            deltas = np.array([w - mean_w for w in uploads_w])

            # Cluster by cosine similarity using k-means on normalized deltas
            norms = np.linalg.norm(deltas, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normalized = deltas / norms

            from sklearn.cluster import KMeans
            n_clusters = min(C_hat, K)
            if n_clusters < 2:
                for k in range(K):
                    client_w[k] = np.mean(uploads_w, axis=0)
                    client_b[k] = np.mean(uploads_b, axis=0)
                continue
            kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=t)
            labels = kmeans.fit_predict(normalized)

            # Average within clusters and assign each client its cluster model
            for c_idx in range(n_clusters):
                mask = labels == c_idx
                if mask.sum() > 0:
                    cw = np.mean([uploads_w[i] for i in range(K) if mask[i]], axis=0)
                    cb = np.mean([uploads_b[i] for i in range(K) if mask[i]], axis=0)
                    for k in range(K):
                        if mask[k]:
                            client_w[k] = cw
                            client_b[k] = cb

    return acc_matrix, mse_matrix


def run_apfl(
    concept_matrix: np.ndarray,
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    d: int,
    federation_every: int,
    alpha: float = 0.5,
    lr: float = 0.0,
    n_epochs: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """APFL (Deng et al. 2020): fixed alpha interpolation between local and global.

    Each client maintains a personal model as alpha * local + (1-alpha) * global.
    Unlike our Shrinkage, alpha is a fixed hyperparameter, not data-adaptive.

    Returns (acc_matrix, mse_matrix).
    """
    K, T = concept_matrix.shape
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    mse_matrix = np.zeros((K, T), dtype=np.float64)
    global_w = np.zeros(d, dtype=np.float64)
    global_b = np.zeros(1, dtype=np.float64)
    # Per-client personal models
    personal_w = [np.zeros(d, dtype=np.float64) for _ in range(K)]
    personal_b = [np.zeros(1, dtype=np.float64) for _ in range(K)]

    for t in range(T):
        uploads_w = []
        uploads_b = []
        for k in range(K):
            X, y = data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            # Personal model = alpha * local + (1-alpha) * global
            p_w = alpha * personal_w[k] + (1 - alpha) * global_w
            p_b = alpha * personal_b[k] + (1 - alpha) * global_b

            mse_val = _mse(X_test, y_test, p_w, p_b)
            y_var = float(np.var(y_test.astype(np.float64)))
            acc_matrix[k, t] = max(1.0 - mse_val / max(y_var, 1e-12), 0.0)
            mse_matrix[k, t] = mse_val

            # Train locally
            w_local, b_local = _ols_fit(X_train, y_train, d)
            personal_w[k] = w_local
            personal_b[k] = b_local
            uploads_w.append(w_local)
            uploads_b.append(b_local)

        if (t + 1) % federation_every == 0 and t < T - 1:
            global_w = np.mean(uploads_w, axis=0)
            global_b = np.mean(uploads_b, axis=0)

    return acc_matrix, mse_matrix


# ---------------------------------------------------------------------------
# Theoretical prediction
# ---------------------------------------------------------------------------

def theory_crossover_snr(K: int, C: int, d: int, delta: float, sigma: float, n: int) -> float:
    """Compute SNR_concept = Kn * B_j^2 / (sigma^2 * d).

    For balanced simplex placement, B_j^2 ≈ (C-1)/C^2 * delta^2.
    Crossover when SNR > C - 1.
    """
    # B_j^2 for simplex placement
    B_j_sq = (C - 1) / (C ** 2) * delta ** 2
    n_train = n // 2  # half used for training
    snr = K * n_train * B_j_sq / (sigma ** 2 * d)
    return snr


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_one_point(
    K: int, T: int, C: int, d: int, delta: float, sigma: float,
    n_per_client: int, stability_tau: int | None,
    federation_every: int, lr: float, n_epochs: int, seed: int,
) -> dict:
    """Run all methods for one parameter point."""
    concept_matrix, data, w_stars = generate_gaussian_fl_data(
        K=K, T=T, C=C, d=d, delta=delta, sigma=sigma,
        n_per_client=n_per_client, stability_tau=stability_tau, seed=seed,
    )

    C_true = len(set(int(v) for v in np.unique(concept_matrix)))

    fedavg_acc, fedavg_mse = run_fedavg(concept_matrix, data, d, federation_every)
    oracle_acc, oracle_mse = run_oracle_concept(concept_matrix, data, d, federation_every)
    shrink_acc, shrink_mse = run_shrinkage(concept_matrix, data, d, federation_every)
    ifca_acc, ifca_mse = run_ifca(concept_matrix, data, d, federation_every, C_hat=C_true)
    cfl_acc, cfl_mse = run_cfl(concept_matrix, data, d, federation_every, C_hat=C_true)
    # APFL: sweep alpha in {0.2, 0.5, 0.8}, pick best by final MSE
    best_apfl_acc, best_apfl_mse = None, None
    best_apfl_final_mse = float("inf")
    for alpha_val in [0.2, 0.5, 0.8]:
        a_acc, a_mse = run_apfl(concept_matrix, data, d, federation_every, alpha=alpha_val)
        final_mse = float(a_mse[:, -1].mean())
        if final_mse < best_apfl_final_mse:
            best_apfl_final_mse = final_mse
            best_apfl_acc, best_apfl_mse = a_acc, a_mse
    apfl_acc, apfl_mse = best_apfl_acc, best_apfl_mse

    snr = theory_crossover_snr(K, C, d, delta, sigma, n_per_client)
    theory_wins = snr > (C - 1)

    def _final_mse(mse_mat: np.ndarray) -> float:
        return round(float(mse_mat[:, -1].mean()), 6)

    def _final_r2(acc_mat: np.ndarray) -> float:
        return round(float(acc_mat[:, -1].mean()), 4)

    return {
        "K": K, "C": C, "d": d, "delta": round(delta, 3),
        "sigma": sigma, "n": n_per_client,
        "tau": stability_tau if stability_tau else "inf",
        "K_over_C": round(K / C, 2), "seed": seed,
        "SNR_concept": round(snr, 4),
        "theory_oracle_wins": theory_wins,
        "fedavg_final": _final_r2(fedavg_acc),
        "oracle_final": _final_r2(oracle_acc),
        "shrink_final": _final_r2(shrink_acc),
        "ifca_final": _final_r2(ifca_acc),
        "cfl_final": _final_r2(cfl_acc),
        "apfl_final": _final_r2(apfl_acc),
        "fedavg_mse": _final_mse(fedavg_mse),
        "oracle_mse": _final_mse(oracle_mse),
        "shrink_mse": _final_mse(shrink_mse),
        "ifca_mse": _final_mse(ifca_mse),
        "cfl_mse": _final_mse(cfl_mse),
        "apfl_mse": _final_mse(apfl_mse),
        "oracle_advantage": round(float(oracle_acc[:, -1].mean() - fedavg_acc[:, -1].mean()), 4),
        "shrink_advantage": round(float(shrink_acc[:, -1].mean() - fedavg_acc[:, -1].mean()), 4),
        "oracle_mse_advantage": round(float(fedavg_mse[:, -1].mean() - oracle_mse[:, -1].mean()), 6),
        "empirical_oracle_wins": bool(float(oracle_mse[:, -1].mean()) < float(fedavg_mse[:, -1].mean())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Granularity crossover: validate Theorems 1-3")
    parser.add_argument("--results-dir", default="tmp/granularity_crossover")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--d", type=int, default=20, help="Feature dimension")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise std")
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per client per step")
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--smoke", action="store_true", help="Quick test with fewer points")
    # Compatibility with k8s submit template (unused for synthetic data)
    parser.add_argument("--data-root", default=None, help="(unused, k8s compat)")
    parser.add_argument("--feature-cache-dir", default=None, help="(unused, k8s compat)")
    parser.add_argument("--n-workers", type=int, default=0, help="(unused, k8s compat)")
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sweep parameters
    if args.smoke:
        K_vals = [10, 20]
        C_vals = [2, 4]
        delta_vals = [0.5, 2.0, 5.0]
        tau_vals = [None]  # stationary only for smoke
    else:
        K_vals = [10, 20, 40]
        C_vals = [2, 4, 8]
        delta_vals = [0.3, 1.0, 3.0, 8.0]
        tau_vals = [None, 5, 15]  # None = stationary

    total_points = len(K_vals) * len(C_vals) * len(delta_vals) * len(tau_vals) * len(args.seeds)
    print(f"Total sweep points: {total_points}")
    print(f"Params: K={K_vals}, C={C_vals}, delta={delta_vals}, tau={tau_vals}")
    print(f"d={args.d}, sigma={args.sigma}, n={args.n_samples}, T={args.T}")
    print()

    rows: list[dict] = []
    idx = 0
    for tau in tau_vals:
        for K in K_vals:
            for C in C_vals:
                if C > K:
                    continue  # need at least 1 client per concept
                for delta in delta_vals:
                    for seed in args.seeds:
                        idx += 1
                        print(f"[{idx}/{total_points}] K={K} C={C} delta={delta} tau={tau} seed={seed}", end="")
                        try:
                            row = run_one_point(
                                K=K, T=args.T, C=C, d=args.d, delta=delta, sigma=args.sigma,
                                n_per_client=args.n_samples, stability_tau=tau,
                                federation_every=args.federation_every,
                                lr=args.lr, n_epochs=args.n_epochs, seed=seed,
                            )
                            rows.append(row)
                            mark = "Y" if row["empirical_oracle_wins"] == row["theory_oracle_wins"] else "N"
                            print(f"  FedAvg={row['fedavg_final']:.3f} Oracle={row['oracle_final']:.3f} "
                                  f"Shrink={row['shrink_final']:.3f} IFCA={row['ifca_final']:.3f} "
                                  f"CFL={row['cfl_final']:.3f} APFL={row['apfl_final']:.3f} "
                                  f"SNR={row['SNR_concept']:.2f} {mark}")
                        except Exception as e:
                            print(f"  ERROR: {e}")

    # Save raw results
    csv_path = out_dir / "crossover_results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {csv_path}")

    # Aggregate: for each (K, C, delta, tau), average over seeds
    agg: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["K"], r["C"], r["delta"], r["tau"])
        agg.setdefault(key, []).append(r)

    summary_rows = []
    theory_correct = 0
    theory_total = 0
    def _sort_key(item):
        k, c, d, tau = item[0]
        return (k, c, d, 9999 if tau == "inf" else tau)

    for key, group in sorted(agg.items(), key=_sort_key):
        K, C, delta, tau = key
        n_seeds = len(group)
        methods = ["fedavg", "oracle", "shrink", "ifca", "cfl", "apfl"]
        avgs = {}
        for m in methods:
            avgs[f"{m}_r2"] = float(np.mean([r[f"{m}_final"] for r in group]))
            avgs[f"{m}_mse"] = float(np.mean([r[f"{m}_mse"] for r in group]))

        snr = group[0]["SNR_concept"]
        theory_wins = group[0]["theory_oracle_wins"]
        empirical_wins = avgs["oracle_mse"] < avgs["fedavg_mse"]

        match = theory_wins == empirical_wins
        theory_total += 1
        if match:
            theory_correct += 1

        row_data = {
            "K": K, "C": C, "delta": delta, "tau": tau,
            "K_over_C": round(K / C, 2),
            "SNR_concept": round(snr, 3),
        }
        for m in methods:
            row_data[f"{m}_r2"] = round(avgs[f"{m}_r2"], 4)
            row_data[f"{m}_mse"] = round(avgs[f"{m}_mse"], 6)
        row_data.update({
            "oracle_advantage": round(avgs["oracle_r2"] - avgs["fedavg_r2"], 4),
            "theory_oracle_wins": theory_wins,
            "empirical_oracle_wins": bool(empirical_wins),
            "theory_correct": bool(match),
            "n_seeds": n_seeds,
        })
        summary_rows.append(row_data)

    # Print summary table
    print("\n" + "=" * 120)
    print("AGGREGATED CROSSOVER RESULTS (averaged over seeds)")
    print("=" * 120)
    hdr = (f"{'K':>4} {'C':>3} {'K/C':>5} {'delta':>6} {'tau':>5} {'SNR':>7} "
           f"{'FedAvg':>8} {'Oracle':>8} {'Shrink':>8} {'IFCA':>8} {'CFL':>8} {'APFL':>8} {'Match':>6}")
    print(hdr)
    print("-" * 140)
    for r in summary_rows:
        tau_str = str(r["tau"]) if r["tau"] != "inf" else "inf"
        match_str = "Y" if r["theory_correct"] else "N"
        print(f"{r['K']:4d} {r['C']:3d} {r['K_over_C']:5.1f} {r['delta']:6.2f} {tau_str:>5} "
              f"{r['SNR_concept']:7.2f} {r['fedavg_r2']:8.4f} {r['oracle_r2']:8.4f} "
              f"{r['shrink_r2']:8.4f} {r['ifca_r2']:8.4f} {r['cfl_r2']:8.4f} "
              f"{r['apfl_r2']:8.4f} {match_str:>6}")

    accuracy = theory_correct / max(theory_total, 1)
    print(f"\nTheory prediction accuracy: {theory_correct}/{theory_total} = {accuracy:.1%}")

    # Best method per row (by MSE, lower is better)
    method_names = {"fedavg": "FedAvg", "oracle": "Oracle", "shrink": "Shrinkage",
                    "ifca": "IFCA", "cfl": "CFL", "apfl": "APFL"}
    best_counts = {v: 0 for v in method_names.values()}
    for r in summary_rows:
        mses = {method_names[m]: r[f"{m}_mse"] for m in method_names}
        best = min(mses, key=mses.get)
        best_counts[best] += 1
    print(f"Best method counts (by MSE): {best_counts}")

    # Save summary
    summary = {
        "theory_prediction_accuracy": round(accuracy, 4),
        "total_points": theory_total,
        "best_method_counts": best_counts,
        "params": {
            "d": args.d, "sigma": args.sigma, "n": args.n_samples,
            "T": args.T, "federation_every": args.federation_every,
        },
        "rows": summary_rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
