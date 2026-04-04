"""Shrinkage estimators for federated concept-level aggregation.

Implements both isotropic and anisotropic empirical-Bayes shrinkage
estimators following the theory in the paper (Proposition 5 / Theorem 6).

The key insight: isotropic shrinkage uses raw dimension ``d`` in the
variance formula, but when features are anisotropic (e.g., frozen
pretrained features), the effective dimension ``r_eff`` should be used
instead.  This can close a 300x gap between theory-predicted and
empirically optimal shrinkage coefficients on real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_effective_rank(X: np.ndarray) -> float:
    """Compute effective rank of a feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape ``(n_samples, d)``.

    Returns
    -------
    float
        Effective rank ``r_eff = (sum lambda_i)^2 / sum lambda_i^2``,
        where ``lambda_i`` are eigenvalues of the sample covariance.

    Raises
    ------
    ValueError
        If *X* has fewer than 2 samples or 1 feature.
    """
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(
            f"X must be (n_samples >= 2, d >= 1), got {X.shape}"
        )
    # Centre the data
    X_centered = X - X.mean(axis=0, keepdims=True)
    # Use SVD for numerical stability (avoids forming d×d covariance)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = s ** 2 / (X.shape[0] - 1)
    # Discard negligible eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    trace = eigenvalues.sum()
    frobenius_sq = (eigenvalues ** 2).sum()
    if frobenius_sq < 1e-30:
        return 1.0
    return float(trace ** 2 / frobenius_sq)


def compute_effective_rank_from_covariance(
    cov: np.ndarray,
) -> float:
    """Compute effective rank from a precomputed covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of shape ``(d, d)``.

    Returns
    -------
    float
        Effective rank.
    """
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    trace = eigenvalues.sum()
    frobenius_sq = (eigenvalues ** 2).sum()
    if frobenius_sq < 1e-30:
        return 1.0
    return float(trace ** 2 / frobenius_sq)


def compute_shrinkage_lambda(
    sigma2: float,
    sigma_B2: float,
    K: int,
    C: int,
    n: int,
    d_eff: float,
) -> float:
    """Compute the empirical-Bayes shrinkage coefficient.

    Parameters
    ----------
    sigma2 : float
        Noise variance estimate.
    sigma_B2 : float
        Between-concept variance estimate.
    K : int
        Total number of clients.
    C : int
        Number of concepts.
    n : int
        Samples per client.
    d_eff : float
        Effective dimension (use raw ``d`` for isotropic, ``r_eff`` for
        anisotropic).

    Returns
    -------
    float
        Shrinkage coefficient ``lambda`` in ``[0, 1]``.
    """
    n_concept = K / C * n  # effective samples per concept
    if n_concept <= 0 or d_eff <= 0:
        return 0.5
    variance_term = sigma2 * d_eff / n_concept
    denominator = variance_term + sigma_B2
    if denominator < 1e-30:
        return 0.5
    return float(np.clip(variance_term / denominator, 0.0, 1.0))


def estimate_sigma_B2(
    concept_estimates: list[np.ndarray],
    sigma2: float,
    K: int,
    C: int,
    n: int,
) -> float:
    """Estimate between-concept variance from concept-level estimates.

    Parameters
    ----------
    concept_estimates : list[np.ndarray]
        Per-concept weight vectors, each of shape ``(d,)`` or ``(d, n_cls)``.
    sigma2 : float
        Noise variance.
    K : int
        Total clients.
    C : int
        Number of concepts.
    n : int
        Samples per client.

    Returns
    -------
    float
        Non-negative estimate of sigma_B^2.
    """
    if len(concept_estimates) < 2:
        return 0.0
    stacked = np.stack([w.ravel() for w in concept_estimates])
    global_mean = stacked.mean(axis=0)
    d = stacked.shape[1]
    n_concept = K / C * n
    # Unbiased estimator: (1/((C-1)*d)) * sum ||w_j - w_bar||^2 - sigma^2/(n_concept)
    spread = np.sum((stacked - global_mean) ** 2) / ((C - 1) * d)
    correction = sigma2 / n_concept if n_concept > 0 else 0.0
    return float(max(spread - correction, 0.0))


@dataclass
class ShrinkageResult:
    """Result from a shrinkage estimation."""

    shrunk_estimates: list[np.ndarray]
    lambda_value: float
    lambda_iso: float
    lambda_aniso: float
    r_eff: float
    sigma_B2: float
    sigma2: float


class ShrinkageEstimator:
    """Empirical-Bayes shrinkage between global and concept-level estimates.

    Supports both isotropic (uses raw ``d``) and anisotropic (uses
    ``r_eff``) shrinkage.

    Parameters
    ----------
    use_anisotropic : bool
        If True, compute effective rank from features and use ``r_eff``
        instead of ``d`` in the shrinkage formula.
    """

    def __init__(self, use_anisotropic: bool = True) -> None:
        self.use_anisotropic = use_anisotropic

    def fit_predict(
        self,
        concept_estimates: list[np.ndarray],
        global_estimate: np.ndarray,
        sigma2: float,
        K: int,
        C: int,
        n: int,
        feature_matrix: np.ndarray | None = None,
    ) -> ShrinkageResult:
        """Compute shrunk estimates.

        Parameters
        ----------
        concept_estimates : list[np.ndarray]
            Per-concept estimates (weight vectors).
        global_estimate : np.ndarray
            Global pooled estimate.
        sigma2 : float
            Noise variance.
        K, C, n : int
            Federation parameters.
        feature_matrix : np.ndarray | None
            Full feature matrix for computing ``r_eff``.  Required when
            ``use_anisotropic=True``.

        Returns
        -------
        ShrinkageResult
        """
        d = concept_estimates[0].ravel().shape[0]
        sigma_B2 = estimate_sigma_B2(concept_estimates, sigma2, K, C, n)

        # Isotropic lambda (using raw d)
        lambda_iso = compute_shrinkage_lambda(sigma2, sigma_B2, K, C, n, d)

        # Anisotropic lambda (using r_eff)
        if feature_matrix is not None:
            r_eff = compute_effective_rank(feature_matrix)
        else:
            r_eff = float(d)
        lambda_aniso = compute_shrinkage_lambda(
            sigma2, sigma_B2, K, C, n, r_eff
        )

        # Choose which lambda to use
        lam = lambda_aniso if self.use_anisotropic else lambda_iso

        # Apply shrinkage: w_shrunk = (1 - lambda) * w_concept + lambda * w_global
        global_flat = global_estimate.ravel()
        shrunk = []
        for w in concept_estimates:
            shape = w.shape
            w_flat = w.ravel()
            w_shrunk = (1 - lam) * w_flat + lam * global_flat
            shrunk.append(w_shrunk.reshape(shape))

        return ShrinkageResult(
            shrunk_estimates=shrunk,
            lambda_value=lam,
            lambda_iso=lambda_iso,
            lambda_aniso=lambda_aniso,
            r_eff=r_eff,
            sigma_B2=sigma_B2,
            sigma2=sigma2,
        )

    def grid_search_lambda(
        self,
        concept_estimates: list[np.ndarray],
        global_estimate: np.ndarray,
        true_weights: list[np.ndarray],
        n_grid: int = 51,
    ) -> tuple[float, float]:
        """Grid search for optimal lambda (oracle, for validation only).

        Parameters
        ----------
        concept_estimates : list[np.ndarray]
            Per-concept OLS estimates.
        global_estimate : np.ndarray
            Global OLS estimate.
        true_weights : list[np.ndarray]
            Ground-truth concept weights (for oracle evaluation).
        n_grid : int
            Number of grid points in [0, 1].

        Returns
        -------
        lambda_star : float
            Optimal lambda.
        best_mse : float
            MSE at optimal lambda.
        """
        lambdas = np.linspace(0, 1, n_grid)
        global_flat = global_estimate.ravel()
        best_lam = 0.5
        best_mse = float("inf")
        for lam in lambdas:
            total_mse = 0.0
            for w_hat, w_true in zip(concept_estimates, true_weights):
                w_shrunk = (1 - lam) * w_hat.ravel() + lam * global_flat
                total_mse += np.sum((w_shrunk - w_true.ravel()) ** 2)
            avg_mse = total_mse / len(concept_estimates)
            if avg_mse < best_mse:
                best_mse = avg_mse
                best_lam = float(lam)
        return best_lam, best_mse
