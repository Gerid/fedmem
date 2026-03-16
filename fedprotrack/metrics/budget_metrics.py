from __future__ import annotations

import numpy as np


def compute_accuracy_auc(accuracy_curve: np.ndarray) -> float:
    """Compute the area under the mean-accuracy curve using the trapezoidal rule.

    Parameters
    ----------
    accuracy_curve : np.ndarray
        Array of shape (K, T) with per-client per-step classification accuracy.

    Returns
    -------
    float
        Trapezoidal AUC of the mean accuracy across clients over time steps 0..T-1.
    """
    mean_acc: np.ndarray = accuracy_curve.mean(axis=0)  # shape (T,)
    return float(np.trapz(mean_acc))


def budget_normalized_score(
    accuracy_curve: np.ndarray,
    total_bytes: float,
) -> float:
    """Compute communication-budget-normalised accuracy AUC.

    Parameters
    ----------
    accuracy_curve : np.ndarray
        Array of shape (K, T) with per-client per-step classification accuracy.
    total_bytes : float
        Total communication budget in bytes consumed by the algorithm.

    Returns
    -------
    float
        AUC of mean accuracy divided by ``total_bytes``.

    Raises
    ------
    ValueError
        If ``total_bytes`` <= 0.
    ValueError
        If ``accuracy_curve`` has fewer than 2 time steps (T < 2).
    """
    if total_bytes <= 0:
        raise ValueError(f"total_bytes must be positive, got {total_bytes}")

    T = accuracy_curve.shape[1] if accuracy_curve.ndim > 1 else accuracy_curve.shape[0]
    if accuracy_curve.ndim < 2 or accuracy_curve.shape[1] < 2:
        raise ValueError(
            f"accuracy_curve must have at least 2 time steps (T >= 2), "
            f"got shape {accuracy_curve.shape}"
        )

    auc = compute_accuracy_auc(accuracy_curve)
    return auc / total_bytes


def fedavg_total_bytes(
    K: int,
    T: int,
    n_params: int,
    precision_bits: int = 32,
) -> float:
    """Estimate total bytes communicated by FedAvg over T rounds.

    Each round every client uploads its model to the server and then receives the
    aggregated model back (upload + download = factor 2).

    Parameters
    ----------
    K : int
        Number of federated clients.
    T : int
        Number of communication rounds (time steps).
    n_params : int
        Number of model parameters.
    precision_bits : int
        Bit-width of each parameter (default 32-bit float).

    Returns
    -------
    float
        Total bytes communicated: T * K * 2 * n_params * (precision_bits / 8).
    """
    bytes_per_param = precision_bits / 8
    return float(T * K * 2 * n_params * bytes_per_param)
