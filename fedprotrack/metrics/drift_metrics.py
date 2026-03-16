from __future__ import annotations

import numpy as np


def find_drift_points(ground_truth: np.ndarray) -> list[tuple[int, int]]:
    """Return all (client_idx, timestep) pairs where a concept drift occurs.

    A drift at (k, t) is defined as ground_truth[k, t] != ground_truth[k, t-1],
    for t >= 1.

    Parameters
    ----------
    ground_truth : np.ndarray
        Integer array of shape (K, T) containing ground-truth concept IDs.

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of (client_idx, timestep) tuples where drift occurs.
    """
    K, T = ground_truth.shape
    drift_points: list[tuple[int, int]] = []
    for k in range(K):
        for t in range(1, T):
            if ground_truth[k, t] != ground_truth[k, t - 1]:
                drift_points.append((k, t))
    return drift_points


def worst_window_dip_recovery(
    accuracy_curve: np.ndarray | None,
    ground_truth: np.ndarray,
    window_size: int = 3,
) -> tuple[float, int]:
    """Compute the worst post-drift accuracy dip and recovery time.

    For each concept drift event the function computes:
    - The accuracy *dip*: how much the mean accuracy drops immediately after drift.
    - The *recovery time*: how many steps after drift until accuracy is back to
      95 % of its pre-drift level (capped at ``window_size * 2``).

    Parameters
    ----------
    accuracy_curve : np.ndarray | None
        Array of shape (K, T) with per-client per-step classification accuracy.
        Must not be None.
    ground_truth : np.ndarray
        Integer array of shape (K, T) with ground-truth concept IDs.
    window_size : int
        Number of steps after drift to look for the post-drift minimum.

    Returns
    -------
    tuple[float, int]
        ``(worst_dip, worst_recovery_time)`` across all drift events.
        Returns ``(0.0, 0)`` when no drift events exist.

    Raises
    ------
    ValueError
        If ``accuracy_curve`` is None.
    """
    if accuracy_curve is None:
        raise ValueError("accuracy_curve must not be None")

    K, T = ground_truth.shape

    # Edge case: single timestep — no drift possible
    if T == 1:
        return (0.0, 0)

    drift_points = find_drift_points(ground_truth)

    if not drift_points:
        return (0.0, 0)

    # Mean accuracy across clients at each timestep
    mean_acc: np.ndarray = accuracy_curve.mean(axis=0)  # shape (T,)

    worst_dip: float = 0.0
    worst_recovery: int = 0

    max_recovery = window_size * 2

    for k, t_drift in drift_points:
        # Drift at the very last timestep: no post-drift window
        if t_drift >= T - 1:
            continue

        pre_drift_acc: float = float(mean_acc[t_drift - 1]) if t_drift > 0 else 0.0

        # Post-drift minimum within window
        post_end = min(t_drift + window_size, T)
        post_drift_min: float = float(mean_acc[t_drift:post_end].min())

        dip: float = max(0.0, pre_drift_acc - post_drift_min)

        # Recovery time: steps after t_drift until mean_acc >= pre_drift_acc * 0.95
        threshold = pre_drift_acc * 0.95
        recovery: int = max_recovery  # default: did not recover
        for step in range(t_drift, min(t_drift + max_recovery, T)):
            if mean_acc[step] >= threshold:
                recovery = step - t_drift
                break

        if dip > worst_dip:
            worst_dip = dip
        if recovery > worst_recovery:
            worst_recovery = recovery

    return (worst_dip, worst_recovery)
