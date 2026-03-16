from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.metrics.drift_metrics import find_drift_points, worst_window_dip_recovery


# ---------------------------------------------------------------------------
# find_drift_points tests
# ---------------------------------------------------------------------------


def test_find_drift_points_sync():
    """Drift at the same timestep for all clients is detected for every client."""
    K, T = 3, 6
    gt = np.zeros((K, T), dtype=np.int32)
    # All clients drift at t=3
    gt[:, 3:] = 1
    points = find_drift_points(gt)
    assert len(points) == K
    for k in range(K):
        assert (k, 3) in points


def test_find_drift_points_async():
    """Drift at different timesteps per client is detected independently."""
    K, T = 3, 8
    gt = np.zeros((K, T), dtype=np.int32)
    drift_times = {0: 2, 1: 4, 2: 6}
    for k, t in drift_times.items():
        gt[k, t:] = 1
    points = find_drift_points(gt)
    assert len(points) == K
    for k, t in drift_times.items():
        assert (k, t) in points


def test_find_drift_points_no_drift():
    """Constant ground truth yields no drift points."""
    gt = np.ones((4, 10), dtype=np.int32)
    assert find_drift_points(gt) == []


# ---------------------------------------------------------------------------
# worst_window_dip_recovery tests
# ---------------------------------------------------------------------------


def test_no_drift_returns_zero():
    """Constant concept matrix → no drift → (0.0, 0) returned."""
    K, T = 4, 10
    gt = np.zeros((K, T), dtype=np.int32)
    acc = np.full((K, T), 0.85, dtype=np.float64)
    dip, rec = worst_window_dip_recovery(acc, gt)
    assert dip == 0.0
    assert rec == 0


def test_t1_no_drift():
    """T=1 has no drift possible → (0.0, 0)."""
    K = 3
    gt = np.zeros((K, 1), dtype=np.int32)
    acc = np.full((K, 1), 0.9, dtype=np.float64)
    dip, rec = worst_window_dip_recovery(acc, gt)
    assert dip == 0.0
    assert rec == 0


def test_known_dip_at_known_position():
    """Drift at t=3 with accuracy drop from 0.9 to 0.5 → dip ≈ 0.4."""
    K, T = 2, 10
    # Both clients drift at t=3
    gt = np.zeros((K, T), dtype=np.int32)
    gt[:, 3:] = 1

    # Accuracy: 0.9 before drift, drops to 0.5 at drift, recovers after
    acc = np.full((K, T), 0.9, dtype=np.float64)
    acc[:, 3] = 0.5   # drop at drift step
    acc[:, 4] = 0.6
    acc[:, 5] = 0.9   # recovered

    dip, rec = worst_window_dip_recovery(acc, gt, window_size=3)

    # pre-drift mean = 0.9, post-drift min = 0.5 → dip = 0.4
    assert abs(dip - 0.4) < 1e-9
    # recovery: mean_acc at t=3 is 0.5 (<0.855), t=4 is 0.6 (<0.855), t=5 is 0.9 (≥0.855)
    # recovery steps = 5 - 3 = 2
    assert rec == 2


def test_no_recovery_within_window():
    """If accuracy never recovers within window*2 steps, recovery = window_size*2."""
    K, T = 2, 20
    gt = np.zeros((K, T), dtype=np.int32)
    gt[:, 3:] = 1

    # Accuracy: 0.9 before t=3, stays at 0.3 forever after
    acc = np.full((K, T), 0.9, dtype=np.float64)
    acc[:, 3:] = 0.3

    window_size = 3
    dip, rec = worst_window_dip_recovery(acc, gt, window_size=window_size)

    assert dip > 0.0
    assert rec == window_size * 2


def test_none_accuracy_raises():
    """Passing None as accuracy_curve must raise ValueError."""
    gt = np.zeros((2, 5), dtype=np.int32)
    with pytest.raises(ValueError):
        worst_window_dip_recovery(None, gt)


def test_drift_at_last_step_ignored():
    """Drift at t=T-1 has no post-drift window; dip and recovery should be 0."""
    K, T = 2, 4
    gt = np.zeros((K, T), dtype=np.int32)
    gt[:, T - 1] = 1  # drift only at very last step

    acc = np.full((K, T), 0.9, dtype=np.float64)
    acc[:, T - 1] = 0.1  # big drop, but should be ignored

    dip, rec = worst_window_dip_recovery(acc, gt, window_size=3)
    assert dip == 0.0
    assert rec == 0
