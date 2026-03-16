from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.metrics.budget_metrics import (
    budget_normalized_score,
    compute_accuracy_auc,
    fedavg_total_bytes,
)


# ---------------------------------------------------------------------------
# compute_accuracy_auc tests
# ---------------------------------------------------------------------------


def test_constant_accuracy_auc():
    """Constant accuracy=1.0 for K clients over T=5 steps → AUC = 4.0.

    np.trapz over [1, 1, 1, 1, 1] with unit spacing = 4.0.
    """
    K, T = 3, 5
    acc = np.ones((K, T), dtype=np.float64)
    auc = compute_accuracy_auc(acc)
    assert abs(auc - 4.0) < 1e-9


def test_zero_then_one():
    """Accuracy rises linearly from 0 to 1 over T=5 steps → AUC = 2.0.

    Mean across K=2 clients: [0, 0.25, 0.5, 0.75, 1.0] (same for both clients).
    np.trapz = 0.5 * (0+0.25) + 0.5*(0.25+0.5) + 0.5*(0.5+0.75) + 0.5*(0.75+1.0) = 2.0.
    """
    K, T = 2, 5
    # Both clients have identical linearly increasing accuracy
    row = np.linspace(0.0, 1.0, T)
    acc = np.tile(row, (K, 1)).astype(np.float64)
    auc = compute_accuracy_auc(acc)
    assert abs(auc - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# budget_normalized_score tests
# ---------------------------------------------------------------------------


def test_budget_normalized():
    """Known AUC divided by known bytes gives expected ratio."""
    K, T = 2, 5
    acc = np.ones((K, T), dtype=np.float64)
    total_bytes = 8.0
    # AUC = 4.0 (constant 1.0 over T=5)
    score = budget_normalized_score(acc, total_bytes)
    assert abs(score - 4.0 / 8.0) < 1e-9


def test_zero_bytes_raises():
    """total_bytes=0 must raise ValueError."""
    acc = np.ones((2, 5), dtype=np.float64)
    with pytest.raises(ValueError):
        budget_normalized_score(acc, 0.0)


def test_negative_bytes_raises():
    """Negative total_bytes must raise ValueError."""
    acc = np.ones((2, 5), dtype=np.float64)
    with pytest.raises(ValueError):
        budget_normalized_score(acc, -1.0)


def test_single_timestep_raises():
    """T=1 has no time dimension for integration → ValueError."""
    acc = np.ones((2, 1), dtype=np.float64)
    with pytest.raises(ValueError):
        budget_normalized_score(acc, 10.0)


# ---------------------------------------------------------------------------
# fedavg_total_bytes tests
# ---------------------------------------------------------------------------


def test_fedavg_budget():
    """Verify fedavg_total_bytes against the formula T*K*2*n_params*(bits/8)."""
    K, T, n_params, bits = 5, 10, 1000, 32
    expected = T * K * 2 * n_params * (bits / 8)
    result = fedavg_total_bytes(K, T, n_params, bits)
    assert result == expected


def test_fedavg_budget_16bit():
    """16-bit precision halves the byte count compared to 32-bit."""
    K, T, n_params = 4, 20, 500
    bytes_32 = fedavg_total_bytes(K, T, n_params, precision_bits=32)
    bytes_16 = fedavg_total_bytes(K, T, n_params, precision_bits=16)
    assert abs(bytes_32 / bytes_16 - 2.0) < 1e-9


def test_fedavg_budget_default_precision():
    """Default precision is 32 bits."""
    K, T, n_params = 3, 5, 100
    result_default = fedavg_total_bytes(K, T, n_params)
    result_32 = fedavg_total_bytes(K, T, n_params, precision_bits=32)
    assert result_default == result_32
