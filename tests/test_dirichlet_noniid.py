from __future__ import annotations

"""Tests for Dirichlet-based non-IID label heterogeneity control.

Validates that:
- ``dirichlet_alpha=None`` (default) preserves existing balanced behaviour.
- Low alpha (0.01) produces highly skewed label distributions.
- High alpha (100.0) produces near-uniform label distributions.
- Seeding is deterministic and reproducible.
- Config validation rejects non-positive alpha.
"""

import numpy as np
import pytest

from fedprotrack.real_data.cifar100_recurrence import (
    CIFAR100RecurrenceConfig,
    _draw_balanced_batch,
    _draw_dirichlet_batch,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_synthetic_pool(
    n_classes: int = 5,
    samples_per_class: int = 200,
    n_features: int = 8,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic feature pool with exactly balanced classes."""
    rng = np.random.RandomState(seed)
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for cls in range(n_classes):
        X_parts.append(rng.randn(samples_per_class, n_features).astype(np.float32))
        y_parts.append(np.full(samples_per_class, cls, dtype=np.int64))
    return np.concatenate(X_parts), np.concatenate(y_parts)


# ------------------------------------------------------------------
# Batch-level tests
# ------------------------------------------------------------------

class TestDrawDirichletBatch:
    """Unit tests for the ``_draw_dirichlet_batch`` function."""

    def test_output_shapes(self) -> None:
        X_pool, y_pool = _make_synthetic_pool()
        rng = np.random.RandomState(42)
        X_batch, y_batch = _draw_dirichlet_batch(
            X_pool, y_pool, 100, 1.0, rng,
        )
        assert X_batch.shape == (100, 8)
        assert y_batch.shape == (100,)

    def test_counts_sum_to_n_samples(self) -> None:
        X_pool, y_pool = _make_synthetic_pool()
        for alpha in [0.01, 0.1, 0.5, 1.0, 10.0, 100.0]:
            rng = np.random.RandomState(42)
            X_batch, y_batch = _draw_dirichlet_batch(
                X_pool, y_pool, 200, alpha, rng,
            )
            assert len(y_batch) == 200, f"alpha={alpha}"

    def test_low_alpha_produces_skewed_distribution(self) -> None:
        """With alpha=0.01 most samples should come from 1-2 classes."""
        X_pool, y_pool = _make_synthetic_pool(n_classes=10, samples_per_class=500)
        skew_scores: list[float] = []
        for trial in range(20):
            rng = np.random.RandomState(trial)
            _, y_batch = _draw_dirichlet_batch(
                X_pool, y_pool, 200, 0.01, rng,
            )
            _, counts = np.unique(y_batch, return_counts=True)
            # Fraction of samples in the dominant class.
            skew_scores.append(counts.max() / len(y_batch))
        # On average the dominant class should hold > 50% of samples.
        assert np.mean(skew_scores) > 0.50

    def test_high_alpha_produces_near_uniform_distribution(self) -> None:
        """With alpha=100.0 each class should get roughly equal counts."""
        X_pool, y_pool = _make_synthetic_pool(n_classes=5, samples_per_class=500)
        max_deviations: list[float] = []
        for trial in range(20):
            rng = np.random.RandomState(trial)
            _, y_batch = _draw_dirichlet_batch(
                X_pool, y_pool, 500, 100.0, rng,
            )
            _, counts = np.unique(y_batch, return_counts=True)
            expected = 500 / 5
            max_deviations.append(np.max(np.abs(counts - expected)) / expected)
        # Maximum deviation should be small (< 15% of expected per class).
        assert np.mean(max_deviations) < 0.15

    def test_deterministic_seeding(self) -> None:
        X_pool, y_pool = _make_synthetic_pool()
        rng1 = np.random.RandomState(12345)
        X1, y1 = _draw_dirichlet_batch(X_pool, y_pool, 100, 0.5, rng1)
        rng2 = np.random.RandomState(12345)
        X2, y2 = _draw_dirichlet_batch(X_pool, y_pool, 100, 0.5, rng2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(X1, X2)

    def test_different_seeds_produce_different_batches(self) -> None:
        X_pool, y_pool = _make_synthetic_pool()
        rng1 = np.random.RandomState(1)
        _, y1 = _draw_dirichlet_batch(X_pool, y_pool, 100, 0.5, rng1)
        rng2 = np.random.RandomState(2)
        _, y2 = _draw_dirichlet_batch(X_pool, y_pool, 100, 0.5, rng2)
        # Extremely unlikely to be identical with different seeds.
        assert not np.array_equal(y1, y2)


# ------------------------------------------------------------------
# Config validation tests
# ------------------------------------------------------------------

class TestConfigValidation:
    """Ensure CIFAR100RecurrenceConfig validates ``dirichlet_alpha``."""

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="dirichlet_alpha must be > 0"):
            CIFAR100RecurrenceConfig(dirichlet_alpha=0.0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="dirichlet_alpha must be > 0"):
            CIFAR100RecurrenceConfig(dirichlet_alpha=-1.0)

    def test_accepts_none(self) -> None:
        cfg = CIFAR100RecurrenceConfig(dirichlet_alpha=None)
        assert cfg.dirichlet_alpha is None

    def test_accepts_positive(self) -> None:
        cfg = CIFAR100RecurrenceConfig(dirichlet_alpha=0.5)
        assert cfg.dirichlet_alpha == 0.5

    def test_accepts_large_positive(self) -> None:
        cfg = CIFAR100RecurrenceConfig(dirichlet_alpha=100.0)
        assert cfg.dirichlet_alpha == 100.0


# ------------------------------------------------------------------
# Balanced batch backward compatibility
# ------------------------------------------------------------------

class TestBalancedBatchBackwardCompatibility:
    """Ensure _draw_balanced_batch is unchanged when alpha=None."""

    def test_balanced_batch_produces_equal_class_counts(self) -> None:
        X_pool, y_pool = _make_synthetic_pool(n_classes=5, samples_per_class=200)
        rng = np.random.RandomState(42)
        _, y_batch = _draw_balanced_batch(X_pool, y_pool, 100, rng)
        _, counts = np.unique(y_batch, return_counts=True)
        # 100 / 5 = 20 per class exactly.
        np.testing.assert_array_equal(counts, [20, 20, 20, 20, 20])

    def test_balanced_batch_deterministic(self) -> None:
        X_pool, y_pool = _make_synthetic_pool()
        rng1 = np.random.RandomState(42)
        _, y1 = _draw_balanced_batch(X_pool, y_pool, 50, rng1)
        rng2 = np.random.RandomState(42)
        _, y2 = _draw_balanced_batch(X_pool, y_pool, 50, rng2)
        np.testing.assert_array_equal(y1, y2)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    """Edge-case behaviour for Dirichlet sampling."""

    def test_single_class_pool(self) -> None:
        """Pool with one class: Dirichlet(alpha * 1) = [1.0] always."""
        X_pool, y_pool = _make_synthetic_pool(n_classes=1, samples_per_class=100)
        rng = np.random.RandomState(42)
        X, y = _draw_dirichlet_batch(X_pool, y_pool, 30, 0.5, rng)
        assert X.shape == (30, 8)
        assert np.all(y == 0)

    def test_small_n_samples(self) -> None:
        """Even with n_samples=1 the function should not crash."""
        X_pool, y_pool = _make_synthetic_pool(n_classes=5)
        rng = np.random.RandomState(42)
        X, y = _draw_dirichlet_batch(X_pool, y_pool, 1, 0.5, rng)
        assert X.shape == (1, 8)
        assert y.shape == (1,)

    def test_n_samples_less_than_classes(self) -> None:
        """When n_samples < n_classes, some classes get zero samples."""
        X_pool, y_pool = _make_synthetic_pool(n_classes=10, samples_per_class=100)
        rng = np.random.RandomState(42)
        X, y = _draw_dirichlet_batch(X_pool, y_pool, 3, 0.01, rng)
        assert X.shape == (3, 8)
        assert y.shape == (3,)
        assert len(y) == 3

    def test_two_classes_extreme_skew(self) -> None:
        """With 2 classes and very low alpha, one class dominates."""
        X_pool, y_pool = _make_synthetic_pool(n_classes=2, samples_per_class=500)
        dominant_ratios: list[float] = []
        for trial in range(30):
            rng = np.random.RandomState(trial)
            _, y_batch = _draw_dirichlet_batch(X_pool, y_pool, 100, 0.01, rng)
            _, counts = np.unique(y_batch, return_counts=True)
            dominant_ratios.append(counts.max() / 100.0)
        # With alpha=0.01 and 2 classes, one class should get > 90% on average.
        assert np.mean(dominant_ratios) > 0.85
