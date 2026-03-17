"""Tests for Rotating MNIST data loader."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.real_data.rotating_mnist import (
    RotatingMNISTConfig,
    generate_rotating_mnist_dataset,
    _rotation_angles,
)


class TestRotationAngles:
    def test_single_concept(self) -> None:
        angles = _rotation_angles(1, 0.5)
        assert len(angles) == 1
        assert angles[0] == 0.0

    def test_two_concepts_low_delta(self) -> None:
        angles = _rotation_angles(2, 0.1)
        assert len(angles) == 2
        assert angles[0] == 0.0
        assert abs(angles[1] - 18.0) < 1e-6

    def test_two_concepts_high_delta(self) -> None:
        angles = _rotation_angles(2, 1.0)
        assert angles[1] == 180.0

    def test_four_concepts(self) -> None:
        angles = _rotation_angles(4, 1.0)
        assert len(angles) == 4
        assert angles[0] == 0.0
        assert abs(angles[-1] - 180.0) < 1e-6


@pytest.mark.slow
class TestRotatingMNISTDataset:
    def test_basic_generation(self) -> None:
        cfg = RotatingMNISTConfig(
            K=3, T=4, n_samples=50,
            rho=3.0, alpha=0.5, delta=0.5,
            n_features=5, seed=42,
        )
        ds = generate_rotating_mnist_dataset(cfg)

        assert ds.concept_matrix.shape == (3, 4)
        assert len(ds.data) == 12  # K * T
        for (k, t), (X, y) in ds.data.items():
            assert X.shape == (50, 5)
            assert y.shape == (50,)
            assert set(y).issubset({0, 1})

    def test_different_seeds(self) -> None:
        cfg1 = RotatingMNISTConfig(K=2, T=3, n_samples=30,
                                   n_features=5, seed=42)
        cfg2 = RotatingMNISTConfig(K=2, T=3, n_samples=30,
                                   n_features=5, seed=99)
        ds1 = generate_rotating_mnist_dataset(cfg1)
        ds2 = generate_rotating_mnist_dataset(cfg2)

        # Different seeds should produce different data
        X1, _ = ds1.data[(0, 0)]
        X2, _ = ds2.data[(0, 0)]
        assert not np.allclose(X1, X2)

    def test_concept_matrix_valid(self) -> None:
        cfg = RotatingMNISTConfig(K=4, T=6, n_samples=30,
                                  rho=3.0, alpha=0.5, delta=0.5,
                                  n_features=5, seed=42)
        ds = generate_rotating_mnist_dataset(cfg)

        # Concept IDs should be non-negative
        assert np.all(ds.concept_matrix >= 0)
        # Should have the right shape
        assert ds.concept_matrix.shape == (4, 6)
