"""Tests for FMOW temporal-drift dataset loader."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.real_data.fmow import (
    FMOWConfig,
    _draw_balanced_batch,
    _generate_synthetic_fmow_proxy,
    _year_to_concept,
    generate_fmow_dataset,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestFMOWConfig:
    def test_defaults(self):
        cfg = FMOWConfig()
        assert cfg.K == 5
        assert cfg.T == 10
        assert cfg.n_concepts == 4

    def test_invalid_K(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            FMOWConfig(K=0)

    def test_invalid_T(self):
        with pytest.raises(ValueError, match="T must be >= 2"):
            FMOWConfig(T=1)

    def test_invalid_n_concepts(self):
        with pytest.raises(ValueError, match="n_concepts must be >= 2"):
            FMOWConfig(n_concepts=1)

    def test_invalid_n_features(self):
        with pytest.raises(ValueError, match="n_features must be >= 2"):
            FMOWConfig(n_features=1)

    def test_invalid_n_classes(self):
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            FMOWConfig(n_classes=1)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            FMOWConfig(alpha=1.5)

    def test_invalid_delta(self):
        with pytest.raises(ValueError, match="delta"):
            FMOWConfig(delta=0.0)


# ---------------------------------------------------------------------------
# Year-to-concept binning
# ---------------------------------------------------------------------------

class TestYearToConcept:
    def test_basic_binning(self):
        years = np.array([2002, 2005, 2010, 2017])
        cids = _year_to_concept(years, n_concepts=4)
        assert cids.shape == (4,)
        assert cids.min() >= 0
        assert cids.max() < 4
        # 2002 should be in the first bin, 2017 in the last
        assert cids[0] == 0
        assert cids[-1] == 3

    def test_two_concepts(self):
        years = np.array([2002, 2003, 2015, 2017])
        cids = _year_to_concept(years, n_concepts=2)
        assert cids[0] == 0
        assert cids[-1] == 1

    def test_single_year(self):
        years = np.array([2010, 2010, 2010])
        cids = _year_to_concept(years, n_concepts=3)
        # All same year => all concept 0
        assert np.all(cids == 0)

    def test_monotonic(self):
        years = np.arange(2002, 2018)
        cids = _year_to_concept(years, n_concepts=4)
        # Should be non-decreasing
        assert np.all(np.diff(cids) >= 0)


# ---------------------------------------------------------------------------
# Balanced batch drawing
# ---------------------------------------------------------------------------

class TestDrawBalancedBatch:
    def test_correct_size(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        y = np.array([0] * 50 + [1] * 50, dtype=np.int64)
        X_b, y_b = _draw_balanced_batch(X, y, 20, rng)
        assert X_b.shape == (20, 10)
        assert y_b.shape == (20,)

    def test_roughly_balanced(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        y = np.array([0] * 100 + [1] * 100, dtype=np.int64)
        X_b, y_b = _draw_balanced_batch(X, y, 40, rng)
        # Each class should get ~20 samples
        assert abs(np.sum(y_b == 0) - 20) <= 1
        assert abs(np.sum(y_b == 1) - 20) <= 1


# ---------------------------------------------------------------------------
# Synthetic proxy
# ---------------------------------------------------------------------------

class TestSyntheticProxy:
    def test_shapes(self):
        images, labels, years = _generate_synthetic_fmow_proxy(5, 42)
        assert images.ndim == 4  # (N, 3, 64, 64)
        assert images.shape[1] == 3
        assert labels.shape == (images.shape[0],)
        assert years.shape == (images.shape[0],)
        assert len(np.unique(labels)) == 5

    def test_year_range(self):
        _, _, years = _generate_synthetic_fmow_proxy(3, 42)
        assert years.min() >= 2002
        assert years.max() <= 2017


# ---------------------------------------------------------------------------
# End-to-end dataset generation (using synthetic fallback)
# ---------------------------------------------------------------------------

class TestGenerateFMOWDataset:
    """Test full pipeline using synthetic proxy (no real data needed)."""

    def test_basic_generation(self):
        cfg = FMOWConfig(
            K=3, T=5, n_samples=50, n_concepts=3,
            n_features=8, n_classes=4, seed=42,
            # Force synthetic fallback by using a non-existent root
            data_root=".test_fmow_cache_nonexist",
            feature_cache_dir=".test_fmow_features_nonexist",
        )
        ds = generate_fmow_dataset(cfg)

        assert ds.concept_matrix.shape == (3, 5)
        assert len(ds.data) == 15  # K * T
        for (k, t), (X, y) in ds.data.items():
            assert X.shape == (50, 8), f"Bad shape at ({k}, {t}): {X.shape}"
            assert y.shape == (50,)
            assert X.dtype == np.float32
        assert ds.config.generator_type == "fmow"

    def test_deterministic(self):
        cfg = FMOWConfig(
            K=2, T=3, n_samples=30, n_concepts=2,
            n_features=4, n_classes=3, seed=99,
            data_root=".test_fmow_cache_nonexist",
            feature_cache_dir=".test_fmow_features_nonexist",
        )
        ds1 = generate_fmow_dataset(cfg)
        ds2 = generate_fmow_dataset(cfg)

        np.testing.assert_array_equal(ds1.concept_matrix, ds2.concept_matrix)
        for key in ds1.data:
            np.testing.assert_array_equal(ds1.data[key][0], ds2.data[key][0])
            np.testing.assert_array_equal(ds1.data[key][1], ds2.data[key][1])

    def test_concept_specs(self):
        cfg = FMOWConfig(
            K=2, T=4, n_samples=20, n_concepts=3,
            n_features=4, n_classes=3, seed=42,
            data_root=".test_fmow_cache_nonexist",
            feature_cache_dir=".test_fmow_features_nonexist",
        )
        ds = generate_fmow_dataset(cfg)
        assert len(ds.concept_specs) >= 2
        for spec in ds.concept_specs:
            assert spec.generator_type == "fmow"
