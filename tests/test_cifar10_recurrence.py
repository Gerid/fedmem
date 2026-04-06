"""Tests for CIFAR-10 recurrence dataset loader."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.real_data.cifar10_recurrence import (
    CIFAR10RecurrenceConfig,
    DEFAULT_CONCEPT_GROUPS,
    _apply_concept_style,
    _draw_balanced_batch,
    _select_concept_indices,
    generate_cifar10_recurrence_dataset,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestCIFAR10RecurrenceConfig:
    def test_defaults(self):
        cfg = CIFAR10RecurrenceConfig()
        assert cfg.K == 6
        assert cfg.T == 12
        assert cfg.n_samples == 400
        assert cfg.n_features == 128
        assert cfg.concept_groups == DEFAULT_CONCEPT_GROUPS

    def test_invalid_K(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            CIFAR10RecurrenceConfig(K=0)

    def test_invalid_T(self):
        with pytest.raises(ValueError, match="T must be >= 2"):
            CIFAR10RecurrenceConfig(T=1)

    def test_invalid_n_samples(self):
        with pytest.raises(ValueError, match="n_samples must be >= 1"):
            CIFAR10RecurrenceConfig(n_samples=0)

    def test_invalid_samples_per_class(self):
        with pytest.raises(ValueError, match="samples_per_class must be >= 1"):
            CIFAR10RecurrenceConfig(samples_per_class=0)

    def test_invalid_n_features(self):
        with pytest.raises(ValueError, match="n_features must be >= 2"):
            CIFAR10RecurrenceConfig(n_features=1)

    def test_invalid_alpha_low(self):
        with pytest.raises(ValueError, match="alpha"):
            CIFAR10RecurrenceConfig(alpha=-0.1)

    def test_invalid_alpha_high(self):
        with pytest.raises(ValueError, match="alpha"):
            CIFAR10RecurrenceConfig(alpha=1.5)

    def test_invalid_delta_zero(self):
        with pytest.raises(ValueError, match="delta"):
            CIFAR10RecurrenceConfig(delta=0.0)

    def test_invalid_delta_high(self):
        with pytest.raises(ValueError, match="delta"):
            CIFAR10RecurrenceConfig(delta=1.5)

    def test_invalid_concept_groups_too_few(self):
        with pytest.raises(ValueError, match="concept_groups must have >= 2"):
            CIFAR10RecurrenceConfig(concept_groups={0: [0, 1]})

    def test_invalid_concept_groups_empty_list(self):
        with pytest.raises(ValueError, match="is empty"):
            CIFAR10RecurrenceConfig(concept_groups={0: [0, 1], 1: []})

    def test_invalid_concept_groups_bad_class(self):
        with pytest.raises(ValueError, match="invalid class 10"):
            CIFAR10RecurrenceConfig(concept_groups={0: [0, 1], 1: [10]})

    def test_invalid_concept_groups_negative_class(self):
        with pytest.raises(ValueError, match="invalid class -1"):
            CIFAR10RecurrenceConfig(concept_groups={0: [0, 1], 1: [-1]})

    def test_to_generator_config(self):
        cfg = CIFAR10RecurrenceConfig(K=4, T=8, seed=99)
        gen = cfg.to_generator_config()
        assert gen.K == 4
        assert gen.T == 8
        assert gen.seed == 99
        assert gen.generator_type == "cifar10_recurrence"

    def test_groups_hash_deterministic(self):
        cfg1 = CIFAR10RecurrenceConfig()
        cfg2 = CIFAR10RecurrenceConfig()
        assert cfg1._groups_hash() == cfg2._groups_hash()

    def test_groups_hash_changes_with_groups(self):
        cfg1 = CIFAR10RecurrenceConfig()
        cfg2 = CIFAR10RecurrenceConfig(
            concept_groups={0: [0, 1, 2], 1: [3, 4], 2: [5, 6], 3: [7, 8, 9]}
        )
        assert cfg1._groups_hash() != cfg2._groups_hash()


# ---------------------------------------------------------------------------
# Default concept groups coverage
# ---------------------------------------------------------------------------

class TestDefaultConceptGroups:
    def test_all_classes_covered(self):
        """Every CIFAR-10 class (0-9) appears in exactly one group."""
        all_classes = set()
        for classes in DEFAULT_CONCEPT_GROUPS.values():
            all_classes.update(classes)
        assert all_classes == set(range(10))

    def test_no_overlap(self):
        """No class appears in more than one concept group."""
        seen: set[int] = set()
        for classes in DEFAULT_CONCEPT_GROUPS.values():
            for cls in classes:
                assert cls not in seen, f"class {cls} appears in multiple groups"
                seen.add(cls)

    def test_five_concepts(self):
        assert len(DEFAULT_CONCEPT_GROUPS) == 5


# ---------------------------------------------------------------------------
# Concept style transform
# ---------------------------------------------------------------------------

class TestConceptStyle:
    def test_output_shape_preserved(self):
        import torch

        img = torch.rand(3, 32, 32)
        for cid in range(5):
            out = _apply_concept_style(img, cid, delta=0.85)
            assert out.shape == (3, 32, 32)

    def test_output_range(self):
        import torch

        img = torch.rand(3, 32, 32)
        for cid in range(5):
            out = _apply_concept_style(img, cid, delta=0.85)
            assert out.min() >= 0.0
            assert out.max() <= 1.0

    def test_identity_style(self):
        """Style 0 should return the image unchanged."""
        import torch

        img = torch.rand(3, 32, 32)
        out = _apply_concept_style(img, concept_id=0, delta=0.85)
        assert torch.allclose(out, img.clamp(0.0, 1.0))


# ---------------------------------------------------------------------------
# Index selection helper
# ---------------------------------------------------------------------------

class TestSelectConceptIndices:
    def test_basic_selection(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        groups = {0: [0, 1], 1: [2, 3], 2: [4]}
        result = _select_concept_indices(labels, groups, samples_per_class=2, seed=42)
        assert set(result.keys()) == {0, 1, 2}
        # concept 0 has classes 0 and 1, 2 samples each -> 4 indices
        assert len(result[0]) == 4
        # concept 2 has class 4 only, 2 samples -> 2 indices
        assert len(result[2]) == 2

    def test_samples_per_class_cap(self):
        labels = np.array([0] * 100 + [1] * 100)
        groups = {0: [0], 1: [1]}
        result = _select_concept_indices(labels, groups, samples_per_class=10, seed=42)
        assert len(result[0]) == 10
        assert len(result[1]) == 10


# ---------------------------------------------------------------------------
# Balanced batch drawing
# ---------------------------------------------------------------------------

class TestDrawBalancedBatch:
    def test_output_shapes(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 8).astype(np.float32)
        y = np.repeat(np.arange(5), 20).astype(np.int64)
        X_batch, y_batch = _draw_balanced_batch(X, y, n_samples=50, rng=rng)
        assert X_batch.shape == (50, 8)
        assert y_batch.shape == (50,)

    def test_class_balance(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 4).astype(np.float32)
        y = np.repeat(np.arange(2), 50).astype(np.int64)
        _, y_batch = _draw_balanced_batch(X, y, n_samples=40, rng=rng)
        counts = np.bincount(y_batch)
        # 40 samples / 2 classes = 20 each
        assert counts[0] == 20
        assert counts[1] == 20

    def test_dtypes(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 4).astype(np.float64)
        y = np.zeros(50, dtype=np.int32)
        X_batch, y_batch = _draw_balanced_batch(X, y, n_samples=10, rng=rng)
        assert X_batch.dtype == np.float32
        assert y_batch.dtype == np.int64


# ---------------------------------------------------------------------------
# End-to-end dataset generation (requires download -- slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGenerateCIFAR10RecurrenceDataset:
    def test_smoke(self):
        cfg = CIFAR10RecurrenceConfig(
            K=2, T=3, n_samples=10, rho=1.5,
            samples_per_class=20, n_features=16,
        )
        ds = generate_cifar10_recurrence_dataset(cfg)
        assert ds.concept_matrix.shape == (2, 3)
        assert len(ds.data) == 2 * 3
        for (k, t), (X, y) in ds.data.items():
            assert X.shape == (10, 16)
            assert y.shape == (10,)
            assert X.dtype == np.float32
            assert y.dtype == np.int64

    def test_concept_matrix_values_in_range(self):
        cfg = CIFAR10RecurrenceConfig(
            K=3, T=4, n_samples=10, rho=2.0,
            samples_per_class=20, n_features=16,
        )
        ds = generate_cifar10_recurrence_dataset(cfg)
        n_concepts = cfg.to_generator_config().n_concepts
        assert ds.concept_matrix.min() >= 0
        assert ds.concept_matrix.max() < n_concepts

    def test_concept_specs_populated(self):
        cfg = CIFAR10RecurrenceConfig(
            K=2, T=3, n_samples=10, rho=1.5,
            samples_per_class=20, n_features=16,
        )
        ds = generate_cifar10_recurrence_dataset(cfg)
        assert len(ds.concept_specs) >= 2
        for spec in ds.concept_specs:
            assert spec.generator_type == "cifar10_recurrence"

    def test_deterministic(self):
        cfg = CIFAR10RecurrenceConfig(
            K=2, T=3, n_samples=10, rho=1.5,
            samples_per_class=20, n_features=16, seed=123,
        )
        ds1 = generate_cifar10_recurrence_dataset(cfg)
        ds2 = generate_cifar10_recurrence_dataset(cfg)
        np.testing.assert_array_equal(ds1.concept_matrix, ds2.concept_matrix)
        for key in ds1.data:
            np.testing.assert_array_equal(ds1.data[key][0], ds2.data[key][0])
            np.testing.assert_array_equal(ds1.data[key][1], ds2.data[key][1])
