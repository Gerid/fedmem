"""Tests for predictive signature helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.posterior.predictive_signatures import (
    classifier_rows_from_params,
    extract_batch_prototypes,
    project_batch_prototype_signatures,
    project_classifier_row_signatures,
    project_update_delta_signatures,
)


def _binary_params(n_features: int = 4) -> dict[str, np.ndarray]:
    return {
        "coef": np.arange(n_features, dtype=np.float64),
        "intercept": np.array([0.5], dtype=np.float64),
    }


def _multiclass_params(n_features: int = 4, n_classes: int = 3) -> dict[str, np.ndarray]:
    return {
        "coef": np.arange(n_classes * n_features, dtype=np.float64),
        "intercept": np.arange(n_classes, dtype=np.float64),
    }


class TestClassifierRows:
    def test_binary_rows_are_symmetric(self) -> None:
        rows = classifier_rows_from_params(_binary_params(), n_features=4, n_classes=2)
        assert rows.shape == (2, 5)
        np.testing.assert_allclose(rows[0], -rows[1])

    def test_multiclass_rows_shape(self) -> None:
        rows = classifier_rows_from_params(_multiclass_params(), n_features=4, n_classes=3)
        assert rows.shape == (3, 5)

    def test_missing_keys_raise(self) -> None:
        with pytest.raises(KeyError):
            classifier_rows_from_params({"coef": np.ones(4)}, n_features=4, n_classes=2)


class TestProjectedSignatures:
    def test_projection_is_deterministic(self) -> None:
        sig1 = project_classifier_row_signatures(
            _multiclass_params(),
            n_features=4,
            n_classes=3,
            output_dim=6,
            seed=42,
        )
        sig2 = project_classifier_row_signatures(
            _multiclass_params(),
            n_features=4,
            n_classes=3,
            output_dim=6,
            seed=42,
        )
        np.testing.assert_allclose(sig1, sig2)

    def test_projection_rows_are_normalized(self) -> None:
        sig = project_classifier_row_signatures(
            _multiclass_params(),
            n_features=4,
            n_classes=3,
            output_dim=6,
            seed=7,
        )
        norms = np.linalg.norm(sig, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-12)

    def test_update_delta_zero_when_params_match(self) -> None:
        sig = project_update_delta_signatures(
            _multiclass_params(),
            _multiclass_params(),
            n_features=4,
            n_classes=3,
            output_dim=6,
            seed=9,
        )
        np.testing.assert_allclose(sig, np.zeros_like(sig))

    def test_update_delta_changes_with_new_params(self) -> None:
        current = _multiclass_params()
        previous = _multiclass_params()
        previous["coef"] = previous["coef"] + 1.0
        sig = project_update_delta_signatures(
            current,
            previous,
            n_features=4,
            n_classes=3,
            output_dim=6,
            seed=9,
        )
        assert sig.shape == (3, 6)
        assert np.linalg.norm(sig) > 0.0


class TestBatchPrototypes:
    def test_extracts_prototypes_and_counts(self) -> None:
        X = np.array(
            [
                [1.0, 0.0],
                [3.0, 2.0],
                [10.0, 10.0],
                [14.0, 14.0],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 0, 2, 2], dtype=np.int64)
        prototypes, counts = extract_batch_prototypes(X, y, n_classes=3)
        assert prototypes.shape == (3, 2)
        assert counts.tolist() == [2.0, 0.0, 2.0]
        np.testing.assert_allclose(prototypes[0], np.array([2.0, 1.0]))
        np.testing.assert_allclose(prototypes[1], np.zeros(2))
        np.testing.assert_allclose(prototypes[2], np.array([12.0, 12.0]))

    def test_projected_batch_prototypes_shape(self) -> None:
        X = np.array(
            [
                [1.0, 0.0],
                [3.0, 2.0],
                [10.0, 10.0],
                [14.0, 14.0],
            ],
            dtype=np.float64,
        )
        y = np.array([0, 0, 2, 2], dtype=np.int64)
        sig, counts = project_batch_prototype_signatures(
            X,
            y,
            output_dim=5,
            seed=11,
            n_classes=3,
        )
        assert sig.shape == (3, 5)
        assert counts.tolist() == [2.0, 0.0, 2.0]

    def test_invalid_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            extract_batch_prototypes(np.ones((2, 2, 2)), np.array([0, 1]))
