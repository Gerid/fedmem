from __future__ import annotations

"""Tests for SCAFFOLD baseline (variance reduction via control variates)."""

import numpy as np
import pytest

from fedprotrack.baselines.scaffold import (
    SCAFFOLDClient,
    SCAFFOLDServer,
    SCAFFOLDUpload,
    run_scaffold_full,
)
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.generator import DriftDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n: int = 50,
    d: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple binary classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def _make_small_dataset(
    K: int = 2,
    T: int = 4,
    n_samples: int = 30,
) -> DriftDataset:
    """Create a tiny mock DriftDataset."""
    rng = np.random.RandomState(42)
    concept_matrix = np.zeros((K, T), dtype=np.int32)
    concept_matrix[:, T // 2:] = 1
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            X = rng.randn(n_samples, 4).astype(np.float64)
            if concept_matrix[k, t] == 0:
                y = (X[:, 0] > 0).astype(np.int64)
            else:
                y = (X[:, 1] > 0).astype(np.int64)
            data[(k, t)] = (X, y)
    config = GeneratorConfig(K=K, T=T)
    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=config,
        concept_specs=[],
    )


# ===========================================================================
# Client tests
# ===========================================================================


class TestSCAFFOLDClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = SCAFFOLDClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = SCAFFOLDClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_upload_has_control_delta(self) -> None:
        """After training, upload should contain a non-trivial control delta."""
        client = SCAFFOLDClient(0, 4, 2, seed=42)
        X, y = _make_data()

        # First fit initialises; second fit produces real control deltas.
        client.fit(X, y)
        client.fit(X, y)

        upload = client.get_upload()
        assert isinstance(upload, SCAFFOLDUpload)
        assert upload.client_id == 0
        assert "coef" in upload.control_delta
        assert "intercept" in upload.control_delta
        # Control delta should be non-zero after second fit.
        has_nonzero = any(
            np.any(v != 0) for v in upload.control_delta.values()
        )
        assert has_nonzero, "control_delta should be non-zero after second fit"

    def test_upload_bytes_larger_than_model(self) -> None:
        """SCAFFOLD uploads model + control delta, so cost should be ~2x model."""
        client = SCAFFOLDClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        client.fit(X, y)

        from fedprotrack.baselines.comm_tracker import model_bytes

        model_only = model_bytes(client._model_params)
        total_upload = client.upload_bytes()
        # Upload should be strictly larger than model-only cost.
        assert total_upload > model_only
        # Should be approximately 2x (model + control delta of same shape).
        assert total_upload == pytest.approx(model_only * 2, rel=0.01)

    def test_control_updates_after_fit(self) -> None:
        """Control variate should change between successive fits."""
        client = SCAFFOLDClient(0, 4, 2, seed=42)
        X, y = _make_data()

        # First fit initialises.
        client.fit(X, y)
        c_after_first = {k: v.copy() for k, v in client._control.items()}

        # Second fit updates control.
        client.fit(X, y)
        c_after_second = client._control

        # At least one key should have changed.
        changed = any(
            not np.allclose(c_after_first[k], c_after_second[k])
            for k in c_after_first
        )
        assert changed, "control variate should change after second fit"


# ===========================================================================
# Server tests
# ===========================================================================


class TestSCAFFOLDServer:
    def test_init_control_zeros(self) -> None:
        server = SCAFFOLDServer(n_features=4, n_classes=2)
        assert "coef" in server.global_control
        assert "intercept" in server.global_control
        for v in server.global_control.values():
            assert np.all(v == 0)

    def test_aggregate_updates_control(self) -> None:
        server = SCAFFOLDServer(n_features=4, n_classes=2)
        old_control = {k: v.copy() for k, v in server.global_control.items()}

        upload = SCAFFOLDUpload(
            client_id=0,
            model_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            control_delta={
                "coef": np.ones(4, dtype=np.float64) * 0.1,
                "intercept": np.ones(1, dtype=np.float64) * 0.05,
            },
            n_samples=100,
        )
        _, new_control = server.aggregate([upload])

        # Control should have been updated by (1/N) * delta.
        np.testing.assert_allclose(
            new_control["coef"],
            old_control["coef"] + 0.1,
        )
        np.testing.assert_allclose(
            new_control["intercept"],
            old_control["intercept"] + 0.05,
        )

    def test_aggregate_empty(self) -> None:
        server = SCAFFOLDServer(n_features=4, n_classes=2)
        params, control = server.aggregate([])
        assert "coef" in params
        assert "coef" in control


# ===========================================================================
# Full runner tests
# ===========================================================================


class TestRunSCAFFOLDFull:
    def test_smoke(self) -> None:
        dataset = _make_small_dataset()
        result = run_scaffold_full(dataset, federation_every=2)
        assert result.method_name == "SCAFFOLD"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_accuracy_in_range(self) -> None:
        dataset = _make_small_dataset()
        result = run_scaffold_full(dataset, federation_every=1)
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)

    def test_bytes_positive(self) -> None:
        dataset = _make_small_dataset()
        result = run_scaffold_full(dataset, federation_every=1)
        assert result.total_bytes > 0

    def test_no_concept_tracking(self) -> None:
        """SCAFFOLD has no identity inference -- predicted should be all zeros."""
        dataset = _make_small_dataset()
        result = run_scaffold_full(dataset, federation_every=1)
        assert np.all(result.predicted_concept_matrix == 0)
