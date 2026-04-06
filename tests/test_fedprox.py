from __future__ import annotations

"""Tests for FedProx baseline (FedAvg + proximal regularisation)."""

import numpy as np
import pytest

from fedprotrack.baselines.fedprox import (
    FedProxClient,
    FedProxServer,
    FedProxUpload,
    run_fedprox_full,
)
from fedprotrack.baselines.runners import run_fedprox_full as run_fedprox_runner
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


def _make_multiclass_data(
    n: int = 90,
    d: int = 4,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simple multi-class dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (np.argmax(X[:, :n_classes], axis=1)).astype(np.int64)
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


class TestFedProxClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = FedProxClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = FedProxClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_fit_multiclass(self) -> None:
        client = FedProxClient(0, 4, 3, seed=42)
        X, y = _make_multiclass_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (90,)
        assert np.mean(preds == y) > 0.33

    def test_fit_with_prox_term(self) -> None:
        """Training with proximal term should not crash and should produce valid predictions."""
        client = FedProxClient(0, 4, 2, mu=0.1, seed=42)
        X, y = _make_data()

        # Set global params first
        global_params = {
            "coef": np.zeros(4, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
        }
        client.set_global_params(global_params)

        # Now train with prox
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == y.shape

    def test_prox_constrains_drift_from_global(self) -> None:
        """With high mu, local model should stay closer to global."""
        X, y = _make_data()
        global_params = {
            "coef": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
        }

        # Low mu: more local drift allowed
        client_low = FedProxClient(0, 4, 2, mu=0.001, seed=42)
        client_low.set_global_params(global_params)
        client_low.fit(X, y)
        params_low = client_low._model_params

        # High mu: constrained near global
        client_high = FedProxClient(1, 4, 2, mu=100.0, seed=42)
        client_high.set_global_params(global_params)
        client_high.fit(X, y)
        params_high = client_high._model_params

        # High-mu model should be closer to global params
        drift_low = np.linalg.norm(
            params_low["coef"] - global_params["coef"],
        )
        drift_high = np.linalg.norm(
            params_high["coef"] - global_params["coef"],
        )
        assert drift_high < drift_low

    def test_mu_zero_behaves_like_fedavg(self) -> None:
        """mu=0 should degenerate to plain FedAvg local training."""
        client = FedProxClient(0, 4, 2, mu=0.0, seed=42)
        X, y = _make_data()
        client.set_global_params({
            "coef": np.zeros(4, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
        })
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == y.shape
        assert np.mean(preds == y) > 0.5

    def test_upload_structure(self) -> None:
        client = FedProxClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, FedProxUpload)
        assert upload.client_id == 0
        assert "coef" in upload.model_params
        assert "intercept" in upload.model_params
        assert upload.n_samples == 50

    def test_upload_bytes_positive(self) -> None:
        client = FedProxClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0

    def test_upload_bytes_scales_with_precision(self) -> None:
        client = FedProxClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        b32 = client.upload_bytes(precision_bits=32)
        b64 = client.upload_bytes(precision_bits=64)
        assert b64 == pytest.approx(b32 * 2, rel=0.01)

    def test_mu_validation(self) -> None:
        with pytest.raises(ValueError, match="mu"):
            FedProxClient(0, 4, 2, mu=-0.1)


# ===========================================================================
# Server tests
# ===========================================================================


class TestFedProxServer:
    def test_init_creates_global_params(self) -> None:
        server = FedProxServer(n_features=4, n_classes=2)
        assert "coef" in server.global_params
        assert "intercept" in server.global_params

    def test_aggregate_empty(self) -> None:
        server = FedProxServer(n_features=4, n_classes=2)
        result = server.aggregate([])
        assert "coef" in result

    def test_aggregate_single_client(self) -> None:
        server = FedProxServer(n_features=4, n_classes=2)
        old_coef = server.global_params["coef"].copy()
        upload = FedProxUpload(
            client_id=0,
            model_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            n_samples=100,
        )
        result = server.aggregate([upload])
        np.testing.assert_allclose(result["coef"], np.ones(4))

    def test_aggregate_weighted_average(self) -> None:
        server = FedProxServer(n_features=4, n_classes=2)
        u1 = FedProxUpload(
            client_id=0,
            model_params={
                "coef": np.zeros(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            n_samples=50,
        )
        u2 = FedProxUpload(
            client_id=1,
            model_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            n_samples=50,
        )
        result = server.aggregate([u1, u2])
        # Equal weights -> midpoint
        np.testing.assert_allclose(result["coef"], 0.5 * np.ones(4))

    def test_download_bytes_positive(self) -> None:
        server = FedProxServer(n_features=4, n_classes=2)
        assert server.download_bytes(n_clients=5) > 0

    def test_download_bytes_scales_with_clients(self) -> None:
        server = FedProxServer(n_features=4, n_classes=2)
        b1 = server.download_bytes(n_clients=1)
        b5 = server.download_bytes(n_clients=5)
        assert b5 == pytest.approx(b1 * 5, rel=0.01)


# ===========================================================================
# Full runner tests
# ===========================================================================


class TestRunFedProxFull:
    def test_smoke_direct(self) -> None:
        """Direct runner from fedprox module."""
        dataset = _make_small_dataset()
        result = run_fedprox_full(dataset, federation_every=2)
        assert result.method_name == "FedProx"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_smoke_via_runners(self) -> None:
        """Runner re-exported through runners.py."""
        dataset = _make_small_dataset()
        result = run_fedprox_runner(dataset, federation_every=2)
        assert result.method_name == "FedProx"
        assert result.accuracy_matrix.shape == (2, 4)

    def test_accuracy_in_range(self) -> None:
        dataset = _make_small_dataset()
        result = run_fedprox_full(dataset, federation_every=1)
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)

    def test_byte_tracking(self) -> None:
        dataset = _make_small_dataset()
        result_freq1 = run_fedprox_full(dataset, federation_every=1)
        result_freq2 = run_fedprox_full(dataset, federation_every=2)
        # More frequent federation = more bytes
        assert result_freq1.total_bytes >= result_freq2.total_bytes

    def test_no_concept_tracking(self) -> None:
        """FedProx has no identity inference -- predicted should be all zeros."""
        dataset = _make_small_dataset()
        result = run_fedprox_full(dataset, federation_every=1)
        assert np.all(result.predicted_concept_matrix == 0)
