from __future__ import annotations

"""Tests for Adaptive-FedAvg baseline (FedAvg + drift-adaptive learning rate)."""

import numpy as np
import pytest

from fedprotrack.baselines.adaptive_fedavg import (
    AdaptiveFedAvgClient,
    AdaptiveFedAvgServer,
    AdaptiveFedAvgUpload,
    run_adaptive_fedavg_full,
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
    """Create a tiny mock DriftDataset with a concept shift at T//2."""
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


class TestAdaptiveFedAvgClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = AdaptiveFedAvgClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = AdaptiveFedAvgClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_upload_structure(self) -> None:
        client = AdaptiveFedAvgClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, AdaptiveFedAvgUpload)
        assert upload.client_id == 0
        assert "coef" in upload.model_params
        assert "intercept" in upload.model_params
        assert upload.n_samples == 50

    def test_upload_bytes_positive(self) -> None:
        client = AdaptiveFedAvgClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0

    def test_lr_decays_on_stable_data(self) -> None:
        """On repeated stable data the lr should decay below its initial value."""
        client = AdaptiveFedAvgClient(
            0, 4, 2, lr=0.01, decay_factor=0.9, seed=42,
        )
        X, y = _make_data()
        initial_lr = client.current_lr

        # Fit several rounds on the same data -- loss should be stable
        for _ in range(5):
            client.fit(X, y)

        assert client.current_lr < initial_lr

    def test_lr_boosts_on_drift(self) -> None:
        """After training on concept A, switching to concept B should trigger
        a loss spike and boost the learning rate."""
        client = AdaptiveFedAvgClient(
            0, 4, 2,
            lr=0.01,
            boost_factor=3.0,
            drift_threshold=1.2,
            ema_alpha=0.05,
            seed=42,
        )
        rng = np.random.RandomState(99)

        # Train on concept A (feature 0 determines label) for many rounds
        # so the model converges and loss EMA becomes low
        for _ in range(15):
            X = rng.randn(80, 4).astype(np.float64)
            y = (X[:, 0] > 0).astype(np.int64)
            client.fit(X, y)

        lr_before_drift = client.current_lr

        # Switch to concept B (random labels w.r.t. learned decision boundary)
        # This should cause a large loss spike relative to the low EMA
        X_drift = rng.randn(80, 4).astype(np.float64)
        y_drift = (1 - (X_drift[:, 0] > 0)).astype(np.int64)  # flip labels
        client.fit(X_drift, y_drift)

        # lr should have increased (boosted)
        assert client.current_lr > lr_before_drift

    def test_lr_stays_bounded(self) -> None:
        """LR must not exceed 1.0 or drop below 1e-5."""
        client = AdaptiveFedAvgClient(
            0, 4, 2,
            lr=0.5,
            boost_factor=10.0,
            decay_factor=0.001,
            drift_threshold=1.0,
            seed=42,
        )
        X, y = _make_data()

        for _ in range(20):
            client.fit(X, y)

        assert 1e-5 <= client.current_lr <= 1.0


# ===========================================================================
# Server tests
# ===========================================================================


class TestAdaptiveFedAvgServer:
    def test_init_creates_global_params(self) -> None:
        server = AdaptiveFedAvgServer(n_features=4, n_classes=2)
        assert "coef" in server.global_params
        assert "intercept" in server.global_params

    def test_aggregate_empty(self) -> None:
        server = AdaptiveFedAvgServer(n_features=4, n_classes=2)
        result = server.aggregate([])
        assert "coef" in result


# ===========================================================================
# Full runner tests
# ===========================================================================


class TestRunAdaptiveFedAvgFull:
    def test_smoke(self) -> None:
        dataset = _make_small_dataset()
        result = run_adaptive_fedavg_full(dataset, federation_every=2)
        assert result.method_name == "Adaptive-FedAvg"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_accuracy_in_range(self) -> None:
        dataset = _make_small_dataset()
        result = run_adaptive_fedavg_full(dataset, federation_every=1)
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)

    def test_bytes_positive(self) -> None:
        dataset = _make_small_dataset()
        result = run_adaptive_fedavg_full(dataset, federation_every=1)
        assert result.total_bytes > 0

    def test_no_concept_tracking(self) -> None:
        """Adaptive-FedAvg has no identity inference -- predicted should be all zeros."""
        dataset = _make_small_dataset()
        result = run_adaptive_fedavg_full(dataset, federation_every=1)
        assert np.all(result.predicted_concept_matrix == 0)
