from __future__ import annotations

"""Tests for FedGWC baseline (Gaussian Weighting with Wasserstein Clustering)."""

import numpy as np
import pytest

from fedprotrack.baselines.fedgwc import (
    FedGWCClient,
    FedGWCServer,
    FedGWCUpload,
    run_fedgwc_full,
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


class TestFedGWCClient:
    def test_predict_before_fit(self) -> None:
        client = FedGWCClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = FedGWCClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_cluster_id_default(self) -> None:
        client = FedGWCClient(0, 4, 2)
        assert client.cluster_id == 0

    def test_upload_structure(self) -> None:
        client = FedGWCClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, FedGWCUpload)
        assert upload.client_id == 0
        assert "coef" in upload.model_params
        assert "intercept" in upload.model_params
        assert upload.n_samples == 50


# ===========================================================================
# Server tests
# ===========================================================================


class TestFedGWCServer:
    def test_init(self) -> None:
        server = FedGWCServer(n_features=4, n_classes=2)
        assert server.n_features == 4
        assert server.n_classes == 2
        assert server.sigma == 1.0
        assert server.dbscan_eps == 0.5

    def test_aggregate_assigns_clusters(self) -> None:
        server = FedGWCServer(n_features=4, n_classes=3, dbscan_eps=0.9)
        # Create two distinct uploads (different coefs) and one similar
        u1 = FedGWCUpload(
            client_id=0,
            model_params={
                "coef": np.ones(3 * 4, dtype=np.float64),
                "intercept": np.zeros(3, dtype=np.float64),
            },
            n_samples=50,
        )
        u2 = FedGWCUpload(
            client_id=1,
            model_params={
                "coef": np.ones(3 * 4, dtype=np.float64) * 1.01,
                "intercept": np.zeros(3, dtype=np.float64),
            },
            n_samples=50,
        )
        cluster_models, client_clusters = server.aggregate([u1, u2])
        # Both should get a cluster assignment
        assert 0 in client_clusters
        assert 1 in client_clusters
        # At least one cluster model should exist
        assert len(cluster_models) >= 1

    def test_aggregate_single_client(self) -> None:
        server = FedGWCServer(n_features=4, n_classes=2)
        u = FedGWCUpload(
            client_id=0,
            model_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            n_samples=100,
        )
        cluster_models, client_clusters = server.aggregate([u])
        assert client_clusters[0] == 0
        assert 0 in cluster_models
        np.testing.assert_allclose(cluster_models[0]["coef"], np.ones(4))

    def test_aggregate_empty(self) -> None:
        server = FedGWCServer(n_features=4, n_classes=2)
        cluster_models, client_clusters = server.aggregate([])
        assert cluster_models == {}
        assert client_clusters == {}


# ===========================================================================
# Full runner tests
# ===========================================================================


class TestRunFedGWCFull:
    def test_smoke(self) -> None:
        dataset = _make_small_dataset()
        result = run_fedgwc_full(dataset, federation_every=2)
        assert result.method_name == "FedGWC"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_accuracy_in_range(self) -> None:
        dataset = _make_small_dataset()
        result = run_fedgwc_full(dataset, federation_every=1)
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)

    def test_predicted_has_cluster_ids(self) -> None:
        dataset = _make_small_dataset()
        result = run_fedgwc_full(dataset, federation_every=1)
        # After federation rounds, predicted concept matrix should contain
        # integer cluster IDs (>= 0)
        assert result.predicted_concept_matrix.dtype == np.int32
        assert np.all(result.predicted_concept_matrix >= 0)
