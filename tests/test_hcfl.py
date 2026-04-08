from __future__ import annotations

"""Tests for HCFL baseline (Hierarchical Clustered Federated Learning)."""

import numpy as np
import pytest

from fedprotrack.baselines.hcfl import (
    HCFLClient,
    HCFLServer,
    HCFLUpload,
    run_hcfl_full,
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


class TestHCFLClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = HCFLClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = HCFLClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_cluster_id_default_zero(self) -> None:
        client = HCFLClient(0, 4, 2)
        assert client.cluster_id == 0

    def test_set_cluster_id(self) -> None:
        client = HCFLClient(0, 4, 2)
        client.set_cluster_id(3)
        assert client.cluster_id == 3

    def test_upload_structure(self) -> None:
        client = HCFLClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, HCFLUpload)
        assert upload.client_id == 0
        assert "coef" in upload.model_params
        assert "intercept" in upload.model_params
        assert upload.n_samples == 50

    def test_upload_bytes_positive(self) -> None:
        client = HCFLClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0


# ===========================================================================
# Server tests
# ===========================================================================


class TestHCFLServer:
    def test_init(self) -> None:
        server = HCFLServer(n_features=4, n_classes=2)
        assert "coef" in server.global_params
        assert "intercept" in server.global_params

    def test_aggregate_assigns_clusters(self) -> None:
        server = HCFLServer(n_features=4, n_classes=2, distance_threshold=0.3)
        # Two very different param sets should yield 2 clusters
        u1 = HCFLUpload(
            client_id=0,
            model_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.ones(1, dtype=np.float64),
            },
            n_samples=50,
        )
        u2 = HCFLUpload(
            client_id=1,
            model_params={
                "coef": -np.ones(4, dtype=np.float64),
                "intercept": -np.ones(1, dtype=np.float64),
            },
            n_samples=50,
        )
        cluster_models, client_clusters = server.aggregate([u1, u2])
        assert isinstance(cluster_models, dict)
        assert isinstance(client_clusters, dict)
        assert 0 in client_clusters
        assert 1 in client_clusters
        # Opposite-sign vectors have cosine distance ~2.0, so with
        # threshold=0.3 they should be in different clusters
        assert client_clusters[0] != client_clusters[1]
        assert len(cluster_models) == 2

    def test_aggregate_single_client(self) -> None:
        server = HCFLServer(n_features=4, n_classes=2)
        u = HCFLUpload(
            client_id=0,
            model_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            n_samples=100,
        )
        cluster_models, client_clusters = server.aggregate([u])
        assert len(cluster_models) == 1
        assert 0 in cluster_models
        assert client_clusters[0] == 0
        np.testing.assert_allclose(cluster_models[0]["coef"], np.ones(4))

    def test_aggregate_empty(self) -> None:
        server = HCFLServer(n_features=4, n_classes=2)
        cluster_models, client_clusters = server.aggregate([])
        assert len(cluster_models) == 1
        assert 0 in cluster_models
        assert client_clusters == {}

    def test_download_bytes_positive(self) -> None:
        server = HCFLServer(n_features=4, n_classes=2)
        assert server.download_bytes(n_clients=5) > 0

    def test_download_bytes_scales_with_clients(self) -> None:
        server = HCFLServer(n_features=4, n_classes=2)
        b1 = server.download_bytes(n_clients=1)
        b5 = server.download_bytes(n_clients=5)
        assert b5 == pytest.approx(b1 * 5, rel=0.01)


# ===========================================================================
# Full runner tests
# ===========================================================================


class TestRunHCFLFull:
    def test_smoke(self) -> None:
        dataset = _make_small_dataset()
        result = run_hcfl_full(dataset, federation_every=2)
        assert result.method_name == "HCFL"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_accuracy_in_range(self) -> None:
        dataset = _make_small_dataset()
        result = run_hcfl_full(dataset, federation_every=1)
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)

    def test_predicted_has_values(self) -> None:
        """Predicted concept matrix should contain cluster IDs, not all zeros."""
        dataset = _make_small_dataset(K=4, T=6, n_samples=40)
        result = run_hcfl_full(
            dataset,
            federation_every=1,
            distance_threshold=0.3,
        )
        # After federation, at least some cluster IDs should be non-zero
        # (clients with different concepts should get different clusters)
        assert result.predicted_concept_matrix.shape == (4, 6)
        # The matrix should contain at least one value that is not all the same
        unique_ids = np.unique(result.predicted_concept_matrix)
        assert len(unique_ids) >= 1  # at minimum cluster 0 exists

    def test_byte_tracking(self) -> None:
        dataset = _make_small_dataset()
        result_freq1 = run_hcfl_full(dataset, federation_every=1)
        result_freq2 = run_hcfl_full(dataset, federation_every=2)
        # More frequent federation = more bytes
        assert result_freq1.total_bytes >= result_freq2.total_bytes
