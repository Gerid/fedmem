from __future__ import annotations

"""Tests for FedCCFA-Impl baseline (DBSCAN classifier clustering + feature alignment)."""

import numpy as np
import pytest

from fedprotrack.baselines.fedccfa_impl import (
    FedCCFAImplClient,
    FedCCFAImplServer,
    FedCCFAImplUpload,
    FedCCFAImplUpdate,
    _classifier_rows,
    _cluster_rows_dbscan,
)
from fedprotrack.baselines.runners import run_fedccfa_impl_full
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
# Unit tests: helpers
# ===========================================================================


class TestClassifierRows:
    def test_binary_single_row(self) -> None:
        """Binary classifier with single-row coef expands to 2 rows."""
        params = {
            "coef": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "intercept": np.array([0.5], dtype=np.float64),
        }
        rows = _classifier_rows(params, n_features=4, n_classes=2)
        assert rows.shape == (2, 4)
        np.testing.assert_allclose(rows[0], -rows[1])

    def test_multiclass(self) -> None:
        """Multi-class classifier returns correct shape."""
        params = {
            "coef": np.arange(12, dtype=np.float64),
            "intercept": np.zeros(3, dtype=np.float64),
        }
        rows = _classifier_rows(params, n_features=4, n_classes=3)
        assert rows.shape == (3, 4)


class TestClusterRowsDBSCAN:
    def test_single_row(self) -> None:
        """Single row should be assigned cluster 0."""
        rows = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        labels = _cluster_rows_dbscan(rows, eps=0.5, min_samples=1)
        assert labels.shape == (1,)
        assert labels[0] == 0

    def test_empty(self) -> None:
        """Empty input returns empty labels."""
        rows = np.zeros((0, 3), dtype=np.float64)
        labels = _cluster_rows_dbscan(rows, eps=0.5, min_samples=1)
        assert labels.shape == (0,)

    def test_identical_vectors_same_cluster(self) -> None:
        """Identical vectors should land in the same cluster."""
        rows = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float64)
        labels = _cluster_rows_dbscan(rows, eps=0.5, min_samples=1)
        assert len(set(labels)) == 1

    def test_orthogonal_vectors_separate(self) -> None:
        """Orthogonal vectors with tight eps should separate."""
        rows = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)
        labels = _cluster_rows_dbscan(rows, eps=0.05, min_samples=1)
        assert labels[0] != labels[1]


# ===========================================================================
# Client tests
# ===========================================================================


class TestFedCCFAImplClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = FedCCFAImplClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = FedCCFAImplClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_fit_multiclass(self) -> None:
        client = FedCCFAImplClient(0, 4, 3, seed=42)
        X, y = _make_multiclass_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (90,)
        # Should do better than random (1/3)
        assert np.mean(preds == y) > 0.33

    def test_upload_structure(self) -> None:
        client = FedCCFAImplClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, FedCCFAImplUpload)
        assert upload.client_id == 0
        assert "coef" in upload.model_params
        assert "intercept" in upload.model_params
        assert upload.classifier_rows.shape == (2, 4)
        assert upload.n_samples == 50
        assert len(upload.feature_centroids) > 0

    def test_upload_bytes_positive(self) -> None:
        client = FedCCFAImplClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0

    def test_upload_bytes_scales_with_precision(self) -> None:
        client = FedCCFAImplClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        b32 = client.upload_bytes(precision_bits=32)
        b64 = client.upload_bytes(precision_bits=64)
        assert b64 == pytest.approx(b32 * 2, rel=0.01)

    def test_gamma_validation(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            FedCCFAImplClient(0, 4, 2, gamma=-1.0)

    def test_set_update_changes_cluster_id(self) -> None:
        client = FedCCFAImplClient(0, 4, 2, seed=42)
        assert client.cluster_id == 0
        update = FedCCFAImplUpdate(
            aggregated_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            cluster_id=3,
            cluster_centroids={0: np.ones(4), 1: -np.ones(4)},
        )
        client.set_update(update)
        assert client.cluster_id == 3

    def test_alignment_loss_runs_without_error(self) -> None:
        """Ensure training with alignment centroids doesn't crash."""
        client = FedCCFAImplClient(0, 4, 2, gamma=5.0, seed=42)
        X, y = _make_data()
        # Set cluster centroids before training
        client._cluster_centroids = {
            0: np.array([-1.0, 0.0, 0.0, 0.0]),
            1: np.array([1.0, 0.0, 0.0, 0.0]),
        }
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == y.shape


# ===========================================================================
# Server tests
# ===========================================================================


class TestFedCCFAImplServer:
    def test_aggregate_empty(self) -> None:
        server = FedCCFAImplServer(n_features=4, n_classes=2)
        assert server.aggregate([]) == {}

    def test_single_client(self) -> None:
        server = FedCCFAImplServer(n_features=4, n_classes=2)
        upload = FedCCFAImplUpload(
            client_id=0,
            model_params={
                "coef": np.ones(4, dtype=np.float64),
                "intercept": np.zeros(1, dtype=np.float64),
            },
            classifier_rows=np.array([[-1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float64),
            feature_centroids={0: np.zeros(4), 1: np.ones(4)},
            n_samples=50,
        )
        updates = server.aggregate([upload])
        assert 0 in updates
        assert updates[0].cluster_id >= 0
        assert updates[0].aggregated_params

    def test_similar_clients_same_cluster(self) -> None:
        """Clients with identical classifiers should be clustered together."""
        server = FedCCFAImplServer(
            n_features=4, n_classes=2, dbscan_eps=0.5,
        )
        shared_rows = np.array([[-1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float64)
        u1 = FedCCFAImplUpload(
            client_id=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            classifier_rows=shared_rows.copy(),
            feature_centroids={0: np.zeros(4)},
            n_samples=50,
        )
        u2 = FedCCFAImplUpload(
            client_id=1,
            model_params={"coef": np.ones(4) * 1.01, "intercept": np.zeros(1)},
            classifier_rows=shared_rows.copy(),
            feature_centroids={0: np.zeros(4)},
            n_samples=50,
        )
        updates = server.aggregate([u1, u2])
        assert updates[0].cluster_id == updates[1].cluster_id

    def test_dissimilar_clients_different_clusters(self) -> None:
        """Clients with orthogonal classifiers should be separated (tight eps)."""
        server = FedCCFAImplServer(
            n_features=4, n_classes=2, dbscan_eps=0.05,
        )
        u1 = FedCCFAImplUpload(
            client_id=0,
            model_params={"coef": np.array([1, 0, 0, 0], dtype=np.float64),
                          "intercept": np.zeros(1)},
            classifier_rows=np.array([[-1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float64),
            feature_centroids={},
            n_samples=50,
        )
        u2 = FedCCFAImplUpload(
            client_id=1,
            model_params={"coef": np.array([0, 0, 0, 1], dtype=np.float64),
                          "intercept": np.zeros(1)},
            classifier_rows=np.array([[0, 0, 0, -1], [0, 0, 0, 1]], dtype=np.float64),
            feature_centroids={},
            n_samples=50,
        )
        updates = server.aggregate([u1, u2])
        assert updates[0].cluster_id != updates[1].cluster_id

    def test_cluster_centroids_computed(self) -> None:
        server = FedCCFAImplServer(n_features=4, n_classes=2, dbscan_eps=1.0)
        u1 = FedCCFAImplUpload(
            client_id=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            classifier_rows=np.ones((2, 4)),
            feature_centroids={0: np.ones(4) * 2.0, 1: np.ones(4) * 4.0},
            n_samples=50,
        )
        updates = server.aggregate([u1])
        assert 0 in updates[0].cluster_centroids

    def test_drift_detection(self) -> None:
        server = FedCCFAImplServer(n_features=4, n_classes=2, dbscan_eps=0.05)
        u1 = FedCCFAImplUpload(
            client_id=0,
            model_params={"coef": np.array([1, 0, 0, 0], dtype=np.float64),
                          "intercept": np.zeros(1)},
            classifier_rows=np.array([[-1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float64),
            feature_centroids={},
            n_samples=50,
        )
        server.aggregate([u1])
        first_drift = server.drift_detected

        # Same structure again -- no drift
        server.aggregate([u1])
        assert not server.drift_detected

        # Change client -> new cluster structure
        u2 = FedCCFAImplUpload(
            client_id=0,
            model_params={"coef": np.array([0, 0, 0, 1], dtype=np.float64),
                          "intercept": np.zeros(1)},
            classifier_rows=np.array([[0, 0, 0, -1], [0, 0, 0, 1]], dtype=np.float64),
            feature_centroids={},
            n_samples=50,
        )
        # Add a second client
        u3 = FedCCFAImplUpload(
            client_id=1,
            model_params={"coef": np.array([0, 1, 0, 0], dtype=np.float64),
                          "intercept": np.zeros(1)},
            classifier_rows=np.array([[0, -1, 0, 0], [0, 1, 0, 0]], dtype=np.float64),
            feature_centroids={},
            n_samples=50,
        )
        server.aggregate([u2, u3])
        assert server.drift_detected

    def test_download_bytes_positive(self) -> None:
        server = FedCCFAImplServer(n_features=4, n_classes=2)
        assert server.download_bytes(n_clients=3) > 0

    def test_download_bytes_invalid_precision(self) -> None:
        server = FedCCFAImplServer(n_features=4, n_classes=2)
        with pytest.raises(ValueError, match="precision_bits"):
            server.download_bytes(n_clients=1, precision_bits=0)

    def test_dbscan_eps_validation(self) -> None:
        with pytest.raises(ValueError, match="dbscan_eps"):
            FedCCFAImplServer(n_features=4, n_classes=2, dbscan_eps=0.0)

    def test_dbscan_min_samples_validation(self) -> None:
        with pytest.raises(ValueError, match="dbscan_min_samples"):
            FedCCFAImplServer(n_features=4, n_classes=2, dbscan_min_samples=0)


# ===========================================================================
# Full runner test
# ===========================================================================


class TestRunFedCCFAImplFull:
    def test_smoke(self) -> None:
        dataset = _make_small_dataset()
        result = run_fedccfa_impl_full(dataset, federation_every=2)
        assert result.method_name == "FedCCFA-Impl"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_accuracy_in_range(self) -> None:
        dataset = _make_small_dataset()
        result = run_fedccfa_impl_full(dataset, federation_every=1)
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)

    def test_byte_tracking(self) -> None:
        dataset = _make_small_dataset()
        result_freq1 = run_fedccfa_impl_full(dataset, federation_every=1)
        result_freq2 = run_fedccfa_impl_full(dataset, federation_every=2)
        # More frequent federation = more bytes
        assert result_freq1.total_bytes >= result_freq2.total_bytes
