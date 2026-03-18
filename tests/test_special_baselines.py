from __future__ import annotations

"""Focused tests for ATP, FLUX, and FLUX-prior."""

import numpy as np

from fedprotrack.baselines.atp import ATPClient, ATPServer, ATPUpload, run_atp_full
from fedprotrack.baselines.flux import (
    FLUXClient,
    FLUXServer,
    FLUXPriorServer,
    FLUXUpload,
    run_flux_full,
)
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.generator import DriftDataset


def _make_drift_dataset(K: int = 4, T: int = 4, n_samples: int = 40) -> DriftDataset:
    """Small synthetic drift stream for runner-style smoke tests."""
    rng = np.random.RandomState(7)
    concept_matrix = np.zeros((K, T), dtype=np.int32)
    concept_matrix[:, T // 2 :] = 1
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            X = rng.randn(n_samples, 4).astype(np.float64)
            if k < K // 2:
                if concept_matrix[k, t] == 0:
                    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(np.int64)
                else:
                    y = (X[:, 1] > 0).astype(np.int64)
            else:
                if concept_matrix[k, t] == 0:
                    y = (X[:, 1] > 0).astype(np.int64)
                else:
                    y = (X[:, 0] + 0.25 * X[:, 1] > 0).astype(np.int64)
            data[(k, t)] = (X, y)
    cfg = GeneratorConfig(K=K, T=T)
    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=cfg,
        concept_specs=[],
    )


def _make_batch(
    seed: int = 0,
    shift: float = 0.0,
    n: int = 50,
    d: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    X[:, 0] += shift
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)
    return X, y


class TestATPClient:
    def test_fit_and_predict(self) -> None:
        client = ATPClient(0, 4, 2, seed=42)
        X, y = _make_batch(seed=1)
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (len(X),)
        assert np.mean(preds == y) > 0.6

    def test_adaptation_rates_change_after_drift(self) -> None:
        client = ATPClient(0, 4, 2, seed=42)
        X1, y1 = _make_batch(seed=2)
        X2, y2 = _make_batch(seed=3, shift=1.5)
        client.fit(X1, y1)
        initial = client.adaptation_rates
        client.fit(X2, y2)
        updated = client.adaptation_rates
        assert updated.shape == initial.shape
        assert np.all(updated >= 0.0)
        assert not np.allclose(initial, updated)

    def test_upload_and_server_aggregate(self) -> None:
        client_a = ATPClient(0, 4, 2, seed=1)
        client_b = ATPClient(1, 4, 2, seed=2)
        Xa, ya = _make_batch(seed=4)
        Xb, yb = _make_batch(seed=5, shift=1.0)
        client_a.fit(Xa, ya)
        client_b.fit(Xb, yb)
        upload_a = client_a.get_upload()
        upload_b = client_b.get_upload()
        assert isinstance(upload_a, ATPUpload)
        server = ATPServer(4, 2, seed=42)
        update = server.aggregate([upload_a, upload_b])
        assert "coef" in update.model_params
        assert update.adaptation_rates.shape == upload_a.adaptation_rates.shape


class TestATPRunner:
    def test_run_atp_full_smoke(self) -> None:
        dataset = _make_drift_dataset(K=4, T=4, n_samples=30)
        result = run_atp_full(dataset, federation_every=2)
        assert result.method_name == "ATP"
        assert result.accuracy_matrix.shape == (4, 4)
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)
        assert result.total_bytes > 0.0


class TestFLUXClient:
    def test_fit_and_predict(self) -> None:
        client = FLUXClient(0, 4, 2, seed=42)
        X, y = _make_batch(seed=10)
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (len(X),)
        assert np.mean(preds == y) > 0.6

    def test_descriptor_shape(self) -> None:
        client = FLUXClient(0, 4, 2, seed=42)
        X, y = _make_batch(seed=11, shift=0.5)
        desc = client.descriptor(X, y)
        assert desc.ndim == 1
        assert desc.size == 4 + 4 + 2 + 8

    def test_upload_structure(self) -> None:
        client = FLUXClient(0, 4, 2, seed=42)
        X, y = _make_batch(seed=12)
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, FLUXUpload)
        assert upload.descriptor.size > 0
        assert "coef" in upload.model_params


class TestFLUXServer:
    def test_cluster_clients_without_prior(self) -> None:
        server = FLUXServer(4, 2, seed=42)
        clients = [
            FLUXClient(0, 4, 2, seed=0),
            FLUXClient(1, 4, 2, seed=1),
            FLUXClient(2, 4, 2, seed=2),
            FLUXClient(3, 4, 2, seed=3),
        ]
        uploads = []
        for idx, client in enumerate(clients):
            shift = 0.0 if idx < 2 else 2.0
            X, y = _make_batch(seed=20 + idx, shift=shift)
            client.fit(X, y)
            uploads.append(client.get_upload())
        updates = server.aggregate(uploads)
        assert len(updates) == 4
        cluster_ids = {update.cluster_id for update in updates.values()}
        assert len(cluster_ids) >= 2

    def test_cluster_clients_with_prior(self) -> None:
        server = FLUXPriorServer(4, 2, n_clusters=2, seed=42)
        clients = [
            FLUXClient(0, 4, 2, seed=0),
            FLUXClient(1, 4, 2, seed=1),
            FLUXClient(2, 4, 2, seed=2),
            FLUXClient(3, 4, 2, seed=3),
        ]
        uploads = []
        for idx, client in enumerate(clients):
            shift = 0.0 if idx < 2 else 2.0
            X, y = _make_batch(seed=30 + idx, shift=shift)
            client.fit(X, y)
            uploads.append(client.get_upload())
        updates = server.aggregate(uploads)
        cluster_ids = {update.cluster_id for update in updates.values()}
        assert len(cluster_ids) == 2
        assert server.download_bytes(updates) > 0.0


class TestFLUXRunner:
    def test_run_flux_full_smoke(self) -> None:
        dataset = _make_drift_dataset(K=4, T=4, n_samples=30)
        result = run_flux_full(dataset, federation_every=2)
        assert result.method_name == "FLUX"
        assert result.accuracy_matrix.shape == (4, 4)
        assert result.predicted_cluster_matrix.shape == (4, 4)
        assert result.total_bytes > 0.0

    def test_run_flux_prior_smoke(self) -> None:
        dataset = _make_drift_dataset(K=4, T=4, n_samples=30)
        result = run_flux_full(dataset, federation_every=2, prior_n_clusters=2)
        assert result.method_name == "FLUX-prior"
        assert result.accuracy_matrix.shape == (4, 4)
        assert np.unique(result.predicted_cluster_matrix).size >= 1

