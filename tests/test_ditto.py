from __future__ import annotations

import numpy as np

from fedprotrack.baselines.ditto import (
    DittoClient,
    DittoServer,
    DittoUpload,
    run_ditto_full,
)
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.generator import DriftDataset


def _make_data(
    n: int = 50, d: int = 4, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def _make_small_dataset(
    K: int = 2, T: int = 4, n_samples: int = 40
) -> DriftDataset:
    rng = np.random.RandomState(123)
    concept_matrix = np.zeros((K, T), dtype=np.int32)
    concept_matrix[:, T // 2 :] = 1
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            X = rng.randn(n_samples, 4).astype(np.float64)
            if concept_matrix[k, t] == 0:
                y = (X[:, 0] > 0).astype(np.int64)
            else:
                y = (X[:, 1] > 0).astype(np.int64)
            data[(k, t)] = (X, y)
    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=GeneratorConfig(K=K, T=T),
        concept_specs=[],
    )


class TestDittoClient:
    def test_predict_before_fit(self) -> None:
        client = DittoClient(0, 4, 2, seed=42)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = DittoClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        acc = float(np.mean(client.predict(X) == y))
        assert acc > 0.5

    def test_upload_structure(self) -> None:
        client = DittoClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, DittoUpload)
        assert upload.client_id == 0
        assert upload.n_samples == len(X)
        assert "coef" in upload.model_params
        assert "intercept" in upload.model_params

    def test_upload_bytes_positive(self) -> None:
        client = DittoClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0

    def test_personalization_differs_from_global(self) -> None:
        client = DittoClient(0, 4, 2, lamda=0.1, tau=5, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        global_params = client._global_model.get_params()
        personal_params = client._personal_model.get_params()
        # After personalisation the two models should differ
        differs = any(
            not np.allclose(global_params[k], personal_params[k])
            for k in global_params
        )
        assert differs


class TestDittoServer:
    def test_init_creates_params(self) -> None:
        server = DittoServer(4, 2, seed=42)
        assert "coef" in server.global_params
        assert "intercept" in server.global_params

    def test_aggregate_empty(self) -> None:
        server = DittoServer(4, 2, seed=42)
        params = server.aggregate([])
        assert "coef" in params
        assert np.allclose(params["coef"], server.global_params["coef"])

    def test_aggregate_weighted(self) -> None:
        server = DittoServer(4, 2, seed=42)
        upload_a = DittoUpload(
            client_id=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            n_samples=10,
        )
        upload_b = DittoUpload(
            client_id=1,
            model_params={"coef": np.ones(4) * 3, "intercept": np.zeros(1)},
            n_samples=30,
        )
        params = server.aggregate([upload_a, upload_b])
        # Weighted average: (10*1 + 30*3) / 40 = 100/40 = 2.5
        assert np.allclose(params["coef"], 2.5)


class TestRunDittoFull:
    def test_smoke(self) -> None:
        ds = _make_small_dataset()
        result = run_ditto_full(ds, federation_every=2)
        assert result.method_name == "Ditto"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)

    def test_accuracy_in_range(self) -> None:
        ds = _make_small_dataset()
        result = run_ditto_full(ds, federation_every=1)
        mean_acc = result.accuracy_matrix.mean()
        assert 0.0 <= mean_acc <= 1.0

    def test_bytes_positive(self) -> None:
        ds = _make_small_dataset()
        result = run_ditto_full(ds, federation_every=1)
        assert result.total_bytes > 0
