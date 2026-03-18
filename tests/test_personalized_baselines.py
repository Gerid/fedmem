from __future__ import annotations

import numpy as np

from fedprotrack.baselines.apfl import APFLClient, APFLServer, APFLUpload, run_apfl_full
from fedprotrack.baselines.fedem import FedEMClient, FedEMServer, FedEMUpload, run_fedem_full
from fedprotrack.baselines.pfedme import PFedMeClient, PFedMeServer, PFedMeUpload, run_pfedme_full
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.generator import DriftDataset


def _make_small_dataset(K: int = 2, T: int = 4, n_samples: int = 40) -> DriftDataset:
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


def _make_batch(n: int = 64, d: int = 4, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


class TestPFedMe:
    def test_client_fit_and_upload(self) -> None:
        client = PFedMeClient(0, 4, 2, seed=42)
        X, y = _make_batch()
        assert np.all(client.predict(X) == 0)
        client.fit(X, y)
        assert np.mean(client.predict(X) == y) > 0.55
        upload = client.get_upload()
        assert isinstance(upload, PFedMeUpload)
        assert upload.client_id == 0
        assert upload.n_samples == len(X)
        assert client.upload_bytes() > 0

    def test_server_aggregate_and_broadcast(self) -> None:
        server = PFedMeServer(4, 2, seed=42)
        upload = PFedMeUpload(
            client_id=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            personalized_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            n_samples=10,
        )
        params = server.aggregate([upload])
        assert "coef" in params
        assert server.download_bytes(3) > 0

    def test_full_runner(self) -> None:
        result = run_pfedme_full(_make_small_dataset(), federation_every=2)
        assert result.method_name == "pFedMe"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0


class TestAPFL:
    def test_client_alpha_updates(self) -> None:
        client = APFLClient(0, 4, 2, alpha=0.8, seed=42)
        X, y = _make_batch()
        client.fit(X, y)
        assert 0.0 <= client.alpha <= 1.0
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.55
        upload = client.get_upload()
        assert isinstance(upload, APFLUpload)
        assert upload.client_id == 0
        assert upload.alpha == client.alpha
        assert client.upload_bytes() > 0

    def test_server_aggregate(self) -> None:
        server = APFLServer(4, 2, seed=42)
        upload = APFLUpload(
            client_id=0,
            global_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            alpha=0.5,
            n_samples=10,
        )
        params = server.aggregate([upload])
        assert "coef" in params
        assert server.download_bytes(2) > 0

    def test_full_runner(self) -> None:
        result = run_apfl_full(_make_small_dataset(), federation_every=2)
        assert result.method_name == "APFL"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0


class TestFedEM:
    def test_client_fit_and_responsibilities(self) -> None:
        client = FedEMClient(0, 4, 2, n_components=3, seed=42)
        X, y = _make_batch()
        assert np.all(client.predict(X) == 0)
        client.fit(X, y)
        assert np.isclose(client._responsibilities.sum(), 1.0)
        preds = client.predict(X)
        assert preds.shape == (len(X),)
        assert client.upload_bytes() > 0

    def test_server_aggregate(self) -> None:
        server = FedEMServer(4, 2, n_components=3, seed=42)
        upload = FedEMUpload(
            client_id=0,
            expert_params=[
                {"coef": np.ones(4), "intercept": np.zeros(1)},
                {"coef": np.ones(4) * 2, "intercept": np.zeros(1)},
                {"coef": np.ones(4) * 3, "intercept": np.zeros(1)},
            ],
            responsibilities=np.array([0.2, 0.3, 0.5], dtype=np.float64),
            n_samples=10,
        )
        experts = server.aggregate([upload])
        assert len(experts) == 3
        assert server.download_bytes(2) > 0

    def test_full_runner(self) -> None:
        result = run_fedem_full(_make_small_dataset(), federation_every=2, n_components=3)
        assert result.method_name == "FedEM"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0
