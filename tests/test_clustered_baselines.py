from __future__ import annotations

"""Tests for CFL, FeSEM, and FedRC adapters."""

import numpy as np

from fedprotrack.baselines.cfl import CFLClient, CFLServer, CFLUpload
from fedprotrack.baselines.fedrc import FedRCClient, FedRCServer, FedRCUpload
from fedprotrack.baselines.fesem import FeSEMClient, FeSEMServer


def _make_binary_data(
    n: int = 64,
    d: int = 4,
    seed: int = 42,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (X[:, axis] > 0).astype(np.int64)
    return X, y


def _linear_params(scale: float, n_features: int = 4) -> dict[str, np.ndarray]:
    return {
        "coef": np.asarray([scale] * n_features, dtype=np.float64),
        "intercept": np.asarray([0.0], dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# CFL
# ---------------------------------------------------------------------------


class TestCFL:
    def test_client_upload_contains_update_vector(self) -> None:
        client = CFLClient(0, 4, 2, seed=7)
        X, y = _make_binary_data(seed=7)
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, CFLUpload)
        assert upload.client_id == 0
        assert upload.n_samples == len(X)
        assert upload.update_vector.ndim == 1
        assert np.linalg.norm(upload.update_vector) > 0.0

    def test_server_splits_one_cluster_into_two(self) -> None:
        server = CFLServer(
            n_features=4,
            n_classes=2,
            warmup_rounds=0,
            eps_1=0.1,
            eps_2=1.0,
            max_clusters=4,
            seed=0,
        )
        base = _linear_params(0.0)
        uploads = [
            CFLUpload(0, base, np.array([2.0, 0.0, 0.0, 0.0, 0.0]), 10, 0),
            CFLUpload(1, base, np.array([1.5, 0.0, 0.0, 0.0, 0.0]), 10, 0),
            CFLUpload(2, base, np.array([-2.0, 0.0, 0.0, 0.0, 0.0]), 10, 0),
            CFLUpload(3, base, np.array([-1.5, 0.0, 0.0, 0.0, 0.0]), 10, 0),
        ]

        models = server.aggregate(uploads, round_idx=50)
        assert len(models) == 2
        assert len(set(server.client_cluster_map.values())) == 2

    def test_client_round_trip_prediction(self) -> None:
        client = CFLClient(0, 4, 2, seed=11)
        X, y = _make_binary_data(seed=11)
        client.set_model_params(_linear_params(0.0))
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == y.shape
        assert np.mean(preds == y) > 0.5


# ---------------------------------------------------------------------------
# FeSEM
# ---------------------------------------------------------------------------


class TestFeSEM:
    def test_client_prefers_better_cluster_model(self) -> None:
        client = FeSEMClient(0, 4, 2, seed=13)
        server = FeSEMServer(n_clusters=2, n_features=4, n_classes=2, seed=13)
        client.set_cluster_models(
            [
                _linear_params(-5.0),
                _linear_params(5.0),
            ]
        )
        X, y = _make_binary_data(seed=13)
        client.fit(X, y)
        upload = client.get_upload()
        assert upload.selected_cluster == 1
        assert np.mean(client.predict(X) == y) > 0.5
        assert isinstance(server, FeSEMServer)

    def test_server_aggregates_cluster_members_independently(self) -> None:
        server = FeSEMServer(n_clusters=2, n_features=4, n_classes=2, seed=21)
        uploads = [
            FeSEMClient(0, 4, 2, seed=21),
            FeSEMClient(1, 4, 2, seed=22),
        ]
        X0, y0 = _make_binary_data(seed=21)
        X1, y1 = _make_binary_data(seed=22, axis=1)
        for client, X, y in [(uploads[0], X0, y0), (uploads[1], X1, y1)]:
            client.set_cluster_models(
                [
                    _linear_params(-4.0),
                    _linear_params(4.0),
                ]
            )
            client.fit(X, y)

        result = server.aggregate([c.get_upload() for c in uploads])
        assert len(result) == 2
        assert all("coef" in params for params in result)


# ---------------------------------------------------------------------------
# FedRC
# ---------------------------------------------------------------------------


class TestFedRC:
    def test_client_soft_assignment_prefers_matching_cluster(self) -> None:
        client = FedRCClient(0, 4, 2, n_clusters=2, seed=31)
        client.set_cluster_state(
            [
                _linear_params(-6.0),
                _linear_params(6.0),
            ],
            [
                np.array([0.8, 0.2], dtype=np.float64),
                np.array([0.2, 0.8], dtype=np.float64),
            ],
        )
        X, y = _make_binary_data(seed=31)
        client.fit(X, y)
        upload = client.get_upload()
        assert np.isclose(upload.cluster_probs.sum(), 1.0)
        assert upload.selected_cluster == 1
        assert np.mean(client.predict(X) == y) > 0.5

    def test_server_aggregates_by_cluster_probability(self) -> None:
        server = FedRCServer(n_features=4, n_classes=2, n_clusters=2, seed=41)
        upload_a = FedRCUpload(
            client_id=0,
            model_params=_linear_params(0.0),
            cluster_probs=np.array([0.9, 0.1], dtype=np.float64),
            label_hist=np.array([0.8, 0.2], dtype=np.float64),
            n_samples=20,
            batch_size=20,
            selected_cluster=0,
        )
        upload_b = FedRCUpload(
            client_id=1,
            model_params=_linear_params(4.0),
            cluster_probs=np.array([0.1, 0.9], dtype=np.float64),
            label_hist=np.array([0.2, 0.8], dtype=np.float64),
            n_samples=20,
            batch_size=20,
            selected_cluster=1,
        )

        result = server.aggregate([upload_a, upload_b])
        assert len(result) == 2
        assert result[1]["coef"].mean() > result[0]["coef"].mean()
        assert np.isclose(server.cluster_label_hists[0].sum(), 1.0)
        assert np.isclose(server.cluster_label_hists[1].sum(), 1.0)

