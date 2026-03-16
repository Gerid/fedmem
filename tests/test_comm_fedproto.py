from __future__ import annotations

"""Tests for fedprotrack.baselines comm_tracker and fedproto modules."""

import numpy as np
import pytest

from fedprotrack.baselines.comm_tracker import (
    fingerprint_bytes,
    model_bytes,
    prototype_bytes,
)
from fedprotrack.baselines.fedproto import (
    ClientProtoUpload,
    FedProtoAggregator,
    FedProtoClient,
)


# ---------------------------------------------------------------------------
# comm_tracker tests
# ---------------------------------------------------------------------------


class TestModelBytes:
    def test_model_bytes_empty(self) -> None:
        """Empty parameter dict should return 0.0."""
        assert model_bytes({}) == 0.0

    def test_model_bytes_known(self) -> None:
        """Single (4,) float32 array: 4 elements * 4 bytes = 16 bytes."""
        params = {"w": np.zeros((4,), dtype=np.float32)}
        assert model_bytes(params, precision_bits=32) == 16.0

    def test_model_bytes_multiple_arrays(self) -> None:
        """Sum across all arrays: (4 + 6) elements * 4 bytes = 40 bytes."""
        params = {
            "w": np.zeros((4,), dtype=np.float32),
            "b": np.zeros((6,), dtype=np.float32),
        }
        assert model_bytes(params, precision_bits=32) == 40.0

    def test_model_bytes_64bit(self) -> None:
        """2 elements at 64 bits = 16 bytes."""
        params = {"w": np.zeros((2,))}
        assert model_bytes(params, precision_bits=64) == 16.0

    def test_model_bytes_invalid_precision(self) -> None:
        with pytest.raises(ValueError):
            model_bytes({"w": np.zeros((4,))}, precision_bits=0)


class TestPrototypeBytes:
    def test_prototype_bytes_known(self) -> None:
        """2 classes each with 3-dimensional prototype: 6 elements * 4 bytes = 24."""
        protos = {
            0: np.zeros((3,), dtype=np.float32),
            1: np.zeros((3,), dtype=np.float32),
        }
        assert prototype_bytes(protos, precision_bits=32) == 24.0

    def test_prototype_bytes_empty(self) -> None:
        assert prototype_bytes({}) == 0.0

    def test_prototype_bytes_different_precisions(self) -> None:
        protos = {0: np.zeros((4,))}
        assert prototype_bytes(protos, precision_bits=16) == 8.0
        assert prototype_bytes(protos, precision_bits=64) == 32.0

    def test_prototype_bytes_invalid_precision(self) -> None:
        with pytest.raises(ValueError):
            prototype_bytes({0: np.zeros((4,))}, precision_bits=-1)


class TestFingerprintBytes:
    def test_fingerprint_bytes_basic(self) -> None:
        """n_features=4, n_classes=3: (4+3) * 4 bytes = 28 bytes."""
        assert fingerprint_bytes(n_features=4, n_classes=3, precision_bits=32) == 28.0

    def test_fingerprint_bytes_64bit(self) -> None:
        """n_features=4, n_classes=3 at 64-bit: 7 * 8 = 56 bytes."""
        assert fingerprint_bytes(n_features=4, n_classes=3, precision_bits=64) == 56.0

    def test_fingerprint_bytes_minimum(self) -> None:
        """n_features=1, n_classes=1: 2 * 4 = 8 bytes."""
        assert fingerprint_bytes(n_features=1, n_classes=1, precision_bits=32) == 8.0

    def test_fingerprint_bytes_invalid_precision(self) -> None:
        with pytest.raises(ValueError):
            fingerprint_bytes(n_features=4, n_classes=2, precision_bits=0)

    def test_fingerprint_bytes_invalid_n_features(self) -> None:
        with pytest.raises(ValueError):
            fingerprint_bytes(n_features=0, n_classes=2, precision_bits=32)

    def test_fingerprint_bytes_invalid_n_classes(self) -> None:
        with pytest.raises(ValueError):
            fingerprint_bytes(n_features=4, n_classes=0, precision_bits=32)


# ---------------------------------------------------------------------------
# FedProto tests
# ---------------------------------------------------------------------------


class TestFedProtoClientNoPrototypes:
    def test_fedproto_no_prototypes_predict(self) -> None:
        """Before set_global_prototypes is called, predict returns all zeros."""
        client = FedProtoClient(client_id=0, n_features=4, n_classes=2)
        X = np.random.default_rng(42).random((10, 4))
        preds = client.predict(X)
        assert preds.shape == (10,)
        assert np.all(preds == 0)


class TestFedProtoClientFit:
    def test_fedproto_fit_updates_prototypes(self) -> None:
        """After fit, local prototype for class c equals per-class mean of X."""
        rng = np.random.default_rng(0)
        X = rng.random((20, 3))
        y = np.array([0] * 10 + [1] * 10)

        client = FedProtoClient(client_id=0, n_features=3, n_classes=2)
        client.fit(X, y)

        upload = client.get_upload()
        expected_proto_0 = X[:10].mean(axis=0)
        expected_proto_1 = X[10:].mean(axis=0)

        np.testing.assert_allclose(upload.prototypes[0], expected_proto_0)
        np.testing.assert_allclose(upload.prototypes[1], expected_proto_1)
        assert upload.class_counts[0] == 10
        assert upload.class_counts[1] == 10
        assert upload.n_samples == 20

    def test_fedproto_fit_incremental(self) -> None:
        """Two sequential fit calls should yield the same mean as a single call."""
        rng = np.random.default_rng(1)
        X = rng.random((20, 4))
        y = np.zeros(20, dtype=int)

        # Single-batch fit
        client_single = FedProtoClient(client_id=0, n_features=4, n_classes=1)
        client_single.fit(X, y)

        # Two-batch incremental fit
        client_incr = FedProtoClient(client_id=0, n_features=4, n_classes=1)
        client_incr.fit(X[:10], y[:10])
        client_incr.fit(X[10:], y[10:])

        np.testing.assert_allclose(
            client_single.get_upload().prototypes[0],
            client_incr.get_upload().prototypes[0],
            atol=1e-12,
        )


class TestFedProtoClientPredict:
    def test_fedproto_predict_nearest(self) -> None:
        """Samples closer to class-0 prototype are predicted as 0, and vice versa."""
        client = FedProtoClient(client_id=0, n_features=2, n_classes=2)

        global_protos = {
            0: np.array([0.0, 0.0]),
            1: np.array([10.0, 10.0]),
        }
        client.set_global_prototypes(global_protos)

        X = np.array([
            [0.1, 0.1],   # close to class 0
            [9.9, 9.9],   # close to class 1
            [0.5, 0.5],   # still closer to class 0
            [9.0, 9.0],   # still closer to class 1
        ])
        preds = client.predict(X)
        expected = np.array([0, 1, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(preds, expected)

    def test_fedproto_predict_single_class(self) -> None:
        """When only one global prototype exists, all samples map to that class."""
        client = FedProtoClient(client_id=0, n_features=3, n_classes=2)
        client.set_global_prototypes({5: np.array([1.0, 2.0, 3.0])})

        X = np.random.default_rng(7).random((8, 3))
        preds = client.predict(X)
        assert np.all(preds == 5)

    def test_fedproto_upload_bytes(self) -> None:
        """Upload bytes equals n_classes * n_features * 4 bytes."""
        client = FedProtoClient(client_id=0, n_features=5, n_classes=2)
        rng = np.random.default_rng(2)
        X = rng.random((20, 5))
        y = np.array([0] * 10 + [1] * 10)
        client.fit(X, y)

        # 2 classes * 5 features * 4 bytes/float = 40 bytes
        assert client.upload_bytes(precision_bits=32) == 40.0

    def test_fedproto_upload_bytes_empty(self) -> None:
        """Before fit, no prototypes exist, so upload_bytes = 0."""
        client = FedProtoClient(client_id=0, n_features=5, n_classes=2)
        assert client.upload_bytes() == 0.0


class TestFedProtoAggregator:
    def test_fedproto_aggregate_weighted(self) -> None:
        """Weighted average: class-0 global proto = weighted mean of two clients."""
        # Client 0: 10 samples, proto [1.0, 0.0]
        # Client 1:  5 samples, proto [4.0, 0.0]
        # Expected global: (10*1.0 + 5*4.0) / 15 = 30/15 = 2.0
        upload0 = ClientProtoUpload(
            client_id=0,
            prototypes={0: np.array([1.0, 0.0])},
            class_counts={0: 10},
            n_samples=10,
        )
        upload1 = ClientProtoUpload(
            client_id=1,
            prototypes={0: np.array([4.0, 0.0])},
            class_counts={0: 5},
            n_samples=5,
        )

        aggregator = FedProtoAggregator()
        global_protos = aggregator.aggregate([upload0, upload1])

        np.testing.assert_allclose(global_protos[0], np.array([2.0, 0.0]))

    def test_fedproto_aggregate_disjoint_classes(self) -> None:
        """Each client has a different class; both prototypes appear in result."""
        upload0 = ClientProtoUpload(
            client_id=0,
            prototypes={0: np.array([1.0, 2.0])},
            class_counts={0: 8},
            n_samples=8,
        )
        upload1 = ClientProtoUpload(
            client_id=1,
            prototypes={1: np.array([3.0, 4.0])},
            class_counts={1: 4},
            n_samples=4,
        )

        aggregator = FedProtoAggregator()
        global_protos = aggregator.aggregate([upload0, upload1])

        np.testing.assert_allclose(global_protos[0], np.array([1.0, 2.0]))
        np.testing.assert_allclose(global_protos[1], np.array([3.0, 4.0]))

    def test_fedproto_aggregate_empty(self) -> None:
        """Empty upload list returns empty dict."""
        aggregator = FedProtoAggregator()
        assert aggregator.aggregate([]) == {}

    def test_fedproto_aggregate_single_client(self) -> None:
        """Single client: global proto equals that client's proto."""
        proto_val = np.array([5.0, 6.0, 7.0])
        upload = ClientProtoUpload(
            client_id=0,
            prototypes={2: proto_val},
            class_counts={2: 20},
            n_samples=20,
        )
        aggregator = FedProtoAggregator()
        global_protos = aggregator.aggregate([upload])
        np.testing.assert_allclose(global_protos[2], proto_val)

    def test_fedproto_download_bytes(self) -> None:
        """Download bytes = per-client bytes * n_clients."""
        global_protos = {
            0: np.zeros((4,)),
            1: np.zeros((4,)),
        }
        aggregator = FedProtoAggregator()
        # 2 classes * 4 features * 4 bytes = 32 bytes per client * 3 clients = 96
        assert aggregator.download_bytes(global_protos, n_clients=3) == 96.0

    def test_fedproto_download_bytes_zero_clients(self) -> None:
        """Zero clients → 0 bytes."""
        global_protos = {0: np.zeros((4,))}
        aggregator = FedProtoAggregator()
        assert aggregator.download_bytes(global_protos, n_clients=0) == 0.0

    def test_fedproto_download_bytes_invalid(self) -> None:
        aggregator = FedProtoAggregator()
        with pytest.raises(ValueError):
            aggregator.download_bytes({0: np.zeros((4,))}, n_clients=-1)


class TestFedProtoEndToEnd:
    def test_end_to_end_round(self) -> None:
        """Full federation round: fit -> upload -> aggregate -> set -> predict."""
        rng = np.random.default_rng(99)
        n_features = 4
        n_classes = 2

        # Two well-separated clusters
        X0 = rng.random((30, n_features)) + np.array([0.0, 0.0, 0.0, 0.0])
        X1 = rng.random((30, n_features)) + np.array([5.0, 5.0, 5.0, 5.0])
        y0 = np.zeros(30, dtype=int)
        y1 = np.ones(30, dtype=int)

        clients = [
            FedProtoClient(i, n_features, n_classes) for i in range(2)
        ]
        clients[0].fit(np.vstack([X0[:15], X1[:15]]), np.hstack([y0[:15], y1[:15]]))
        clients[1].fit(np.vstack([X0[15:], X1[15:]]), np.hstack([y0[15:], y1[15:]]))

        uploads = [c.get_upload() for c in clients]
        aggregator = FedProtoAggregator()
        global_protos = aggregator.aggregate(uploads)

        for c in clients:
            c.set_global_prototypes(global_protos)

        # Test queries clearly in each cluster
        X_test = np.vstack([
            np.ones((5, n_features)) * 0.5,   # near class-0 proto (~0)
            np.ones((5, n_features)) * 5.5,   # near class-1 proto (~5)
        ])
        y_test = np.array([0] * 5 + [1] * 5)

        for client in clients:
            preds = client.predict(X_test)
            accuracy = (preds == y_test).mean()
            assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy:.0%}"
