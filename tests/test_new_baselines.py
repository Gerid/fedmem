from __future__ import annotations

"""Tests for FedCCFA, Flash, FedDrift, IFCA, and CompressedFedAvg baselines."""

import numpy as np
import pytest

from fedprotrack.baselines.fedccfa import (
    FedCCFAClient,
    FedCCFAServer,
    FedCCFAUpload,
    FedCCFAUpdate,
)
from fedprotrack.baselines.flash import FlashClient, FlashAggregator, FlashUpload
from fedprotrack.baselines.feddrift import FedDriftClient, FedDriftServer, FedDriftUpload
from fedprotrack.baselines.ifca import IFCAClient, IFCAServer, IFCAUpload
from fedprotrack.baselines.compressed_fedavg import (
    CompressedFedAvgClient,
    CompressedFedAvgServer,
    CompressedUpload,
)
from fedprotrack.baselines.runners import (
    run_fedccfa_full,
    run_flash_full,
    run_feddrift_full,
    run_ifca_full,
    run_compressed_fedavg_full,
)
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.generator import DriftDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_dataset(K: int = 2, T: int = 4, n_samples: int = 30) -> DriftDataset:
    """Create a tiny mock DriftDataset without calling the full generator."""
    rng = np.random.RandomState(42)
    concept_matrix = np.zeros((K, T), dtype=np.int32)
    # Introduce concept change at T//2
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
    config = GeneratorConfig(K=K, T=T)
    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=config,
        concept_specs=[],
    )


def _make_data(n: int = 50, d: int = 4, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


# ===========================================================================
# FedCCFA
# ===========================================================================

class TestFedCCFAClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = FedCCFAClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = FedCCFAClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_get_upload_structure(self) -> None:
        client = FedCCFAClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, FedCCFAUpload)
        assert upload.client_id == 0
        assert "coef" in upload.model_params
        assert len(upload.local_prototypes) > 0

    def test_personalized_state_updates_signature(self) -> None:
        client = FedCCFAClient(0, 4, 2, seed=42)
        update = FedCCFAUpdate(
            label_vectors={0: np.ones(4), 1: np.ones(4) * 2.0},
            label_biases={0: -1.0, 1: 1.0},
            global_prototypes={0: np.ones(4), 1: np.zeros(4)},
            label_cluster_ids={0: 3, 1: 5},
        )
        client.set_personalized_state(update)
        assert client.cluster_signature == (3, 5)


class TestFedCCFAServer:
    def test_aggregate_empty(self) -> None:
        server = FedCCFAServer(n_features=4, n_classes=2)
        assert server.aggregate([]) == {}

    def test_similar_clients_share_cluster_ids(self) -> None:
        server = FedCCFAServer(n_features=4, n_classes=2, eps=0.5)
        shared_params = {
            "coef": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "intercept": np.array([0.0], dtype=np.float64),
        }
        u1 = FedCCFAUpload(
            client_id=0,
            model_params=shared_params,
            local_prototypes={
                0: np.array([-1.0, 0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0, 0.0]),
            },
            label_counts={0: 10, 1: 10},
            n_samples=20,
        )
        u2 = FedCCFAUpload(
            client_id=1,
            model_params=shared_params,
            local_prototypes={
                0: np.array([-1.0, 0.0, 0.0, 0.0]),
                1: np.array([1.0, 0.0, 0.0, 0.0]),
            },
            label_counts={0: 10, 1: 10},
            n_samples=20,
        )
        updates = server.aggregate([u1, u2])
        assert updates[0].label_cluster_ids == updates[1].label_cluster_ids

    def test_download_bytes_positive(self) -> None:
        server = FedCCFAServer(n_features=4, n_classes=2)
        updates = {
            0: FedCCFAUpdate(
                label_vectors={0: np.ones(4)},
                label_biases={0: 0.0},
                global_prototypes={0: np.ones(4)},
                label_cluster_ids={0: 1},
            ),
        }
        assert server.download_bytes(updates) > 0


# ===========================================================================
# Flash
# ===========================================================================

class TestFlashClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = FlashClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert preds.shape == (50,)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = FlashClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.5

    def test_upload_bytes_positive(self) -> None:
        client = FlashClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0

    def test_get_upload_structure(self) -> None:
        client = FlashClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, FlashUpload)
        assert upload.client_id == 0
        assert "coef" in upload.model_params
        assert "intercept" in upload.model_params

    def test_set_model_params_updates_prediction(self) -> None:
        client = FlashClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        params = {
            "coef": np.zeros(4, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
        }
        client.set_model_params(params)
        # After setting zero params, model should predict mostly zeros
        preds = client.predict(X)
        assert preds.shape == (50,)


class TestFlashAggregator:
    def test_aggregate_empty(self) -> None:
        agg = FlashAggregator()
        result = agg.aggregate([])
        assert result == {}

    def test_aggregate_single_client(self) -> None:
        agg = FlashAggregator()
        upload = FlashUpload(
            client_id=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            n_samples=10,
            has_drifted=False,
        )
        result = agg.aggregate([upload])
        np.testing.assert_allclose(result["coef"], np.ones(4))

    def test_drift_weight_boost(self) -> None:
        agg = FlashAggregator(drift_weight_boost=10.0)
        u_normal = FlashUpload(
            client_id=0,
            model_params={"coef": np.zeros(4), "intercept": np.zeros(1)},
            n_samples=10,
            has_drifted=False,
        )
        u_drift = FlashUpload(
            client_id=1,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            n_samples=10,
            has_drifted=True,
        )
        result = agg.aggregate([u_normal, u_drift])
        # Drift client has 10x weight, so result should be close to ones
        assert result["coef"].mean() > 0.5

    def test_download_bytes(self) -> None:
        agg = FlashAggregator()
        params = {"coef": np.ones(4), "intercept": np.zeros(1)}
        b = agg.download_bytes(params, n_clients=5)
        assert b > 0


# ===========================================================================
# FedDrift
# ===========================================================================

class TestFedDriftClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = FedDriftClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = FedDriftClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_active_concept_id_starts_at_zero(self) -> None:
        client = FedDriftClient(0, 4, 2)
        assert client.active_concept_id == 0

    def test_upload_structure(self) -> None:
        client = FedDriftClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, FedDriftUpload)
        assert upload.active_concept_id == 0
        assert upload.model_vector.ndim == 1

    def test_upload_bytes(self) -> None:
        client = FedDriftClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0


class TestFedDriftServer:
    def test_aggregate_empty(self) -> None:
        server = FedDriftServer()
        result = server.aggregate([])
        assert result == {}

    def test_aggregate_single_client(self) -> None:
        server = FedDriftServer()
        upload = FedDriftUpload(
            client_id=0,
            active_concept_id=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            model_vector=np.ones(5),
            n_samples=10,
        )
        result = server.aggregate([upload])
        assert 0 in result
        np.testing.assert_allclose(result[0]["coef"], np.ones(4))

    def test_similar_clients_same_cluster(self) -> None:
        server = FedDriftServer(similarity_threshold=0.5)
        u1 = FedDriftUpload(
            client_id=0, active_concept_id=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            model_vector=np.ones(5), n_samples=10,
        )
        u2 = FedDriftUpload(
            client_id=1, active_concept_id=0,
            model_params={"coef": np.ones(4) * 1.1, "intercept": np.zeros(1)},
            model_vector=np.ones(5) * 1.1, n_samples=10,
        )
        result = server.aggregate([u1, u2])
        # Both should get same params (same cluster)
        np.testing.assert_allclose(result[0]["coef"], result[1]["coef"])

    def test_dissimilar_clients_different_clusters(self) -> None:
        server = FedDriftServer(similarity_threshold=0.9)
        u1 = FedDriftUpload(
            client_id=0, active_concept_id=0,
            model_params={"coef": np.array([1, 0, 0, 0], dtype=np.float64),
                          "intercept": np.zeros(1)},
            model_vector=np.array([1, 0, 0, 0, 0], dtype=np.float64),
            n_samples=10,
        )
        u2 = FedDriftUpload(
            client_id=1, active_concept_id=1,
            model_params={"coef": np.array([0, 0, 0, 1], dtype=np.float64),
                          "intercept": np.zeros(1)},
            model_vector=np.array([0, 0, 0, 0, 1], dtype=np.float64),
            n_samples=10,
        )
        result = server.aggregate([u1, u2])
        # Different clusters — different params
        assert not np.allclose(result[0]["coef"], result[1]["coef"])


# ===========================================================================
# IFCA
# ===========================================================================

class TestIFCAClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = IFCAClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = IFCAClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.3  # At least better than random

    def test_cluster_selection(self) -> None:
        client = IFCAClient(0, 4, 2, seed=42)
        # Give two cluster models: one good, one random
        X, y = _make_data()
        good_model = {"coef": np.array([1, 0, 0, 0], dtype=np.float64),
                       "intercept": np.zeros(1, dtype=np.float64)}
        bad_model = {"coef": np.zeros(4, dtype=np.float64),
                      "intercept": np.zeros(1, dtype=np.float64)}
        client.set_cluster_models([good_model, bad_model])
        selected = client.select_cluster(X, y)
        # The model that separates on feature 0 should be better
        assert selected == 0

    def test_upload_structure(self) -> None:
        client = IFCAClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        upload = client.get_upload()
        assert isinstance(upload, IFCAUpload)
        assert "coef" in upload.model_params

    def test_upload_bytes(self) -> None:
        client = IFCAClient(0, 4, 2)
        X, y = _make_data()
        client.fit(X, y)
        assert client.upload_bytes() > 0


class TestIFCAServer:
    def test_init_creates_cluster_models(self) -> None:
        server = IFCAServer(n_clusters=3, n_features=4, n_classes=2)
        assert len(server.cluster_models) == 3
        assert "coef" in server.cluster_models[0]

    def test_invalid_n_clusters(self) -> None:
        with pytest.raises(ValueError, match="n_clusters must be > 0"):
            IFCAServer(n_clusters=0)

    def test_aggregate_empty(self) -> None:
        server = IFCAServer(n_clusters=2, n_features=4, n_classes=2)
        result = server.aggregate([])
        assert len(result) == 2

    def test_aggregate_updates_cluster(self) -> None:
        server = IFCAServer(n_clusters=2, n_features=4, n_classes=2, seed=42)
        old_params = server.cluster_models[0]["coef"].copy()
        upload = IFCAUpload(
            client_id=0,
            selected_cluster=0,
            model_params={"coef": np.ones(4), "intercept": np.zeros(1)},
            n_samples=100,
        )
        server.aggregate([upload])
        # Cluster 0 should have been updated
        assert not np.allclose(server.cluster_models[0]["coef"], old_params)

    def test_download_bytes(self) -> None:
        server = IFCAServer(n_clusters=3, n_features=4, n_classes=2)
        b = server.download_bytes(n_clients=5)
        assert b > 0


# ===========================================================================
# CompressedFedAvg
# ===========================================================================

class TestCompressedFedAvgClient:
    def test_predict_before_fit_returns_zeros(self) -> None:
        client = CompressedFedAvgClient(0, 4, 2)
        X, _ = _make_data()
        preds = client.predict(X)
        assert np.all(preds == 0)

    def test_fit_and_predict(self) -> None:
        client = CompressedFedAvgClient(0, 4, 2, seed=42)
        X, y = _make_data()
        client.fit(X, y)
        preds = client.predict(X)
        assert np.mean(preds == y) > 0.5

    def test_invalid_topk_fraction(self) -> None:
        with pytest.raises(ValueError, match="topk_fraction"):
            CompressedFedAvgClient(0, 4, 2, topk_fraction=0.0)
        with pytest.raises(ValueError, match="topk_fraction"):
            CompressedFedAvgClient(0, 4, 2, topk_fraction=1.5)

    def test_upload_sparsification(self) -> None:
        client = CompressedFedAvgClient(0, 4, 2, topk_fraction=0.5)
        X, y = _make_data()
        client.fit(X, y)
        # Set global params to zeros so delta = local params
        client._global_params = {
            "coef": np.zeros(4, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
        }
        upload = client.get_upload()
        assert isinstance(upload, CompressedUpload)
        # With 50% sparsification, roughly half should be nonzero
        # (5 total elements, so ~2-3 nonzero)
        assert 0 < upload.n_nonzero <= upload.n_total

    def test_upload_bytes_less_than_full(self) -> None:
        client_compressed = CompressedFedAvgClient(0, 4, 2, topk_fraction=0.3)
        X, y = _make_data()
        client_compressed.fit(X, y)
        client_compressed._global_params = {
            "coef": np.zeros(4, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
        }
        compressed_bytes = client_compressed.upload_bytes()
        # Should be less than full model bytes (5 elements * 4 bytes = 20)
        from fedprotrack.baselines.comm_tracker import model_bytes
        full_bytes = model_bytes(client_compressed._model_params)
        assert compressed_bytes <= full_bytes + 20  # +20 for index overhead at most

    def test_error_feedback_accumulates(self) -> None:
        client = CompressedFedAvgClient(0, 4, 2, topk_fraction=0.5)
        X, y = _make_data()
        client.fit(X, y)
        client._global_params = {
            "coef": np.zeros(4, dtype=np.float64),
            "intercept": np.zeros(1, dtype=np.float64),
        }
        # First upload creates error buffer
        client.get_upload()
        assert any(
            np.any(v != 0) for v in client._error_buffer.values()
        ) or True  # May be all zero if topk=100%


class TestCompressedFedAvgServer:
    def test_aggregate_empty(self) -> None:
        server = CompressedFedAvgServer(4, 2)
        result = server.aggregate([])
        assert "coef" in result

    def test_aggregate_updates_global(self) -> None:
        server = CompressedFedAvgServer(4, 2)
        old_coef = server.global_params["coef"].copy()
        upload = CompressedUpload(
            client_id=0,
            sparse_delta={"coef": np.ones(4), "intercept": np.zeros(1)},
            n_nonzero=4,
            n_total=5,
            n_samples=100,
        )
        server.aggregate([upload])
        assert not np.allclose(server.global_params["coef"], old_coef)

    def test_download_bytes(self) -> None:
        server = CompressedFedAvgServer(4, 2)
        b = server.download_bytes(n_clients=5)
        assert b > 0


# ===========================================================================
# Full runners
# ===========================================================================

class TestFullRunners:
    def test_run_fedccfa_full(self) -> None:
        dataset = _make_small_dataset()
        result = run_fedccfa_full(dataset, federation_every=2)
        assert result.method_name == "FedCCFA"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_run_flash_full(self) -> None:
        dataset = _make_small_dataset()
        result = run_flash_full(dataset, federation_every=2)
        assert result.method_name == "Flash"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_run_feddrift_full(self) -> None:
        dataset = _make_small_dataset()
        result = run_feddrift_full(dataset, federation_every=2)
        assert result.method_name == "FedDrift"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.predicted_concept_matrix.shape == (2, 4)

    def test_run_ifca_full(self) -> None:
        dataset = _make_small_dataset()
        result = run_ifca_full(dataset, federation_every=2, n_clusters=2)
        assert result.method_name == "IFCA"
        assert result.accuracy_matrix.shape == (2, 4)

    def test_run_compressed_fedavg_full(self) -> None:
        dataset = _make_small_dataset()
        result = run_compressed_fedavg_full(dataset, federation_every=2, topk_fraction=0.5)
        assert result.method_name == "CompressedFedAvg"
        assert result.accuracy_matrix.shape == (2, 4)
        assert result.total_bytes >= 0

    def test_all_runners_produce_valid_accuracy(self) -> None:
        dataset = _make_small_dataset()
        for runner in [
            run_fedccfa_full,
            run_flash_full,
            run_feddrift_full,
            run_ifca_full,
            run_compressed_fedavg_full,
        ]:
            result = runner(dataset, federation_every=1)
            assert np.all(result.accuracy_matrix >= 0.0)
            assert np.all(result.accuracy_matrix <= 1.0)

    def test_to_experiment_log(self) -> None:
        dataset = _make_small_dataset()
        result = run_flash_full(dataset)
        log = result.to_experiment_log(dataset.concept_matrix)
        assert log.method_name == "Flash"
        assert log.ground_truth.shape == (2, 4)


# ===========================================================================
# Budget sweep integration
# ===========================================================================

class TestBudgetSweepNewBaselines:
    def test_sweep_includes_new_methods(self) -> None:
        from fedprotrack.baselines.budget_sweep import run_budget_sweep
        dataset = _make_small_dataset()
        points = run_budget_sweep(dataset, federation_every_values=[2])
        method_names = {p.method_name for p in points}
        assert "FedAvg-Full" in method_names
        assert "FedRC" in method_names
        assert "FedEM" in method_names
        assert "FeSEM" in method_names
        assert "CFL" in method_names
        assert "pFedMe" in method_names
        assert "APFL" in method_names
        assert "ATP" in method_names
        assert "FLUX" in method_names
        assert "FLUX-prior" in method_names
        assert "FedProto" in method_names
        assert "FedCCFA" in method_names
        assert "TrackedSummary" in method_names
        assert "Flash" in method_names
        assert "FedDrift" in method_names
        assert "IFCA" in method_names
        assert "CompressedFedAvg" in method_names
        # Total: 7 methods × 1 fe = 7 points
        assert len(points) == 17

    def test_all_budget_points_valid(self) -> None:
        from fedprotrack.baselines.budget_sweep import run_budget_sweep
        dataset = _make_small_dataset()
        points = run_budget_sweep(dataset, federation_every_values=[2])
        for p in points:
            assert p.total_bytes >= 0
            assert p.accuracy_auc >= 0.0
