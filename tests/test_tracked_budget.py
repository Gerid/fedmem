from __future__ import annotations

"""Tests for TrackedSummary and budget sweep utilities."""

import numpy as np
import pytest

from fedprotrack.baselines.tracked_summary import (
    TrackedSummaryClient,
    TrackedSummaryServer,
    TrackedUpload,
)
from fedprotrack.baselines.budget_sweep import (
    BudgetPoint,
    find_crossover_points,
    run_budget_sweep,
)
from fedprotrack.drift_generator.generator import DriftDataset
from fedprotrack.drift_generator.configs import GeneratorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_upload(
    client_id: int,
    fv: np.ndarray,
    n_samples: int = 10,
    n_features: int = 4,
    n_classes: int = 2,
) -> TrackedUpload:
    """Build a minimal TrackedUpload with a dummy model_params dict."""
    model_params = {
        "coef": np.zeros(n_features * (1 if n_classes == 2 else n_classes)),
        "intercept": np.zeros(1 if n_classes == 2 else n_classes),
    }
    return TrackedUpload(
        client_id=client_id,
        fingerprint_vector=fv,
        model_params=model_params,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
    )


def _make_small_dataset(K: int = 2, T: int = 4, n_samples: int = 30) -> DriftDataset:
    """Create a tiny mock DriftDataset without calling the full generator."""
    rng = np.random.RandomState(42)
    concept_matrix = np.zeros((K, T), dtype=np.int32)
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            X = rng.randn(n_samples, 4).astype(np.float64)
            y = (X[:, 0] > 0).astype(np.int64)
            data[(k, t)] = (X, y)
    config = GeneratorConfig(K=K, T=T)
    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=config,
        concept_specs=[],
    )


# ---------------------------------------------------------------------------
# TrackedSummaryServer tests
# ---------------------------------------------------------------------------

class TestTrackedSummaryServer:
    """Tests for the server-side clustering and aggregation."""

    def test_server_single_cluster(self) -> None:
        """Two clients with very similar fingerprints end up in one cluster."""
        n_features, n_classes = 4, 2
        # Nearly identical vectors
        fv_base = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5], dtype=np.float64)
        fv_a = fv_base + 1e-4 * np.ones_like(fv_base)
        fv_b = fv_base - 1e-4 * np.ones_like(fv_base)

        upload_a = _make_upload(0, fv_a, n_features=n_features, n_classes=n_classes)
        upload_b = _make_upload(1, fv_b, n_features=n_features, n_classes=n_classes)

        server = TrackedSummaryServer(similarity_threshold=0.9)
        result = server.aggregate([upload_a, upload_b])

        # Both clients must receive aggregated params
        assert 0 in result, "Client 0 missing from result"
        assert 1 in result, "Client 1 missing from result"
        # Both should receive the same (averaged) params since they are in one cluster
        np.testing.assert_allclose(
            result[0]["coef"], result[1]["coef"],
            err_msg="Clients in the same cluster should receive identical coef",
        )

    def test_server_two_clusters(self) -> None:
        """Two clients with orthogonal fingerprints end up in separate clusters."""
        n_features, n_classes = 4, 2
        # Orthogonal vectors — cosine similarity == 0
        fv_a = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        fv_b = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        upload_a = _make_upload(
            0, fv_a, n_samples=5, n_features=n_features, n_classes=n_classes
        )
        upload_b = _make_upload(
            1, fv_b, n_samples=5, n_features=n_features, n_classes=n_classes
        )

        server = TrackedSummaryServer(similarity_threshold=0.5)
        result = server.aggregate([upload_a, upload_b])

        assert 0 in result, "Client 0 missing from result"
        assert 1 in result, "Client 1 missing from result"
        # The two clusters have different params (no cross-averaging happened)
        # Each cluster has only one member, so its own params are returned as-is.
        np.testing.assert_allclose(
            result[0]["coef"],
            upload_a.model_params["coef"],
            err_msg="Client 0 should get back its own params (solo cluster)",
        )
        np.testing.assert_allclose(
            result[1]["coef"],
            upload_b.model_params["coef"],
            err_msg="Client 1 should get back its own params (solo cluster)",
        )

    def test_server_empty_uploads(self) -> None:
        """Empty upload list returns an empty dict."""
        server = TrackedSummaryServer()
        assert server.aggregate([]) == {}

    def test_server_download_bytes_positive(self) -> None:
        """download_bytes returns a positive float for non-trivial uploads."""
        n_features, n_classes = 4, 2
        fv = np.ones(n_features + n_classes, dtype=np.float64)
        uploads = [_make_upload(i, fv, n_features=n_features, n_classes=n_classes) for i in range(3)]
        server = TrackedSummaryServer()
        b = server.download_bytes(uploads)
        assert b > 0.0


# ---------------------------------------------------------------------------
# BudgetPoint dataclass tests
# ---------------------------------------------------------------------------

class TestBudgetPoint:
    def test_budget_point_dataclass(self) -> None:
        """BudgetPoint can be constructed and fields are accessible."""
        bp = BudgetPoint(
            method_name="FedAvg-Full",
            federation_every=5,
            total_bytes=1024.0,
            accuracy_auc=0.75,
        )
        assert bp.method_name == "FedAvg-Full"
        assert bp.federation_every == 5
        assert bp.total_bytes == 1024.0
        assert bp.accuracy_auc == 0.75


# ---------------------------------------------------------------------------
# find_crossover_points tests
# ---------------------------------------------------------------------------

class TestFindCrossoverPoints:
    def test_crossover_no_intersection(self) -> None:
        """Curves that never intersect return an empty list."""
        # Method A always better than B
        points_a = [
            BudgetPoint("A", 1, 100.0, 0.9),
            BudgetPoint("A", 2, 200.0, 0.85),
            BudgetPoint("A", 5, 500.0, 0.80),
        ]
        points_b = [
            BudgetPoint("B", 1, 100.0, 0.5),
            BudgetPoint("B", 2, 200.0, 0.45),
            BudgetPoint("B", 5, 500.0, 0.40),
        ]
        crossovers = find_crossover_points(points_a, points_b)
        assert crossovers == [], f"Expected no crossovers, got {crossovers}"

    def test_crossover_clear_intersection(self) -> None:
        """Two curves with a clear single crossing produce one crossover point."""
        # A starts high, ends low; B starts low, ends high — they cross around 300
        points_a = [
            BudgetPoint("A", 1, 100.0, 0.9),
            BudgetPoint("A", 2, 300.0, 0.5),
            BudgetPoint("A", 5, 600.0, 0.2),
        ]
        points_b = [
            BudgetPoint("B", 1, 100.0, 0.2),
            BudgetPoint("B", 2, 300.0, 0.5),
            BudgetPoint("B", 5, 600.0, 0.9),
        ]
        crossovers = find_crossover_points(points_a, points_b)
        assert len(crossovers) >= 1, f"Expected at least 1 crossover, got {crossovers}"
        cross_bytes, cross_auc = crossovers[0]
        # Crossover should be somewhere around 300 bytes
        assert 100.0 <= cross_bytes <= 600.0, (
            f"Crossover bytes {cross_bytes} outside expected range [100, 600]"
        )
        assert 0.0 <= cross_auc <= 1.0, (
            f"Crossover auc {cross_auc} outside [0, 1]"
        )

    def test_crossover_too_few_points(self) -> None:
        """Returns empty list when either input has fewer than 2 points."""
        single = [BudgetPoint("X", 1, 100.0, 0.5)]
        multi = [BudgetPoint("Y", 1, 100.0, 0.5), BudgetPoint("Y", 2, 200.0, 0.6)]
        assert find_crossover_points(single, multi) == []
        assert find_crossover_points(multi, single) == []


# ---------------------------------------------------------------------------
# run_budget_sweep integration / smoke test
# ---------------------------------------------------------------------------

class TestRunBudgetSweep:
    def test_run_budget_sweep_smoke(self) -> None:
        """Smoke test on a tiny dataset: 3 methods x 4 federation_every = 12 points."""
        dataset = _make_small_dataset(K=2, T=4, n_samples=30)
        federation_every_values = [1, 2, 3, 4]

        results = run_budget_sweep(
            dataset,
            federation_every_values=federation_every_values,
            similarity_threshold=0.5,
        )

        # 3 methods × 4 federation_every values
        assert len(results) == 12, f"Expected 12 BudgetPoints, got {len(results)}"

        method_names = {bp.method_name for bp in results}
        assert "FedAvg-Full" in method_names
        assert "FedProto" in method_names
        assert "TrackedSummary" in method_names

        for bp in results:
            assert isinstance(bp, BudgetPoint)
            assert bp.total_bytes >= 0.0, (
                f"{bp.method_name} fe={bp.federation_every}: total_bytes={bp.total_bytes}"
            )
            assert 0.0 <= bp.accuracy_auc <= 4.0, (
                # trapz on 4 steps with acc in [0,1] can range [0, ~3]
                f"{bp.method_name} fe={bp.federation_every}: accuracy_auc={bp.accuracy_auc}"
            )

    def test_run_budget_sweep_bytes_positive_when_federated(self) -> None:
        """Methods that federate at every step should produce positive total_bytes."""
        dataset = _make_small_dataset(K=2, T=4, n_samples=30)
        results = run_budget_sweep(dataset, federation_every_values=[1])
        for bp in results:
            assert bp.total_bytes > 0.0, (
                f"{bp.method_name} fe=1 should have total_bytes > 0, got {bp.total_bytes}"
            )

    def test_run_budget_sweep_default_values(self) -> None:
        """Default federation_every_values produce 12 points (3 methods x 4)."""
        dataset = _make_small_dataset(K=2, T=10, n_samples=20)
        results = run_budget_sweep(dataset)
        assert len(results) == 12

    def test_run_budget_sweep_accuracy_auc_range(self) -> None:
        """accuracy_auc is non-negative for all methods."""
        dataset = _make_small_dataset(K=2, T=4, n_samples=30)
        results = run_budget_sweep(dataset, federation_every_values=[2])
        for bp in results:
            assert bp.accuracy_auc >= 0.0, (
                f"{bp.method_name}: accuracy_auc={bp.accuracy_auc} should be >= 0"
            )
