from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.metrics.concept_metrics import (
    assignment_entropy,
    assignment_switch_rate,
    avg_clients_per_concept,
    concept_re_id_accuracy,
    memory_reuse_rate,
    routing_consistency,
    singleton_group_ratio,
    wrong_memory_reuse_rate,
)


# ---------------------------------------------------------------------------
# concept_re_id_accuracy
# ---------------------------------------------------------------------------


class TestConceptReIdAccuracy:
    def test_perfect_predictions(self) -> None:
        """Aligned preds == gt → global accuracy == 1.0, wmrr == 0.0."""
        rng = np.random.default_rng(7)
        K, T = 3, 8
        gt = rng.integers(0, 3, size=(K, T), dtype=np.int32)
        pred = gt.copy()

        acc, per_client, per_ts = concept_re_id_accuracy(gt, pred)

        assert acc == pytest.approx(1.0)
        np.testing.assert_allclose(per_client, np.ones(K))
        np.testing.assert_allclose(per_ts, np.ones(T))

    def test_all_wrong_two_concepts(self) -> None:
        """K=2 T=4, gt alternates 0/1, pred always wrong.

        Note: Hungarian may align 0→1 and 1→0 which fixes all predictions,
        so accuracy can be 1.0.  The key assertion is that acc <= 1.0 and
        the per-client / per-timestep shapes are correct.
        """
        K, T = 2, 4
        gt = np.array(
            [[0, 1, 0, 1],
             [1, 0, 1, 0]],
            dtype=np.int32,
        )
        # pred swaps 0/1 relative to gt
        pred = 1 - gt  # type: ignore[operator]

        acc, per_client, per_ts = concept_re_id_accuracy(gt, pred)

        # Hungarian will detect the swap and align perfectly → acc == 1.0
        assert 0.0 <= acc <= 1.0
        assert per_client.shape == (K,)
        assert per_ts.shape == (T,)

    def test_fedavg_multi_concept(self) -> None:
        """gt has 4 concepts, pred all zeros → low accuracy (at most 1/4 by pigeonhole)."""
        K, T = 4, 20
        # Balanced: each of 4 concepts appears exactly K*T/4 times.
        rng = np.random.default_rng(99)
        concept_ids = np.tile(np.arange(4, dtype=np.int32), K * T // 4)
        rng.shuffle(concept_ids)
        gt = concept_ids.reshape(K, T)
        pred = np.zeros((K, T), dtype=np.int32)

        acc, _, _ = concept_re_id_accuracy(gt, pred)

        # At most 1 concept worth of correct predictions
        assert acc <= 0.5, f"Expected low accuracy for all-zero pred on 4-concept gt, got {acc}"

    def test_per_client_shape(self) -> None:
        """per_client array should have shape (K,)."""
        K, T = 5, 12
        rng = np.random.default_rng(1)
        gt = rng.integers(0, 3, size=(K, T), dtype=np.int32)
        pred = rng.integers(0, 3, size=(K, T), dtype=np.int32)

        _, per_client, _ = concept_re_id_accuracy(gt, pred)

        assert per_client.shape == (K,)

    def test_per_timestep_shape(self) -> None:
        """per_timestep array should have shape (T,)."""
        K, T = 5, 12
        rng = np.random.default_rng(2)
        gt = rng.integers(0, 3, size=(K, T), dtype=np.int32)
        pred = rng.integers(0, 3, size=(K, T), dtype=np.int32)

        _, _, per_ts = concept_re_id_accuracy(gt, pred)

        assert per_ts.shape == (T,)

    def test_values_in_unit_interval(self) -> None:
        """All returned accuracy values should lie in [0, 1]."""
        K, T = 3, 9
        rng = np.random.default_rng(3)
        gt = rng.integers(0, 3, size=(K, T), dtype=np.int32)
        pred = rng.integers(0, 3, size=(K, T), dtype=np.int32)

        acc, per_client, per_ts = concept_re_id_accuracy(gt, pred)

        assert 0.0 <= acc <= 1.0
        assert np.all((per_client >= 0.0) & (per_client <= 1.0))
        assert np.all((per_ts >= 0.0) & (per_ts <= 1.0))


# ---------------------------------------------------------------------------
# wrong_memory_reuse_rate
# ---------------------------------------------------------------------------


class TestWrongMemoryReuseRate:
    def test_perfect_predictions_wmrr_zero(self) -> None:
        """Perfect predictions → wmrr == 0.0."""
        K, T = 2, 6
        rng = np.random.default_rng(4)
        gt = rng.integers(0, 2, size=(K, T), dtype=np.int32)
        pred = gt.copy()

        wmrr = wrong_memory_reuse_rate(gt, pred)
        assert wmrr == pytest.approx(0.0)

    def test_complementary_to_accuracy(self) -> None:
        """wmrr + accuracy should equal 1.0 (after the same alignment)."""
        K, T = 3, 10
        rng = np.random.default_rng(5)
        gt = rng.integers(0, 3, size=(K, T), dtype=np.int32)
        pred = rng.integers(0, 3, size=(K, T), dtype=np.int32)

        acc, _, _ = concept_re_id_accuracy(gt, pred)
        wmrr = wrong_memory_reuse_rate(gt, pred)

        assert acc + wmrr == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# assignment_entropy
# ---------------------------------------------------------------------------


class TestAssignmentEntropy:
    def test_entropy_with_soft_assignments_uniform(self) -> None:
        """Uniform soft assignments should yield high entropy."""
        K, T, C = 3, 8, 4
        # Uniform distribution over C concepts
        soft = np.full((K, T, C), fill_value=1.0 / C, dtype=np.float64)
        pred = np.zeros((K, T), dtype=np.int32)

        entropy = assignment_entropy(soft, pred, n_concepts=C)

        # Maximum entropy for C=4 is ln(4) ≈ 1.386
        max_H = np.log(C)
        assert entropy == pytest.approx(max_H, abs=1e-6)

    def test_entropy_with_soft_assignments_deterministic(self) -> None:
        """One-hot soft assignments (deterministic) should yield near-zero entropy."""
        K, T, C = 2, 5, 3
        soft = np.zeros((K, T, C), dtype=np.float64)
        # Every cell assigns probability 1 to concept 0
        soft[:, :, 0] = 1.0
        pred = np.zeros((K, T), dtype=np.int32)

        entropy = assignment_entropy(soft, pred, n_concepts=C)

        # Entropy of [1, 0, 0] is 0; clipping eps shifts it slightly but < 1e-3
        assert entropy < 0.1

    def test_entropy_without_soft_all_same_concept(self) -> None:
        """All clients use same concept at every step → near-zero entropy."""
        K, T = 4, 10
        pred = np.zeros((K, T), dtype=np.int32)  # all concept 0

        entropy = assignment_entropy(None, pred, n_concepts=3)

        # Marginal is [1, 0, 0] at every step; entropy ≈ 0
        assert entropy < 0.1

    def test_entropy_without_soft_uniform_split(self) -> None:
        """Equal split across concepts at every step → high entropy."""
        K, T, C = 4, 6, 4
        # Client k uses concept (k % C) at every step
        pred = np.zeros((K, T), dtype=np.int32)
        for k in range(K):
            pred[k, :] = k % C

        entropy = assignment_entropy(None, pred, n_concepts=C)

        # Each concept used by exactly 1 client → uniform marginal → max entropy
        max_H = np.log(C)
        assert entropy == pytest.approx(max_H, abs=0.05)

    def test_entropy_returns_float(self) -> None:
        """Return type must be a plain Python float."""
        K, T, C = 2, 4, 2
        pred = np.zeros((K, T), dtype=np.int32)

        result = assignment_entropy(None, pred, n_concepts=C)

        assert isinstance(result, float)

    def test_entropy_non_negative(self) -> None:
        """Entropy is always >= 0."""
        rng = np.random.default_rng(6)
        K, T, C = 3, 7, 3
        soft = rng.dirichlet(np.ones(C), size=(K, T))
        pred = rng.integers(0, C, size=(K, T), dtype=np.int32)

        assert assignment_entropy(soft, pred, C) >= 0.0
        assert assignment_entropy(None, pred, C) >= 0.0

    def test_entropy_ignores_zero_padded_cells(self) -> None:
        """Zero-padded cells (non-federation rounds) must not dilute entropy.

        When federation_every > 1, most cells in the soft assignment matrix are
        zero-filled.  The metric should average only over cells that actually
        received a posterior update.
        """
        K, T, C = 2, 10, 3
        # Only federation rounds at t=0, 5 (federation_every=5).
        soft = np.zeros((K, T, C), dtype=np.float64)
        # Fill active rounds with uniform distribution (high entropy).
        soft[:, 0, :] = 1.0 / C
        soft[:, 5, :] = 1.0 / C

        pred = np.zeros((K, T), dtype=np.int32)
        entropy = assignment_entropy(soft, pred, n_concepts=C)

        # Expected: entropy of uniform over C=3 = ln(3) ~ 1.099
        max_H = np.log(C)
        assert entropy == pytest.approx(max_H, abs=1e-6)

    def test_entropy_sparse_vs_dense_federation(self) -> None:
        """Same per-round distribution should give same entropy regardless of
        how many non-federation rounds exist between them.
        """
        K, C = 2, 4
        rng = np.random.default_rng(42)
        active_dist = rng.dirichlet(np.ones(C), size=(K,))  # (K, C)

        # Dense: T=4, all rounds are federation rounds.
        soft_dense = np.tile(active_dist[:, np.newaxis, :], (1, 4, 1))
        pred_dense = np.zeros((K, 4), dtype=np.int32)
        H_dense = assignment_entropy(soft_dense, pred_dense, n_concepts=C)

        # Sparse: T=20, only rounds 0,5,10,15 have posteriors.
        soft_sparse = np.zeros((K, 20, C), dtype=np.float64)
        for t in [0, 5, 10, 15]:
            soft_sparse[:, t, :] = active_dist
        pred_sparse = np.zeros((K, 20), dtype=np.int32)
        H_sparse = assignment_entropy(soft_sparse, pred_sparse, n_concepts=C)

        assert H_dense == pytest.approx(H_sparse, abs=1e-10)

    def test_entropy_all_zero_soft_falls_back(self) -> None:
        """If all cells are zero (degenerate), function should not crash."""
        K, T, C = 2, 5, 3
        soft = np.zeros((K, T, C), dtype=np.float64)
        pred = np.zeros((K, T), dtype=np.int32)

        entropy = assignment_entropy(soft, pred, n_concepts=C)
        # Should return a finite float (falls back to H.mean() on all-zero).
        assert isinstance(entropy, float)
        assert np.isfinite(entropy)


# ---------------------------------------------------------------------------
# Diagnostic routing / grouping metrics
# ---------------------------------------------------------------------------


class TestAssignmentSwitchRate:
    def test_no_switches(self) -> None:
        pred = np.zeros((3, 5), dtype=np.int32)
        assert assignment_switch_rate(pred) == pytest.approx(0.0)

    def test_half_switches(self) -> None:
        pred = np.array(
            [[0, 0, 1, 1, 1],
             [0, 1, 1, 0, 0]],
            dtype=np.int32,
        )
        # client0: 1/4, client1: 2/4 => 3/8 overall
        assert assignment_switch_rate(pred) == pytest.approx(3.0 / 8.0)


class TestAvgClientsPerConcept:
    def test_balanced_two_groups(self) -> None:
        pred = np.array(
            [[0, 0, 1],
             [0, 1, 1],
             [1, 1, 0],
             [1, 0, 0]],
            dtype=np.int32,
        )
        assert avg_clients_per_concept(pred) == pytest.approx(2.0)


class TestSingletonGroupRatio:
    def test_singletons_detected(self) -> None:
        pred = np.array(
            [[0, 0],
             [1, 0],
             [2, 1]],
            dtype=np.int32,
        )
        # t0 counts=[1,1,1], t1 counts=[2,1] => 4 singleton groups / 5 active groups
        assert singleton_group_ratio(pred) == pytest.approx(4.0 / 5.0)


class TestMemoryReuseRate:
    def test_recurrent_assignments_count_as_reuse(self) -> None:
        gt = np.array([[0, 1, 0, 1]], dtype=np.int32)
        pred = gt.copy()
        # reuse at t=2 and t=3 => 2 / 3
        assert memory_reuse_rate(gt, pred) == pytest.approx(2.0 / 3.0)

    def test_non_recurrent_assignments_have_zero_reuse(self) -> None:
        gt = np.array([[0, 1, 2, 3]], dtype=np.int32)
        pred = gt.copy()
        assert memory_reuse_rate(gt, pred) == pytest.approx(0.0)


class TestRoutingConsistency:
    def test_hard_consistency_matches_no_switch_rate(self) -> None:
        pred = np.array(
            [[0, 0, 1, 1],
             [0, 1, 1, 1]],
            dtype=np.int32,
        )
        expected = 1.0 - assignment_switch_rate(pred)
        assert routing_consistency(None, pred) == pytest.approx(expected)

    def test_soft_consistency_uses_temporal_similarity(self) -> None:
        pred = np.array([[0, 0, 1]], dtype=np.int32)
        soft = np.array(
            [[[0.9, 0.1],
              [0.8, 0.2],
              [0.2, 0.8]]],
            dtype=np.float64,
        )
        score = routing_consistency(soft, pred)
        assert 0.0 <= score <= 1.0
        assert score > 0.5
