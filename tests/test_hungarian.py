from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.metrics.hungarian import align_predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid(K: int, T: int, concept_list: list[int]) -> np.ndarray:
    """Broadcast a flat concept list into a (K, T) grid filled column-wise."""
    arr = np.array(concept_list, dtype=np.int32)
    assert arr.size == K * T, "concept_list length must equal K*T"
    return arr.reshape(K, T)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlignPredictions:
    def test_identity_permutation(self) -> None:
        """When pred == gt, aligned should equal gt exactly and mapping is identity."""
        K, T = 3, 6
        gt = np.tile(np.arange(3, dtype=np.int32), (K, 1))  # each row [0,1,2,0,1,2]
        # Create gt shaped (K, T) with varying concepts
        rng = np.random.default_rng(42)
        gt = rng.integers(0, 3, size=(K, T), dtype=np.int32)
        pred = gt.copy()

        aligned, mapping = align_predictions(gt, pred)

        np.testing.assert_array_equal(aligned, gt)
        assert all(mapping[k] == k for k in mapping), f"Expected identity mapping, got {mapping}"

    def test_simple_swap(self) -> None:
        """K=1 T=4, gt=[0,0,1,1], pred=[1,1,0,0] — after alignment should match gt."""
        K, T = 1, 4
        gt = np.array([[0, 0, 1, 1]], dtype=np.int32)
        pred = np.array([[1, 1, 0, 0]], dtype=np.int32)

        aligned, mapping = align_predictions(gt, pred)

        np.testing.assert_array_equal(aligned, gt)
        # mapping must swap: 1→0 and 0→1
        assert mapping[1] == 0
        assert mapping[0] == 1

    def test_rectangular_more_preds(self) -> None:
        """pred has 3 concepts, gt has 2 — the surplus pred concept maps to -1."""
        K, T = 2, 6
        # gt uses only concepts 0 and 1
        gt = np.array(
            [[0, 0, 0, 1, 1, 1],
             [0, 0, 1, 1, 0, 1]],
            dtype=np.int32,
        )
        # pred uses concepts 0, 1, 2 — concept 2 is extra
        pred = np.array(
            [[0, 0, 0, 1, 1, 2],
             [0, 2, 1, 1, 0, 1]],
            dtype=np.int32,
        )

        aligned, mapping = align_predictions(gt, pred)

        # The unmatched concept (2) must map to -1
        assert mapping[2] == -1, f"Expected mapping[2]==-1, got {mapping[2]}"
        # Cells predicted as 2 must become -1 in aligned
        assert np.all(aligned[pred == 2] == -1)
        # All other cells should still be valid (0 or 1)
        valid_mask = pred != 2
        assert np.all(aligned[valid_mask] >= 0)

    def test_fedavg_baseline(self) -> None:
        """pred all zeros, gt has 2 concepts — 0 should map to the majority gt concept."""
        K, T = 2, 8
        # gt: first 5 cols are concept 1, last 3 cols are concept 0
        gt = np.array(
            [[1, 1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 0, 0, 0, 1, 1]],
            dtype=np.int32,
        )
        # pred is all zeros (classic FedAvg — single global model)
        pred = np.zeros((K, T), dtype=np.int32)

        aligned, mapping = align_predictions(gt, pred)

        # Only one pred concept (0), it must map to *some* gt concept
        assert 0 in mapping
        matched_gt = mapping[0]
        assert matched_gt in (0, 1), f"Unexpected mapped gt concept: {matched_gt}"

        # Verify aligned values are all equal to mapped_gt (no -1 since no surplus)
        assert np.all(aligned == matched_gt)

    def test_perfect_alignment_accuracy(self) -> None:
        """After alignment, a perfectly correct pred should yield 100% match."""
        rng = np.random.default_rng(0)
        K, T = 4, 10
        gt = rng.integers(0, 4, size=(K, T), dtype=np.int32)
        pred = gt.copy()

        aligned, _ = align_predictions(gt, pred)
        accuracy = float((aligned == gt).mean())
        assert accuracy == 1.0

    def test_single_concept_each(self) -> None:
        """Trivial case: one concept in gt and one in pred."""
        K, T = 3, 5
        gt = np.zeros((K, T), dtype=np.int32)
        pred = np.zeros((K, T), dtype=np.int32)

        aligned, mapping = align_predictions(gt, pred)

        np.testing.assert_array_equal(aligned, gt)
        assert mapping[0] == 0
