"""Tests for trust-weighted centroid estimation in DynamicMemoryBank."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.posterior.memory_bank import (
    DynamicMemoryBank,
    MemoryBankConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp(
    n_features: int = 8,
    n_classes: int = 2,
    n_samples: int = 30,
    mean_shift: float = 0.0,
    seed: int = 0,
) -> ConceptFingerprint:
    """Create a fingerprint from synthetic data with controllable mean."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) + mean_shift
    y = rng.randint(0, n_classes, size=n_samples)
    fp = ConceptFingerprint(n_features, n_classes)
    fp.update(X, y)
    return fp


def _make_separated_fps(
    n_per_concept: int = 3,
    n_features: int = 8,
    n_classes: int = 2,
    seed: int = 42,
) -> list[ConceptFingerprint]:
    """Create two groups of well-separated fingerprints with tight clusters.

    Group A: mean = [+50, 0, 0, ...] (first half positive)
    Group B: mean = [0, +50, 0, ...] (second half positive)
    Both groups have large norms so within-group cosine distance is tiny.
    Returns [A0, A1, A2, B0, B1, B2].
    """
    fps = []
    base_a = np.zeros(n_features)
    base_a[:n_features // 2] = 50.0
    base_b = np.zeros(n_features)
    base_b[n_features // 2:] = 50.0

    rng = np.random.RandomState(seed)
    for i in range(n_per_concept):
        X = rng.randn(50, n_features) + base_a
        y = rng.randint(0, n_classes, size=50)
        fp = ConceptFingerprint(n_features, n_classes)
        fp.update(X, y)
        fps.append(fp)
    for i in range(n_per_concept):
        X = rng.randn(50, n_features) + base_b
        y = rng.randint(0, n_classes, size=50)
        fp = ConceptFingerprint(n_features, n_classes)
        fp.update(X, y)
        fps.append(fp)
    return fps


def _make_similar_fps(
    n_clients: int = 4,
    n_features: int = 8,
    n_classes: int = 2,
    seed: int = 7,
) -> list[ConceptFingerprint]:
    """Create fingerprints from the same distribution (single concept)."""
    return [
        _make_fp(n_features=n_features, n_classes=n_classes,
                 mean_shift=0.0, seed=seed + i)
        for i in range(n_clients)
    ]


def _make_trust_bank(
    n_features: int = 8,
    n_classes: int = 2,
    **extra_cfg: object,
) -> DynamicMemoryBank:
    """Create a memory bank with trust estimation enabled."""
    cfg = MemoryBankConfig(enable_trust_estimation=True, **extra_cfg)  # type: ignore[arg-type]
    return DynamicMemoryBank(config=cfg, n_features=n_features, n_classes=n_classes)


def _make_legacy_bank(
    n_features: int = 8,
    n_classes: int = 2,
) -> DynamicMemoryBank:
    """Create a legacy memory bank (trust estimation disabled)."""
    cfg = MemoryBankConfig(enable_trust_estimation=False)
    return DynamicMemoryBank(config=cfg, n_features=n_features, n_classes=n_classes)


# ---------------------------------------------------------------------------
# Test: batch_bootstrap separates distinct concepts
# ---------------------------------------------------------------------------

class TestBatchBootstrap:
    def test_batch_bootstrap_separates_concepts(self) -> None:
        """K=6 fingerprints from 2 well-separated concepts -> 2 slots."""
        fps = _make_separated_fps(n_per_concept=3)
        bank = _make_trust_bank()

        assignments = bank.batch_bootstrap(fps)

        assert len(assignments) == 6
        # Should produce exactly 2 distinct concept IDs
        unique_concepts = set(assignments)
        assert len(unique_concepts) == 2, (
            f"Expected 2 clusters from well-separated data, got {len(unique_concepts)}"
        )

        # Clients 0-2 (concept A) should share one ID, 3-5 (concept B) another
        group_a_ids = set(assignments[:3])
        group_b_ids = set(assignments[3:])
        assert len(group_a_ids) == 1, f"Group A should be one concept, got {group_a_ids}"
        assert len(group_b_ids) == 1, f"Group B should be one concept, got {group_b_ids}"
        assert group_a_ids != group_b_ids, "The two groups should have different concept IDs"

    def test_batch_bootstrap_single_concept(self) -> None:
        """K=4 similar fingerprints -> 1 slot."""
        fps = _make_similar_fps(n_clients=4)
        bank = _make_trust_bank()

        assignments = bank.batch_bootstrap(fps)

        assert len(assignments) == 4
        unique_concepts = set(assignments)
        assert len(unique_concepts) == 1, (
            f"Expected 1 cluster for similar fingerprints, got {len(unique_concepts)}"
        )


# ---------------------------------------------------------------------------
# Test: trust_weighted_update behaviour
# ---------------------------------------------------------------------------

class TestTrustWeightedUpdate:
    def test_trust_weighted_update_ignores_low_trust(self) -> None:
        """trust_scores=[0.0, 0.0] -> centroid unchanged."""
        bank = _make_trust_bank()
        # Seed with a single concept
        seed_fp = _make_fp(mean_shift=0.0, seed=0)
        result = bank.spawn_from_fingerprint(seed_fp)
        cid = result.new_concept_id

        slot = bank.get_slot(cid)
        assert slot is not None
        centroid_before = slot.semantic_anchor_set.mean.copy()

        # Two far-away fingerprints, but trust = 0
        fp1 = _make_fp(mean_shift=50.0, seed=10)
        fp2 = _make_fp(mean_shift=-50.0, seed=11)
        bank.trust_weighted_update(cid, [fp1, fp2], trust_scores=[0.0, 0.0])

        centroid_after = bank.get_slot(cid).semantic_anchor_set.mean
        np.testing.assert_allclose(
            centroid_after, centroid_before, atol=1e-10,
            err_msg="Centroid should not move when all trust scores are zero",
        )

    def test_trust_weighted_update_applies_high_trust(self) -> None:
        """trust_scores=[0.9, 0.85] -> centroid moves toward new fingerprints."""
        bank = _make_trust_bank()
        seed_fp = _make_fp(mean_shift=0.0, seed=0)
        result = bank.spawn_from_fingerprint(seed_fp)
        cid = result.new_concept_id

        centroid_before = bank.get_slot(cid).semantic_anchor_set.mean.copy()

        # Two fingerprints shifted to +5.0
        fp1 = _make_fp(mean_shift=5.0, seed=20)
        fp2 = _make_fp(mean_shift=5.0, seed=21)
        bank.trust_weighted_update(cid, [fp1, fp2], trust_scores=[0.9, 0.85])

        centroid_after = bank.get_slot(cid).semantic_anchor_set.mean
        # Centroid should move toward mean_shift=5.0
        shift = np.mean(centroid_after - centroid_before)
        assert shift > 0.1, (
            f"Centroid should move toward trusted fingerprints, "
            f"but mean shift was only {shift:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: windowed decay forgets old anchors
# ---------------------------------------------------------------------------

class TestWindowedDecay:
    def test_windowed_decay_forgets_old(self) -> None:
        """After W+1 anchors, oldest contributes < 15% to centroid."""
        W = 5
        gamma = 0.7
        bank = _make_trust_bank(trust_buffer_size=W, trust_decay=gamma)

        seed_fp = _make_fp(mean_shift=0.0, seed=0)
        result = bank.spawn_from_fingerprint(seed_fp)
        cid = result.new_concept_id

        # Feed W+1 trusted updates (each at a different location)
        for i in range(W + 1):
            fp = _make_fp(mean_shift=float(i) * 2.0, seed=50 + i)
            bank.trust_weighted_update(cid, [fp], trust_scores=[0.95])

        # Buffer should be trimmed to W
        slot = bank.get_slot(cid)
        assert len(slot.anchor_buffer) == W, (
            f"Buffer should be trimmed to W={W}, got {len(slot.anchor_buffer)}"
        )

        # Analytical check: weight for buffer index i is gamma^(W-1-i).
        # Index 0 is oldest -> weight = gamma^(W-1); index W-1 is newest -> weight = 1.
        weights = np.array([gamma ** (W - 1 - i) for i in range(W)])
        oldest_fraction = weights[0] / weights.sum()
        assert oldest_fraction < 0.15, (
            f"Oldest anchor fraction {oldest_fraction:.3f} should be < 0.15"
        )


# ---------------------------------------------------------------------------
# Test: provisional slot promotion
# ---------------------------------------------------------------------------

class TestProvisionalSlotPromotion:
    def test_provisional_slot_promotion(self) -> None:
        """Provisional slot promoted after W_min trusted updates."""
        W_min = 2
        bank = _make_trust_bank(trust_promotion_threshold=W_min)

        # Bootstrap with two concepts so slots are created as provisional
        fps = _make_separated_fps(n_per_concept=2)
        assignments = bank.batch_bootstrap(fps)
        cid = assignments[0]

        slot = bank.get_slot(cid)
        assert slot.is_provisional, "Bootstrapped slot should start provisional"

        # Feed W_min trusted updates -> should promote
        for i in range(W_min):
            fp = _make_fp(mean_shift=0.0, seed=200 + i)
            bank.trust_weighted_update(cid, [fp], trust_scores=[0.9])

        slot = bank.get_slot(cid)
        assert not slot.is_provisional, (
            f"Concept should be promoted after {W_min} trusted updates"
        )


# ---------------------------------------------------------------------------
# Test: dormancy freezes buffer
# ---------------------------------------------------------------------------

class TestDormancy:
    def test_dormancy_freezes_buffer(self) -> None:
        """When a slot receives no updates for several rounds, buffer is frozen."""
        bank = _make_trust_bank()
        seed_fp = _make_fp(mean_shift=0.0, seed=0)
        result = bank.spawn_from_fingerprint(seed_fp)
        cid = result.new_concept_id

        # Do one trusted update so buffer has content
        fp = _make_fp(mean_shift=1.0, seed=30)
        bank.trust_weighted_update(cid, [fp], trust_scores=[0.9])

        slot = bank.get_slot(cid)
        centroid_before = slot.semantic_anchor_set.mean.copy()
        buffer_len_before = len(slot.anchor_buffer)

        # Simulate several rounds with no updates (call step)
        for _ in range(5):
            bank.step()

        slot = bank.get_slot(cid)
        centroid_after = slot.semantic_anchor_set.mean
        buffer_len_after = len(slot.anchor_buffer)

        assert buffer_len_after == buffer_len_before, (
            "Buffer length should not change during dormancy"
        )
        np.testing.assert_allclose(
            centroid_after, centroid_before, atol=1e-10,
            err_msg="Centroid should not change during dormancy",
        )


# ---------------------------------------------------------------------------
# Test: backward compatibility (legacy mode)
# ---------------------------------------------------------------------------

class TestBackwardCompatLegacy:
    def test_backward_compat_legacy(self) -> None:
        """enable_trust_estimation=False -> absorb_fingerprint works as before."""
        bank = _make_legacy_bank()

        # Seed a concept via legacy spawn
        fp0 = _make_fp(mean_shift=0.0, seed=0)
        result = bank.spawn_from_fingerprint(fp0)
        cid = result.new_concept_id

        slot = bank.get_slot(cid)
        mean_before = slot.semantic_anchor_set.mean.copy()
        count_before = slot.semantic_anchor_set.count

        # Absorb a shifted fingerprint (legacy count-weighted running mean)
        fp1 = _make_fp(mean_shift=3.0, seed=1)
        bank.absorb_fingerprint(cid, fp1)

        slot = bank.get_slot(cid)
        mean_after = slot.semantic_anchor_set.mean
        count_after = slot.semantic_anchor_set.count

        # Count should increase
        assert count_after > count_before, "Legacy absorb should increment count"
        # Mean should shift toward fp1 (mean_shift=3.0)
        shift = np.mean(mean_after - mean_before)
        assert shift > 0.1, (
            f"Legacy absorb should move mean toward new data, shift={shift:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: singleton gets zero trust (no centroid update)
# ---------------------------------------------------------------------------

class TestSingletonTrust:
    def test_singleton_gets_zero_trust(self) -> None:
        """A concept group with 1 client: trust=0 means no centroid update."""
        bank = _make_trust_bank()

        seed_fp = _make_fp(mean_shift=0.0, seed=0)
        result = bank.spawn_from_fingerprint(seed_fp)
        cid = result.new_concept_id

        slot = bank.get_slot(cid)
        centroid_before = slot.semantic_anchor_set.mean.copy()

        # Singleton with trust=0 should not update centroid
        far_fp = _make_fp(mean_shift=100.0, seed=99)
        bank.trust_weighted_update(cid, [far_fp], trust_scores=[0.0])

        centroid_after = bank.get_slot(cid).semantic_anchor_set.mean
        np.testing.assert_allclose(
            centroid_after, centroid_before, atol=1e-10,
            err_msg="Singleton with trust=0 should not move centroid",
        )
