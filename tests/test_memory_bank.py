"""Tests for DynamicMemoryBank."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.posterior.memory_bank import (
    DynamicMemoryBank,
    MemoryBankConfig,
    SpawnResult,
    _merge_fingerprint_into,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp(n_features: int = 2, n_classes: int = 2, n_samples: int = 20,
             mean_shift: float = 0.0, seed: int = 0) -> ConceptFingerprint:
    """Create a fingerprint from synthetic data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) + mean_shift
    y = rng.randint(0, n_classes, size=n_samples)
    fp = ConceptFingerprint(n_features, n_classes)
    fp.update(X, y)
    return fp


# ---------------------------------------------------------------------------
# MemoryBankConfig
# ---------------------------------------------------------------------------

class TestMemoryBankConfig:
    def test_defaults(self) -> None:
        cfg = MemoryBankConfig()
        assert cfg.max_concepts == 20
        assert cfg.merge_threshold == 0.85
        assert cfg.merge_min_support == 1
        assert cfg.min_count == 5.0

    def test_invalid_max_concepts(self) -> None:
        with pytest.raises(ValueError, match="max_concepts"):
            MemoryBankConfig(max_concepts=0)

    def test_invalid_merge_threshold(self) -> None:
        with pytest.raises(ValueError, match="merge_threshold"):
            MemoryBankConfig(merge_threshold=0.0)

    def test_invalid_merge_min_support(self) -> None:
        with pytest.raises(ValueError, match="merge_min_support"):
            MemoryBankConfig(merge_min_support=0)

    def test_invalid_min_count(self) -> None:
        with pytest.raises(ValueError, match="min_count"):
            MemoryBankConfig(min_count=-1.0)


# ---------------------------------------------------------------------------
# SpawnResult
# ---------------------------------------------------------------------------

class TestSpawnResult:
    def test_creation(self) -> None:
        sr = SpawnResult(new_concept_id=3, absorbed=False)
        assert sr.new_concept_id == 3
        assert sr.absorbed is False


# ---------------------------------------------------------------------------
# _merge_fingerprint_into
# ---------------------------------------------------------------------------

class TestMergeFingerprint:
    def test_merge_empty_source(self) -> None:
        target = _make_fp(seed=0)
        source = ConceptFingerprint(2, 2)
        old_count = target.count
        _merge_fingerprint_into(target, source)
        assert target.count == old_count

    def test_merge_into_empty_target(self) -> None:
        target = ConceptFingerprint(2, 2)
        source = _make_fp(seed=0, n_samples=10)
        _merge_fingerprint_into(target, source)
        assert target.count == source.count
        np.testing.assert_allclose(target.mean, source.mean, atol=1e-10)

    def test_merge_two_populated(self) -> None:
        fp_a = _make_fp(seed=0, n_samples=30)
        fp_b = _make_fp(seed=1, n_samples=20)
        total = fp_a.count + fp_b.count
        _merge_fingerprint_into(fp_a, fp_b)
        assert abs(fp_a.count - total) < 1e-10

    def test_merge_preserves_label_counts(self) -> None:
        fp_a = _make_fp(seed=0, n_samples=10)
        fp_b = _make_fp(seed=1, n_samples=10)
        total_labels = fp_a._label_counts.copy() + fp_b._label_counts.copy()
        _merge_fingerprint_into(fp_a, fp_b)
        np.testing.assert_allclose(fp_a._label_counts, total_labels)


# ---------------------------------------------------------------------------
# DynamicMemoryBank - spawn
# ---------------------------------------------------------------------------

class TestSpawn:
    def test_spawn_creates_concept(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        fp = _make_fp(seed=0)
        result = bank.spawn_from_fingerprint(fp)
        assert result.absorbed is False
        assert bank.n_concepts == 1
        assert bank.get_fingerprint(result.new_concept_id) is not None

    def test_spawn_preserves_feature_groups(self) -> None:
        bank = DynamicMemoryBank(n_features=4, n_classes=2)
        fp = _make_fp(n_features=4, seed=0)
        fp = ConceptFingerprint(
            4,
            2,
            feature_groups=[(0, 2, 0.75), (2, 4, 0.25)],
        )
        rng = np.random.RandomState(0)
        X = rng.randn(20, 4)
        y = rng.randint(0, 2, size=20)
        fp.update(X, y)
        result = bank.spawn_from_fingerprint(fp)
        stored = bank.get_fingerprint(result.new_concept_id)
        assert stored is not None
        assert stored.feature_groups == fp.feature_groups

    def test_spawn_increments_id(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        r1 = bank.spawn_from_fingerprint(_make_fp(seed=0))
        r2 = bank.spawn_from_fingerprint(_make_fp(seed=1, mean_shift=5.0))
        assert r2.new_concept_id == r1.new_concept_id + 1

    def test_spawn_absorbs_at_capacity(self) -> None:
        cfg = MemoryBankConfig(max_concepts=2)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.0))
        bank.spawn_from_fingerprint(_make_fp(seed=1, mean_shift=10.0))
        assert bank.n_concepts == 2

        result = bank.spawn_from_fingerprint(_make_fp(seed=2, mean_shift=0.1))
        assert result.absorbed is True
        assert bank.n_concepts == 2

    def test_spawn_multiple(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        for i in range(5):
            bank.spawn_from_fingerprint(_make_fp(seed=i, mean_shift=i * 10.0))
        assert bank.n_concepts == 5


# ---------------------------------------------------------------------------
# DynamicMemoryBank - absorb
# ---------------------------------------------------------------------------

class TestAbsorb:
    def test_absorb_updates_count(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        r = bank.spawn_from_fingerprint(_make_fp(seed=0, n_samples=10))
        old_count = bank.get_fingerprint(r.new_concept_id).count
        bank.absorb_fingerprint(r.new_concept_id, _make_fp(seed=1, n_samples=5))
        new_count = bank.get_fingerprint(r.new_concept_id).count
        assert new_count > old_count

    def test_absorb_unknown_concept_raises(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        with pytest.raises(KeyError):
            bank.absorb_fingerprint(999, _make_fp(seed=0))

    def test_absorb_signature_tracks_running_mean(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        r = bank.spawn_from_fingerprint(_make_fp(seed=0))
        bank.absorb_signature(r.new_concept_id, np.array([1.0, 0.0], dtype=np.float64))
        bank.absorb_signature(r.new_concept_id, np.array([0.0, 1.0], dtype=np.float64))
        signature = bank.get_signature(r.new_concept_id)
        assert signature is not None
        np.testing.assert_allclose(signature, np.array([0.5, 0.5]), atol=1e-10)


# ---------------------------------------------------------------------------
# DynamicMemoryBank - merge
# ---------------------------------------------------------------------------

class TestMerge:
    def test_merge_similar_concepts(self) -> None:
        cfg = MemoryBankConfig(merge_threshold=0.5)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        # Two very similar fingerprints
        bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.0, n_samples=50))
        bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.01, n_samples=50))
        assert bank.n_concepts == 2

        merged = bank.maybe_merge()
        assert len(merged) >= 1
        assert bank.n_concepts == 1

    def test_no_merge_dissimilar(self) -> None:
        cfg = MemoryBankConfig(merge_threshold=0.99)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.0, n_samples=50))
        bank.spawn_from_fingerprint(_make_fp(seed=1, mean_shift=100.0, n_samples=50))

        merged = bank.maybe_merge()
        assert len(merged) == 0
        assert bank.n_concepts == 2

    def test_merge_empty_bank(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        merged = bank.maybe_merge()
        assert merged == []

    def test_merge_combines_signatures(self) -> None:
        cfg = MemoryBankConfig(merge_threshold=0.5)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        r1 = bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.0, n_samples=50))
        r2 = bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.01, n_samples=50))
        bank.absorb_signature(r1.new_concept_id, np.array([1.0, 0.0], dtype=np.float64))
        bank.absorb_signature(r2.new_concept_id, np.array([0.0, 1.0], dtype=np.float64))

        merged = bank.maybe_merge()
        assert len(merged) >= 1
        kept_id = merged[0][0]
        signature = bank.get_signature(kept_id)
        assert signature is not None
        np.testing.assert_allclose(signature, np.array([0.5, 0.5]), atol=1e-10)

    def test_merge_requires_min_support(self) -> None:
        cfg = MemoryBankConfig(merge_threshold=0.5, merge_min_support=2)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        r0 = bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.0, n_samples=50))
        r1 = bank.spawn_from_fingerprint(_make_fp(seed=0, mean_shift=0.01, n_samples=50))

        merged = bank.maybe_merge()
        assert merged == []
        assert bank.n_concepts == 2

        bank.absorb_fingerprint(r0.new_concept_id, _make_fp(seed=10, mean_shift=0.0, n_samples=20))
        bank.absorb_fingerprint(r1.new_concept_id, _make_fp(seed=11, mean_shift=0.01, n_samples=20))

        merged = bank.maybe_merge()
        assert len(merged) >= 1
        assert bank.n_concepts == 1


# ---------------------------------------------------------------------------
# DynamicMemoryBank - shrink
# ---------------------------------------------------------------------------

class TestShrink:
    def test_shrink_removes_low_count(self) -> None:
        cfg = MemoryBankConfig(min_count=15.0)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        # One with high count, one with low
        bank.spawn_from_fingerprint(_make_fp(seed=0, n_samples=50))
        bank.spawn_from_fingerprint(_make_fp(seed=1, n_samples=3))
        assert bank.n_concepts == 2

        removed = bank.maybe_shrink()
        assert len(removed) == 1
        assert bank.n_concepts == 1

    def test_shrink_preserves_last_concept(self) -> None:
        cfg = MemoryBankConfig(min_count=1000.0)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        bank.spawn_from_fingerprint(_make_fp(seed=0, n_samples=5))
        assert bank.n_concepts == 1

        removed = bank.maybe_shrink()
        assert len(removed) == 0
        assert bank.n_concepts == 1

    def test_shrink_empty_bank(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        removed = bank.maybe_shrink()
        assert removed == []

    def test_shrink_keeps_at_least_one(self) -> None:
        cfg = MemoryBankConfig(min_count=1000.0)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        bank.spawn_from_fingerprint(_make_fp(seed=0, n_samples=5))
        bank.spawn_from_fingerprint(_make_fp(seed=1, n_samples=3))
        removed = bank.maybe_shrink()
        assert bank.n_concepts >= 1


# ---------------------------------------------------------------------------
# DynamicMemoryBank - step
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_triggers_maintenance(self) -> None:
        cfg = MemoryBankConfig(merge_every=2, shrink_every=3, min_count=1000.0)
        bank = DynamicMemoryBank(config=cfg, n_features=2, n_classes=2)
        bank.spawn_from_fingerprint(_make_fp(seed=0, n_samples=5))
        bank.spawn_from_fingerprint(_make_fp(seed=1, n_samples=5, mean_shift=100.0))
        initial = bank.n_concepts

        # Step 1 and 2 — merge triggers at step 2
        bank.step()
        bank.step()
        # Should not have merged dissimilar concepts
        assert bank.n_concepts == initial


# ---------------------------------------------------------------------------
# DynamicMemoryBank - concept_library
# ---------------------------------------------------------------------------

class TestConceptLibrary:
    def test_returns_copy(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        bank.spawn_from_fingerprint(_make_fp(seed=0))
        lib = bank.concept_library
        lib[999] = _make_fp(seed=1)
        assert 999 not in bank._library

    def test_get_fingerprint_missing(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        assert bank.get_fingerprint(42) is None

    def test_slot_schema_tracks_support_and_key(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        result = bank.spawn_from_fingerprint(_make_fp(seed=0))
        slot = bank.get_slot(result.new_concept_id)
        assert slot is not None
        assert slot.support_count == 1
        assert slot.center_key is not None
        assert slot.semantic_anchor_set is not None

    def test_routing_library_uses_slot_keys(self) -> None:
        bank = DynamicMemoryBank(n_features=2, n_classes=2)
        r = bank.spawn_from_fingerprint(_make_fp(seed=0))
        routing = bank.routing_library
        assert r.new_concept_id in routing
        assert hasattr(routing[r.new_concept_id], "similarity")
