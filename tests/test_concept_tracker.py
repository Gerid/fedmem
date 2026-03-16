"""Tests for concept tracking module."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.concept_tracker import ConceptFingerprint, ConceptTracker, TrackingResult


class TestConceptFingerprint:
    def test_init(self):
        fp = ConceptFingerprint(n_features=3, n_classes=2)
        assert fp.n_features == 3
        assert fp.n_classes == 2
        assert fp.count == 0

    def test_update_batch(self):
        fp = ConceptFingerprint(n_features=2, n_classes=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        fp.update(X, y)
        assert fp.count == 3
        np.testing.assert_allclose(fp.mean, [3.0, 4.0], atol=0.1)

    def test_label_distribution(self):
        fp = ConceptFingerprint(n_features=2, n_classes=2)
        X = np.array([[0.0, 0.0]] * 6)
        y = np.array([0, 0, 0, 0, 1, 1])
        fp.update(X, y)
        dist = fp.label_distribution
        assert dist[0] > dist[1]
        np.testing.assert_allclose(dist.sum(), 1.0)

    def test_similarity_identical(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = (X[:, 0] > 0).astype(np.int32)

        fp1 = ConceptFingerprint(n_features=3, n_classes=2)
        fp2 = ConceptFingerprint(n_features=3, n_classes=2)
        fp1.update(X, y)
        fp2.update(X, y)

        sim = fp1.similarity(fp2)
        assert sim > 0.9

    def test_similarity_different(self):
        rng = np.random.default_rng(42)
        X1 = rng.standard_normal((100, 2))
        y1 = np.zeros(100, dtype=np.int32)

        X2 = rng.standard_normal((100, 2)) + 10.0
        y2 = np.ones(100, dtype=np.int32)

        fp1 = ConceptFingerprint(n_features=2, n_classes=2)
        fp2 = ConceptFingerprint(n_features=2, n_classes=2)
        fp1.update(X1, y1)
        fp2.update(X2, y2)

        sim = fp1.similarity(fp2)
        assert sim < 0.5

    def test_to_vector(self):
        fp = ConceptFingerprint(n_features=3, n_classes=2)
        X = np.ones((10, 3))
        y = np.zeros(10, dtype=np.int32)
        fp.update(X, y)
        vec = fp.to_vector()
        # 3 features (mean) + 2 classes (label_dist) + 2*3 (class_means)
        assert vec.shape == (11,)


class TestConceptTracker:
    def test_start(self):
        tracker = ConceptTracker(n_features=2, n_classes=2)
        X = np.random.default_rng(0).standard_normal((50, 2))
        y = np.zeros(50, dtype=np.int32)
        cid = tracker.start(X, y)
        assert cid == 0
        assert tracker.active_concept_id == 0
        assert tracker.n_known_concepts == 1

    def test_novel_concept_detection(self):
        tracker = ConceptTracker(
            n_features=2, n_classes=2, similarity_threshold=0.7
        )
        rng = np.random.default_rng(42)

        # Concept 0
        X0 = rng.standard_normal((100, 2))
        y0 = np.zeros(100, dtype=np.int32)
        tracker.start(X0, y0)

        # Very different concept → should be novel
        X1 = rng.standard_normal((100, 2)) + 20.0
        y1 = np.ones(100, dtype=np.int32)
        result = tracker.on_drift_detected(X1, y1)
        assert result.is_novel
        assert result.predicted_concept_id == 1
        assert tracker.n_known_concepts == 2

    def test_recurrence_detection(self):
        tracker = ConceptTracker(
            n_features=2, n_classes=2, similarity_threshold=0.5
        )
        rng = np.random.default_rng(42)

        # Concept 0
        X0 = rng.standard_normal((200, 2))
        y0 = (X0[:, 0] > 0).astype(np.int32)
        tracker.start(X0, y0)

        # Concept 1 (different)
        X1 = rng.standard_normal((200, 2)) + 15.0
        y1 = np.ones(200, dtype=np.int32)
        tracker.on_drift_detected(X1, y1)

        # Concept 0 again (recurrence)
        X0b = rng.standard_normal((200, 2))
        y0b = (X0b[:, 0] > 0).astype(np.int32)
        result = tracker.on_drift_detected(X0b, y0b)
        assert not result.is_novel
        assert result.predicted_concept_id == 0

    def test_observe(self):
        tracker = ConceptTracker(n_features=2, n_classes=2)
        X = np.ones((10, 2))
        y = np.zeros(10, dtype=np.int32)
        tracker.start(X, y)
        tracker.observe(X * 2, y)
        assert tracker.n_known_concepts == 1

    def test_get_all_similarities(self):
        tracker = ConceptTracker(n_features=2, n_classes=2)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        y = np.zeros(50, dtype=np.int32)
        tracker.start(X, y)
        sims = tracker.get_all_similarities(X, y)
        assert 0 in sims
        assert sims[0] > 0.5
