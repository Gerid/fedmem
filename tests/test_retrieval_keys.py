from __future__ import annotations

import numpy as np

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.posterior.retrieval_keys import RetrievalKeyConfig, build_retrieval_key


def _make_fp(labels: list[int], *, seed: int = 0, n_classes: int = 20) -> ConceptFingerprint:
    rng = np.random.RandomState(seed)
    X = rng.randn(len(labels), 4).astype(np.float64)
    y = np.array(labels, dtype=np.int64)
    fp = ConceptFingerprint(4, n_classes)
    fp.update(X, y)
    return fp


class TestRetrievalKeys:
    def test_semantic_only_key_matches_identical_label_distributions(self) -> None:
        fp_a = _make_fp([0, 0, 1, 1, 2, 2], seed=0)
        fp_b = _make_fp([0, 0, 1, 1, 2, 2], seed=1)
        key_cfg = RetrievalKeyConfig(
            mode="multi_scale",
            style_weight=0.0,
            semantic_weight=1.0,
            prototype_weight=0.0,
        )
        sim = build_retrieval_key(fp_a, config=key_cfg).similarity(
            build_retrieval_key(fp_b, config=key_cfg)
        )
        assert sim > 0.99

    def test_semantic_only_key_penalizes_disjoint_label_support(self) -> None:
        fp_a = _make_fp([0, 0, 1, 1, 2, 2], seed=0)
        fp_b = _make_fp([10, 10, 11, 11, 12, 12], seed=1)
        key_cfg = RetrievalKeyConfig(
            mode="multi_scale",
            style_weight=0.0,
            semantic_weight=1.0,
            prototype_weight=0.0,
        )
        sim = build_retrieval_key(fp_a, config=key_cfg).similarity(
            build_retrieval_key(fp_b, config=key_cfg)
        )
        assert sim < 0.1
