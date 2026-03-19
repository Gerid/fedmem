from __future__ import annotations

"""Retrieval-key abstractions for Plan C-compatible concept addressing."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..concept_tracker.fingerprint import ConceptFingerprint


class SupportsSimilarity(Protocol):
    """Minimal protocol for routing keys used by Gibbs posterior inference."""

    count: float

    def similarity(self, other: object) -> float:
        """Return a similarity score in ``[0, 1]``."""

    def to_vector(self) -> np.ndarray:
        """Flatten the key into a vector representation."""


@dataclass
class RetrievalKeyConfig:
    """Configuration for retrieval-key construction."""

    mode: str = "fingerprint"
    ema_decay: float = 0.0
    style_weight: float = 0.25
    semantic_weight: float = 0.30
    prototype_weight: float = 0.45

    def __post_init__(self) -> None:
        if self.mode not in {"fingerprint", "ema", "multi_scale"}:
            raise ValueError(
                f"mode must be one of fingerprint/ema/multi_scale, got {self.mode}"
            )
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be in [0, 1), got {self.ema_decay}"
            )
        weights = np.array(
            [self.style_weight, self.semantic_weight, self.prototype_weight],
            dtype=np.float64,
        )
        if np.any(weights < 0.0):
            raise ValueError("retrieval-key weights must be non-negative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("at least one retrieval-key weight must be positive")

    @property
    def normalized_weights(self) -> tuple[float, float, float]:
        weights = np.array(
            [self.style_weight, self.semantic_weight, self.prototype_weight],
            dtype=np.float64,
        )
        weights /= float(weights.sum())
        return float(weights[0]), float(weights[1]), float(weights[2])


@dataclass
class CompositeRetrievalKey:
    """Multi-component retrieval key derived from a concept fingerprint."""

    style_vec: np.ndarray
    semantic_vec: np.ndarray
    prototype_vec: np.ndarray
    n_features: int
    n_classes: int
    count: float
    style_weight: float = 0.25
    semantic_weight: float = 0.30
    prototype_weight: float = 0.45
    feature_groups: tuple[tuple[int, int, float], ...] | None = None

    @classmethod
    def from_fingerprint(
        cls,
        fingerprint: ConceptFingerprint,
        config: RetrievalKeyConfig | None = None,
        previous: CompositeRetrievalKey | None = None,
    ) -> CompositeRetrievalKey:
        """Build a retrieval key from a fingerprint and optional previous key."""
        cfg = config or RetrievalKeyConfig()
        style_vec, semantic_vec, prototype_vec = _fingerprint_components(fingerprint)

        if previous is not None and cfg.mode in {"ema", "multi_scale"}:
            decay = cfg.ema_decay
            style_vec = decay * previous.style_vec + (1.0 - decay) * style_vec
            semantic_vec = decay * previous.semantic_vec + (1.0 - decay) * semantic_vec
            prototype_vec = decay * previous.prototype_vec + (1.0 - decay) * prototype_vec

        style_weight, semantic_weight, prototype_weight = cfg.normalized_weights
        return cls(
            style_vec=style_vec.astype(np.float64),
            semantic_vec=semantic_vec.astype(np.float64),
            prototype_vec=prototype_vec.astype(np.float64),
            n_features=fingerprint.n_features,
            n_classes=fingerprint.n_classes,
            count=float(fingerprint.count),
            style_weight=style_weight,
            semantic_weight=semantic_weight,
            prototype_weight=prototype_weight,
            feature_groups=fingerprint.feature_groups,
        )

    def similarity(self, other: object) -> float:
        other_style, other_semantic, other_proto = _components_from_any(other)
        feature_groups = _compatible_feature_groups(
            self.feature_groups,
            _feature_groups_from_any(other),
        )
        style_sim = _vector_similarity(
            self.style_vec,
            other_style,
            feature_groups=feature_groups,
        )
        semantic_sim = _distribution_similarity(self.semantic_vec, other_semantic)
        proto_sim = _vector_similarity(
            self.prototype_vec,
            other_proto,
            feature_groups=_expand_feature_groups(
                feature_groups,
                self.n_features,
                self.n_classes,
            ),
        )
        return float(
            self.style_weight * style_sim
            + self.semantic_weight * semantic_sim
            + self.prototype_weight * proto_sim
        )

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [self.style_vec, self.semantic_vec, self.prototype_vec]
        )


def build_retrieval_key(
    fingerprint: ConceptFingerprint,
    config: RetrievalKeyConfig | None = None,
    previous: CompositeRetrievalKey | None = None,
) -> CompositeRetrievalKey:
    """Construct a retrieval key from a concept fingerprint."""
    return CompositeRetrievalKey.from_fingerprint(fingerprint, config=config, previous=previous)


def _fingerprint_components(
    fingerprint: ConceptFingerprint,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    style_vec = fingerprint.mean.astype(np.float64)
    semantic_vec = fingerprint.label_distribution.astype(np.float64)
    prototype_vec = fingerprint.class_means.reshape(-1).astype(np.float64)
    return style_vec, semantic_vec, prototype_vec


def _components_from_any(obj: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(obj, CompositeRetrievalKey):
        return obj.style_vec, obj.semantic_vec, obj.prototype_vec
    if isinstance(obj, ConceptFingerprint):
        return _fingerprint_components(obj)
    if hasattr(obj, "to_vector"):
        vec = np.asarray(getattr(obj, "to_vector")(), dtype=np.float64)
        return vec, np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)
    raise TypeError(f"Unsupported retrieval-key object: {type(obj)!r}")


def _feature_groups_from_any(
    obj: object,
) -> tuple[tuple[int, int, float], ...] | None:
    if isinstance(obj, CompositeRetrievalKey):
        return obj.feature_groups
    if isinstance(obj, ConceptFingerprint):
        return obj.feature_groups
    return None


def _compatible_feature_groups(
    feature_groups: tuple[tuple[int, int, float], ...] | None,
    other_groups: tuple[tuple[int, int, float], ...] | None,
) -> tuple[tuple[int, int, float], ...] | None:
    if feature_groups is None:
        return None
    if other_groups is None:
        return feature_groups
    if len(feature_groups) != len(other_groups):
        return None
    for (start_a, end_a, _), (start_b, end_b, _) in zip(feature_groups, other_groups):
        if start_a != start_b or end_a != end_b:
            return None
    return feature_groups


def _expand_feature_groups(
    feature_groups: tuple[tuple[int, int, float], ...] | None,
    n_features: int,
    n_classes: int,
) -> tuple[tuple[int, int, float], ...] | None:
    if feature_groups is None:
        return None
    expanded: list[tuple[int, int, float]] = []
    for class_idx in range(n_classes):
        offset = class_idx * n_features
        for start, end, weight in feature_groups:
            expanded.append((offset + start, offset + end, weight))
    return tuple(expanded)


def _vector_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    *,
    feature_groups: tuple[tuple[int, int, float], ...] | None = None,
) -> float:
    vec_a = np.asarray(vec_a, dtype=np.float64).reshape(-1)
    vec_b = np.asarray(vec_b, dtype=np.float64).reshape(-1)
    if vec_a.shape != vec_b.shape:
        n = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:n]
        vec_b = vec_b[:n]
    if vec_a.size == 0:
        return 1.0

    if feature_groups is not None:
        total = 0.0
        accum = 0.0
        for start, end, weight in feature_groups:
            if start >= vec_a.size:
                continue
            end = min(end, vec_a.size)
            if end <= start:
                continue
            diff = vec_a[start:end] - vec_b[start:end]
            scale = np.maximum(
                np.mean(np.abs(vec_a[start:end])) + np.mean(np.abs(vec_b[start:end])),
                1e-8,
            )
            sq_dist = float(np.mean(diff ** 2) / scale)
            accum += weight * float(np.exp(-0.5 * sq_dist))
            total += weight
        if total > 0.0:
            return float(accum / total)

    diff = vec_a - vec_b
    scale = np.maximum(np.mean(np.abs(vec_a)) + np.mean(np.abs(vec_b)), 1e-8)
    sq_dist = float(np.mean(diff ** 2) / scale)
    return float(np.exp(-0.5 * sq_dist))


def _distribution_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Probability-distribution similarity via Hellinger distance."""
    vec_a = np.asarray(vec_a, dtype=np.float64).reshape(-1)
    vec_b = np.asarray(vec_b, dtype=np.float64).reshape(-1)
    if vec_a.shape != vec_b.shape:
        n = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:n]
        vec_b = vec_b[:n]
    if vec_a.size == 0:
        return 1.0

    p = np.clip(vec_a, 0.0, None)
    q = np.clip(vec_b, 0.0, None)
    p_sum = float(p.sum())
    q_sum = float(q.sum())
    if p_sum <= 0.0 and q_sum <= 0.0:
        return 1.0
    if p_sum <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / p_sum
    if q_sum <= 0.0:
        q = np.ones_like(q) / float(len(q))
    else:
        q = q / q_sum

    hellinger = float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    return float(1.0 - hellinger)
