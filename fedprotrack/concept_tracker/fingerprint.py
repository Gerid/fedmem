"""Concept fingerprinting via lightweight distributional signatures.

A fingerprint summarizes a data distribution using incremental statistics
(mean, covariance, label distribution, class-conditional means) that can
be cheaply compared to identify recurring concepts across clients and
time steps.
"""

from __future__ import annotations

import numpy as np


def _normalize_feature_groups(
    n_features: int,
    feature_groups: list[tuple[int, int, float]] | tuple[tuple[int, int, float], ...] | None,
) -> tuple[tuple[int, int, float], ...] | None:
    if not feature_groups:
        return None

    normalized: list[tuple[int, int, float]] = []
    total_weight = 0.0
    for start, end, weight in feature_groups:
        start_i = int(start)
        end_i = int(end)
        weight_f = float(weight)
        if not 0 <= start_i < end_i <= n_features:
            raise ValueError(
                f"feature group span must satisfy 0 <= start < end <= {n_features}, "
                f"got {(start, end)}"
            )
        if weight_f < 0.0:
            raise ValueError(f"feature group weights must be non-negative, got {weight}")
        if weight_f == 0.0:
            continue
        normalized.append((start_i, end_i, weight_f))
        total_weight += weight_f

    if not normalized or total_weight <= 0.0:
        raise ValueError("feature_groups must contain at least one positive-weight span")

    return tuple(
        (start_i, end_i, weight_f / total_weight)
        for start_i, end_i, weight_f in normalized
    )


def _group_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    pooled_var: np.ndarray,
    feature_groups: tuple[tuple[int, int, float], ...] | None,
) -> float:
    if not feature_groups:
        diff = vec_a - vec_b
        sq_dist = np.sum(diff ** 2 / pooled_var)
        return float(np.exp(-0.5 * sq_dist))

    total = 0.0
    accum = 0.0
    for start, end, weight in feature_groups:
        diff = vec_a[start:end] - vec_b[start:end]
        local_var = pooled_var[start:end]
        sq_dist = float(np.sum(diff ** 2 / local_var))
        accum += weight * float(np.exp(-0.5 * sq_dist))
        total += weight
    if total <= 0.0:
        return 0.0
    return float(accum / total)


class ConceptFingerprint:
    """Incremental distributional fingerprint for a data concept.

    Maintains running mean, covariance, label distribution, and
    class-conditional means using Welford's online algorithm. Two
    fingerprints can be compared via a composite similarity score
    combining feature-space, label-space, and class-conditional
    distances.

    The class-conditional component is critical for distinguishing
    concepts that share the same marginal P(X) and P(Y) but differ
    in their decision boundary P(Y|X), such as SINE-based synthetic
    drift scenarios.

    Parameters
    ----------
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    decay : float
        Exponential decay factor in (0, 1]. Values < 1 give more weight
        to recent observations. Default 1.0 (no decay, equal weighting).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        decay: float = 1.0,
        feature_groups: list[tuple[int, int, float]] | tuple[tuple[int, int, float], ...] | None = None,
        sim_weights: tuple[float, float, float] | None = None,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.decay = decay
        self._feature_groups = _normalize_feature_groups(n_features, feature_groups)
        # Similarity component weights: (feat, label, class-conditional)
        self._sim_weights = sim_weights or (0.25, 0.30, 0.45)

        # Running statistics (Welford's online algorithm with optional decay)
        self._count: float = 0.0
        self._mean = np.zeros(n_features, dtype=np.float64)
        self._M2 = np.zeros((n_features, n_features), dtype=np.float64)
        self._label_counts = np.zeros(n_classes, dtype=np.float64)

        # Class-conditional running means: E[X | Y = c]
        self._class_counts = np.zeros(n_classes, dtype=np.float64)
        self._class_means = np.zeros((n_classes, n_features), dtype=np.float64)

    @property
    def count(self) -> float:
        return self._count

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Estimated covariance matrix."""
        if self._count < 2:
            return np.eye(self.n_features, dtype=np.float64)
        return self._M2 / (self._count - 1)

    @property
    def label_distribution(self) -> np.ndarray:
        """Normalized label frequency vector."""
        total = self._label_counts.sum()
        if total == 0:
            return np.ones(self.n_classes, dtype=np.float64) / self.n_classes
        return self._label_counts / total

    @property
    def class_means(self) -> np.ndarray:
        """Per-class conditional means E[X | Y = c], shape (n_classes, n_features)."""
        return self._class_means.copy()

    @property
    def feature_groups(self) -> tuple[tuple[int, int, float], ...] | None:
        return self._feature_groups

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incorporate a batch of observations.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
            Feature matrix.
        y : np.ndarray of shape (n,)
            Class labels (integers in [0, n_classes)).
        """
        for i in range(len(X)):
            self._update_one(X[i], int(y[i]))

    def _update_one(self, x: np.ndarray, label: int) -> None:
        """Update statistics with a single observation."""
        self._count = self._count * self.decay + 1.0
        self._label_counts *= self.decay
        if 0 <= label < self.n_classes:
            self._label_counts[label] += 1.0

            # Update per-class feature mean (online mean update)
            self._class_counts *= self.decay
            self._class_counts[label] += 1.0
            n_c = self._class_counts[label]
            if self.decay < 1.0:
                self._class_means[label] *= self.decay
                self._class_means[label] += (x - self._class_means[label] * self.decay) / n_c
            else:
                delta_c = x - self._class_means[label]
                self._class_means[label] += delta_c / n_c

        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._M2 = self._M2 * self.decay + np.outer(delta, delta2)

    def similarity(self, other: ConceptFingerprint) -> float:
        """Compute composite similarity to another fingerprint.

        Combines three components:
        - Feature-space distance (global mean, Mahalanobis-inspired): 0.25
        - Label-space distance (Hellinger on label distribution): 0.30
        - Class-conditional distance (per-class mean difference): 0.45

        The class-conditional component has the highest weight because it
        captures P(Y|X) differences — critical for scenarios like SINE
        where concepts share P(X) and P(Y) but differ in decision boundaries.

        Parameters
        ----------
        other : ConceptFingerprint
            The fingerprint to compare against.

        Returns
        -------
        score : float
            Similarity score in [0, 1].
        """
        if self._count < 2 or other._count < 2:
            return 0.0

        feat_sim = self._feature_similarity(other)
        label_sim = self._label_similarity(other)
        cc_sim = self._class_conditional_similarity(other)
        w_f, w_l, w_c = self._sim_weights
        return w_f * feat_sim + w_l * label_sim + w_c * cc_sim

    def _compatible_feature_groups(
        self,
        other: ConceptFingerprint,
    ) -> tuple[tuple[int, int, float], ...] | None:
        if self._feature_groups is None:
            return None

        other_groups = getattr(other, "feature_groups", None)
        if other_groups is None:
            return self._feature_groups
        if len(self._feature_groups) != len(other_groups):
            return None
        for (start_a, end_a, _), (start_b, end_b, _) in zip(
            self._feature_groups,
            other_groups,
        ):
            if start_a != start_b or end_a != end_b:
                return None
        return self._feature_groups

    def _feature_similarity(self, other: ConceptFingerprint) -> float:
        """Feature-space similarity via squared Euclidean on means,
        scaled by pooled variance."""
        pooled_var = (np.diag(self.covariance) + np.diag(other.covariance)) / 2.0
        pooled_var = np.maximum(pooled_var, 1e-8)
        return _group_similarity(
            self._mean,
            other._mean,
            pooled_var,
            self._compatible_feature_groups(other),
        )

    def _label_similarity(self, other: ConceptFingerprint) -> float:
        """Label distribution similarity via Hellinger distance."""
        p = self.label_distribution
        q = other.label_distribution
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
        return float(1.0 - hellinger)

    def _class_conditional_similarity(self, other: ConceptFingerprint) -> float:
        """Class-conditional mean similarity.

        For each class with sufficient samples in both fingerprints,
        computes the Euclidean distance between class-conditional means
        scaled by pooled variance. Returns the average similarity across
        classes.

        Returns
        -------
        float
            Similarity in [0, 1]. Returns 0.0 if no class has enough
            samples in both fingerprints.
        """
        min_class_count = 3.0
        pooled_var = (np.diag(self.covariance) + np.diag(other.covariance)) / 2.0
        pooled_var = np.maximum(pooled_var, 1e-8)
        feature_groups = self._compatible_feature_groups(other)

        sims: list[float] = []
        for c in range(self.n_classes):
            if (self._class_counts[c] < min_class_count
                    or other._class_counts[c] < min_class_count):
                continue
            sims.append(
                _group_similarity(
                    self._class_means[c],
                    other._class_means[c],
                    pooled_var,
                    feature_groups,
                )
            )

        if not sims:
            return 0.0
        return float(np.mean(sims))

    def to_vector(self) -> np.ndarray:
        """Flatten fingerprint to a fixed-size vector for external use."""
        return np.concatenate([
            self._mean,
            self.label_distribution,
            self._class_means.ravel(),
        ])

    def __repr__(self) -> str:
        return (
            f"ConceptFingerprint(n_features={self.n_features}, "
            f"n_classes={self.n_classes}, count={self._count:.0f})"
        )
