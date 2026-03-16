"""Concept identity tracker — the core FedProTrack algorithm.

When a drift detector signals a change, the tracker compares the incoming
data's fingerprint against a library of known concept fingerprints to decide
whether this is a *recurrence* of a previously seen concept or a *novel* one.

If the concept is recognized, the client can reuse the previously trained model
(warm start), avoiding catastrophic forgetting and cold-start degradation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fingerprint import ConceptFingerprint


@dataclass
class TrackingResult:
    """Outcome of a concept identification attempt."""

    predicted_concept_id: int
    is_novel: bool
    similarity_score: float
    best_match_id: int | None = None


class ConceptTracker:
    """Manages a library of concept fingerprints and performs identification.

    Parameters
    ----------
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    similarity_threshold : float
        Minimum similarity score to consider a match as a recurrence.
        Fingerprints below this threshold are classified as novel concepts.
    decay : float
        Exponential decay factor for fingerprints (passed to ConceptFingerprint).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        similarity_threshold: float = 0.7,
        decay: float = 1.0,
    ):
        self.n_features = n_features
        self.n_classes = n_classes
        self.similarity_threshold = similarity_threshold
        self.decay = decay

        self._library: dict[int, ConceptFingerprint] = {}
        self._next_concept_id: int = 0
        self._active_concept_id: int | None = None
        self._active_fingerprint: ConceptFingerprint | None = None

    @property
    def active_concept_id(self) -> int | None:
        return self._active_concept_id

    @property
    def n_known_concepts(self) -> int:
        return len(self._library)

    def start(self, X: np.ndarray, y: np.ndarray) -> int:
        """Initialize the tracker with the first batch of data.

        Creates concept 0 and sets it as active.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
        y : np.ndarray of shape (n,)

        Returns
        -------
        concept_id : int
            Always 0 for the first call.
        """
        fp = self._make_fingerprint()
        fp.update(X, y)
        concept_id = self._register_concept(fp)
        self._active_concept_id = concept_id
        self._active_fingerprint = fp
        return concept_id

    def observe(self, X: np.ndarray, y: np.ndarray) -> None:
        """Feed data to the active concept's fingerprint (no drift).

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
        y : np.ndarray of shape (n,)
        """
        if self._active_fingerprint is not None:
            self._active_fingerprint.update(X, y)

    def on_drift_detected(self, X: np.ndarray, y: np.ndarray) -> TrackingResult:
        """Handle a detected drift: identify the new concept.

        Compares the incoming batch against all known concept fingerprints.
        If the best match exceeds the similarity threshold, this is a
        recurrence; otherwise a novel concept is created.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
        y : np.ndarray of shape (n,)

        Returns
        -------
        TrackingResult
        """
        # Build fingerprint from post-drift data
        new_fp = self._make_fingerprint()
        new_fp.update(X, y)

        # Compare against library
        best_id = None
        best_sim = -1.0
        for cid, lib_fp in self._library.items():
            sim = new_fp.similarity(lib_fp)
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        if best_sim >= self.similarity_threshold and best_id is not None:
            # Recurrence: reuse existing concept
            self._active_concept_id = best_id
            self._active_fingerprint = self._library[best_id]
            self._active_fingerprint.update(X, y)
            return TrackingResult(
                predicted_concept_id=best_id,
                is_novel=False,
                similarity_score=best_sim,
                best_match_id=best_id,
            )
        else:
            # Novel concept
            concept_id = self._register_concept(new_fp)
            self._active_concept_id = concept_id
            self._active_fingerprint = new_fp
            return TrackingResult(
                predicted_concept_id=concept_id,
                is_novel=True,
                similarity_score=best_sim,
                best_match_id=best_id,
            )

    def get_fingerprint(self, concept_id: int) -> ConceptFingerprint | None:
        """Retrieve a stored fingerprint by concept ID."""
        return self._library.get(concept_id)

    def get_all_similarities(self, X: np.ndarray, y: np.ndarray) -> dict[int, float]:
        """Compute similarity of a batch against all known concepts.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
        y : np.ndarray of shape (n,)

        Returns
        -------
        similarities : dict[int, float]
            Mapping from concept_id to similarity score.
        """
        probe = self._make_fingerprint()
        probe.update(X, y)
        return {cid: probe.similarity(fp) for cid, fp in self._library.items()}

    def _make_fingerprint(self) -> ConceptFingerprint:
        return ConceptFingerprint(
            n_features=self.n_features,
            n_classes=self.n_classes,
            decay=self.decay,
        )

    def _register_concept(self, fp: ConceptFingerprint) -> int:
        concept_id = self._next_concept_id
        self._library[concept_id] = fp
        self._next_concept_id += 1
        return concept_id
