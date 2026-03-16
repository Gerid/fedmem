"""Dynamic memory bank for concept lifecycle management.

Maintains a library of concept fingerprints with spawn/merge/shrink
operations, enabling online adaptation to new, recurring, and obsolete
concepts during federated learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..concept_tracker.fingerprint import ConceptFingerprint


@dataclass
class MemoryBankConfig:
    """Configuration for DynamicMemoryBank.

    Parameters
    ----------
    max_concepts : int
        Hard cap on the number of concepts in the bank.
    merge_threshold : float
        Similarity above which two concepts are merged.
    min_count : float
        Minimum fingerprint count; concepts below this are pruned.
    merge_every : int
        Run merge check every this many steps.
    shrink_every : int
        Run shrink check every this many steps.
    """

    max_concepts: int = 20
    merge_threshold: float = 0.85
    min_count: float = 5.0
    merge_every: int = 5
    shrink_every: int = 5

    def __post_init__(self) -> None:
        if self.max_concepts < 1:
            raise ValueError(f"max_concepts must be >= 1, got {self.max_concepts}")
        if not 0.0 < self.merge_threshold <= 1.0:
            raise ValueError(
                f"merge_threshold must be in (0, 1], got {self.merge_threshold}"
            )
        if self.min_count < 0:
            raise ValueError(f"min_count must be >= 0, got {self.min_count}")


@dataclass
class SpawnResult:
    """Result of a spawn operation.

    Parameters
    ----------
    new_concept_id : int
        The concept ID assigned (new or existing if absorbed).
    absorbed : bool
        True if the fingerprint was absorbed into an existing concept
        because max_concepts was reached.
    """

    new_concept_id: int
    absorbed: bool


class DynamicMemoryBank:
    """Online concept memory with spawn/merge/shrink lifecycle.

    Parameters
    ----------
    config : MemoryBankConfig
        Bank configuration.
    n_features : int
        Feature dimensionality for new fingerprints.
    n_classes : int
        Number of class labels.
    """

    def __init__(
        self,
        config: MemoryBankConfig | None = None,
        n_features: int = 2,
        n_classes: int = 2,
    ):
        self.config = config or MemoryBankConfig()
        self.n_features = n_features
        self.n_classes = n_classes
        self._library: dict[int, ConceptFingerprint] = {}
        self._next_id: int = 0
        self._step_count: int = 0

    @property
    def concept_library(self) -> dict[int, ConceptFingerprint]:
        """Return the current concept library (read-only view)."""
        return dict(self._library)

    @property
    def n_concepts(self) -> int:
        """Number of concepts currently in the bank."""
        return len(self._library)

    def get_fingerprint(self, concept_id: int) -> ConceptFingerprint | None:
        """Get fingerprint for a concept, or None if not found.

        Parameters
        ----------
        concept_id : int

        Returns
        -------
        ConceptFingerprint or None
        """
        return self._library.get(concept_id)

    def spawn_from_fingerprint(self, fp: ConceptFingerprint) -> SpawnResult:
        """Create a new concept from a fingerprint, or absorb if at capacity.

        Parameters
        ----------
        fp : ConceptFingerprint
            The fingerprint to spawn from.

        Returns
        -------
        SpawnResult
        """
        if len(self._library) < self.config.max_concepts:
            cid = self._next_id
            self._next_id += 1
            new_fp = ConceptFingerprint(self.n_features, self.n_classes)
            _merge_fingerprint_into(new_fp, fp)
            self._library[cid] = new_fp
            return SpawnResult(new_concept_id=cid, absorbed=False)

        # At capacity: absorb into the most similar existing concept
        best_id = self._find_most_similar(fp)
        if best_id is not None:
            self.absorb_fingerprint(best_id, fp)
            return SpawnResult(new_concept_id=best_id, absorbed=True)

        # Fallback (empty library but max_concepts < 1 — shouldn't happen)
        cid = self._next_id
        self._next_id += 1
        new_fp = ConceptFingerprint(self.n_features, self.n_classes)
        _merge_fingerprint_into(new_fp, fp)
        self._library[cid] = new_fp
        return SpawnResult(new_concept_id=cid, absorbed=False)

    def absorb_fingerprint(self, concept_id: int, fp: ConceptFingerprint) -> None:
        """Merge a fingerprint into an existing concept via Welford update.

        Parameters
        ----------
        concept_id : int
            Target concept ID.
        fp : ConceptFingerprint
            Fingerprint to absorb.

        Raises
        ------
        KeyError
            If concept_id is not in the bank.
        """
        if concept_id not in self._library:
            raise KeyError(f"Concept {concept_id} not in memory bank")
        _merge_fingerprint_into(self._library[concept_id], fp)

    def maybe_merge(self) -> list[tuple[int, int]]:
        """Merge pairs of concepts whose similarity exceeds the threshold.

        Performs a single pass (greedy). Returns the list of merged pairs
        ``(kept_id, removed_id)``.

        Returns
        -------
        list[tuple[int, int]]
        """
        merged: list[tuple[int, int]] = []
        ids = list(self._library.keys())

        consumed: set[int] = set()
        for i in range(len(ids)):
            if ids[i] in consumed:
                continue
            for j in range(i + 1, len(ids)):
                if ids[j] in consumed:
                    continue
                sim = self._library[ids[i]].similarity(self._library[ids[j]])
                if sim > self.config.merge_threshold:
                    # Merge j into i
                    _merge_fingerprint_into(
                        self._library[ids[i]], self._library[ids[j]]
                    )
                    del self._library[ids[j]]
                    consumed.add(ids[j])
                    merged.append((ids[i], ids[j]))

        return merged

    def maybe_shrink(self) -> list[int]:
        """Remove concepts with fingerprint count below min_count.

        Never removes the last concept.

        Returns
        -------
        list[int]
            IDs of removed concepts.
        """
        if len(self._library) <= 1:
            return []

        to_remove = [
            cid
            for cid, fp in self._library.items()
            if fp.count < self.config.min_count
        ]
        # Never remove all concepts
        if len(to_remove) >= len(self._library):
            # Keep the one with highest count
            best = max(self._library, key=lambda c: self._library[c].count)
            to_remove = [c for c in to_remove if c != best]

        for cid in to_remove:
            del self._library[cid]

        return to_remove

    def step(self) -> None:
        """Advance one step, triggering maintenance if scheduled."""
        self._step_count += 1
        if self._step_count % self.config.merge_every == 0:
            self.maybe_merge()
        if self._step_count % self.config.shrink_every == 0:
            self.maybe_shrink()

    def _find_most_similar(self, fp: ConceptFingerprint) -> int | None:
        """Find the concept most similar to the given fingerprint.

        Parameters
        ----------
        fp : ConceptFingerprint

        Returns
        -------
        int or None
            Best matching concept ID, or None if library is empty.
        """
        if not self._library:
            return None
        best_id = -1
        best_sim = -1.0
        for cid, cfp in self._library.items():
            sim = cfp.similarity(fp)
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id


def _merge_fingerprint_into(target: ConceptFingerprint, source: ConceptFingerprint) -> None:
    """Merge source fingerprint statistics into target (parallel Welford).

    Parameters
    ----------
    target : ConceptFingerprint
        Destination fingerprint (modified in-place).
    source : ConceptFingerprint
        Source fingerprint (not modified).
    """
    if source.count == 0:
        return

    n_a = target._count
    n_b = source._count
    n_ab = n_a + n_b

    if n_ab == 0:
        return

    delta = source._mean - target._mean

    # Combined mean
    new_mean = (n_a * target._mean + n_b * source._mean) / n_ab

    # Combined M2 (parallel Welford)
    new_M2 = target._M2 + source._M2 + np.outer(delta, delta) * (n_a * n_b / n_ab)

    # Combined label counts
    new_labels = target._label_counts + source._label_counts

    # Combined per-class feature means
    n_classes = target.n_classes
    for c in range(n_classes):
        na_c = target._class_counts[c]
        nb_c = source._class_counts[c]
        nab_c = na_c + nb_c
        if nab_c > 0:
            target._class_means[c] = (
                na_c * target._class_means[c] + nb_c * source._class_means[c]
            ) / nab_c
    target._class_counts = target._class_counts + source._class_counts

    target._count = n_ab
    target._mean = new_mean
    target._M2 = new_M2
    target._label_counts = new_labels
