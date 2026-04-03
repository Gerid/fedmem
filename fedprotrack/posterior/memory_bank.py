"""Dynamic memory bank for concept lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ..concept_tracker.fingerprint import ConceptFingerprint
from .retrieval_keys import (
    CompositeRetrievalKey,
    RetrievalKeyConfig,
    SupportsSimilarity,
    build_retrieval_key,
)


@dataclass
class MemoryBankConfig:
    """Configuration for DynamicMemoryBank."""

    max_concepts: int = 20
    merge_threshold: float = 0.85
    merge_min_support: int = 1
    min_count: float = 5.0
    merge_every: int = 5
    shrink_every: int = 5
    preserve_dormant_models: bool = False
    retrieval_key_config: RetrievalKeyConfig = field(default_factory=RetrievalKeyConfig)
    enable_trust_estimation: bool = False
    trust_buffer_size: int = 5
    trust_decay: float = 0.7
    trust_promotion_threshold: int = 2

    def __post_init__(self) -> None:
        if self.max_concepts < 1:
            raise ValueError(f"max_concepts must be >= 1, got {self.max_concepts}")
        if not 0.0 < self.merge_threshold <= 1.0:
            raise ValueError(
                f"merge_threshold must be in (0, 1], got {self.merge_threshold}"
            )
        if self.merge_min_support < 1:
            raise ValueError(
                f"merge_min_support must be >= 1, got {self.merge_min_support}"
            )
        if self.min_count < 0:
            raise ValueError(f"min_count must be >= 0, got {self.min_count}")
        if self.trust_buffer_size < 1:
            raise ValueError(
                f"trust_buffer_size must be >= 1, got {self.trust_buffer_size}"
            )
        if not 0.0 < self.trust_decay <= 1.0:
            raise ValueError(
                f"trust_decay must be in (0, 1], got {self.trust_decay}"
            )
        if self.trust_promotion_threshold < 1:
            raise ValueError(
                f"trust_promotion_threshold must be >= 1, got {self.trust_promotion_threshold}"
            )


@dataclass
class SpawnResult:
    """Result of a spawn operation."""

    new_concept_id: int
    absorbed: bool


@dataclass
class MemorySlot:
    """Plan C-style concept slot while preserving legacy bank methods."""

    slot_id: int
    center_key: SupportsSimilarity
    semantic_anchor_set: ConceptFingerprint
    expert_state: dict[str, np.ndarray] | None = None
    uncertainty: float = 0.0
    age: int = 0
    recurrence: float = 1.0
    support_count: int = 1
    trajectory_state: dict[str, float] | None = None
    anchor_buffer: list[np.ndarray] = field(default_factory=list)
    is_provisional: bool = False
    trusted_update_count: int = 0


class DynamicMemoryBank:
    """Online concept memory with spawn/merge/shrink lifecycle."""

    def __init__(
        self,
        config: MemoryBankConfig | None = None,
        n_features: int = 2,
        n_classes: int = 2,
    ):
        self.config = config or MemoryBankConfig()
        self.n_features = n_features
        self.n_classes = n_classes
        self._slots: dict[int, MemorySlot] = {}
        # Compatibility mirrors: legacy code/tests still access these.
        self._library: dict[int, ConceptFingerprint] = {}
        self._model_store: dict[int, dict[str, np.ndarray]] = {}
        self._signature_store: dict[int, np.ndarray] = {}
        self._signature_counts: dict[int, float] = {}
        self._next_id: int = 0
        self._step_count: int = 0

    @property
    def concept_library(self) -> dict[int, ConceptFingerprint]:
        """Return the current concept library (legacy fingerprint view)."""
        return dict(self._library)

    @property
    def routing_library(self) -> dict[int, SupportsSimilarity]:
        """Return the retrieval-key view used for addressing."""
        return {
            concept_id: slot.center_key
            for concept_id, slot in self._slots.items()
        }

    @property
    def slots(self) -> dict[int, MemorySlot]:
        """Return a shallow copy of all memory slots."""
        return dict(self._slots)

    @property
    def n_concepts(self) -> int:
        return len(self._slots)

    def get_slot(self, concept_id: int) -> MemorySlot | None:
        """Return the slot for a concept."""
        return self._slots.get(concept_id)

    def get_fingerprint(self, concept_id: int) -> ConceptFingerprint | None:
        return self._library.get(concept_id)

    def get_model_params(self, concept_id: int) -> dict[str, np.ndarray] | None:
        params = self._model_store.get(concept_id)
        if params is not None:
            return {key: value.copy() for key, value in params.items()}
        return None

    def get_signature(self, concept_id: int) -> np.ndarray | None:
        """Get stored lightweight routing signature for a concept."""
        signature = self._signature_store.get(concept_id)
        if signature is None:
            return None
        return signature.copy()

    def store_model_params(
        self,
        concept_id: int,
        params: dict[str, np.ndarray],
    ) -> None:
        copied = {key: value.copy() for key, value in params.items()}
        self._model_store[concept_id] = copied
        slot = self._slots.get(concept_id)
        if slot is not None:
            slot.expert_state = {key: value.copy() for key, value in copied.items()}

    def absorb_signature(self, concept_id: int, signature: np.ndarray) -> None:
        """Merge a lightweight routing signature into an existing concept."""
        if concept_id not in self._library:
            raise KeyError(f"Concept {concept_id} not in memory bank")
        sig = np.asarray(signature, dtype=np.float64).reshape(-1)
        if sig.size == 0:
            return
        if concept_id not in self._signature_store:
            self._signature_store[concept_id] = sig.copy()
            self._signature_counts[concept_id] = 1.0
            return
        count = self._signature_counts.get(concept_id, 1.0)
        merged = (count * self._signature_store[concept_id] + sig) / (count + 1.0)
        self._signature_store[concept_id] = merged
        self._signature_counts[concept_id] = count + 1.0

    def spawn_from_fingerprint(self, fp: ConceptFingerprint) -> SpawnResult:
        if self.n_concepts < self.config.max_concepts:
            cid = self._next_id
            self._next_id += 1
            slot = self._build_slot(cid, fp)
            self._slots[cid] = slot
            self._sync_slot(cid)
            return SpawnResult(new_concept_id=cid, absorbed=False)

        best_id = self._find_most_similar(fp)
        if best_id is not None:
            self.absorb_fingerprint(best_id, fp)
            return SpawnResult(new_concept_id=best_id, absorbed=True)

        cid = self._next_id
        self._next_id += 1
        slot = self._build_slot(cid, fp)
        self._slots[cid] = slot
        self._sync_slot(cid)
        return SpawnResult(new_concept_id=cid, absorbed=False)

    def absorb_fingerprint(self, concept_id: int, fp: ConceptFingerprint) -> None:
        if concept_id not in self._slots:
            raise KeyError(f"Concept {concept_id} not in memory bank")
        slot = self._slots[concept_id]
        _merge_fingerprint_into(slot.semantic_anchor_set, fp)
        previous_key = (
            slot.center_key
            if isinstance(slot.center_key, CompositeRetrievalKey)
            else None
        )
        slot.center_key = build_retrieval_key(
            slot.semantic_anchor_set,
            config=self.config.retrieval_key_config,
            previous=previous_key,
        )
        slot.support_count += 1
        slot.recurrence += 1.0
        slot.age = 0
        slot.uncertainty = 1.0 / max(slot.support_count, 1)
        self._sync_slot(concept_id)

    def trust_weighted_update(
        self,
        concept_id: int,
        client_fingerprints: list[ConceptFingerprint],
        trust_scores: list[float],
    ) -> None:
        """Update a slot's centroid using trust-weighted fingerprint anchors.

        Parameters
        ----------
        concept_id : int
            Target concept slot.
        client_fingerprints : list[ConceptFingerprint]
            Fingerprints from clients assigned to this concept.
        trust_scores : list[float]
            Per-client trust scores in [0, 1].
        """
        if concept_id not in self._slots:
            raise KeyError(f"Concept {concept_id} not in memory bank")
        if len(client_fingerprints) != len(trust_scores):
            raise ValueError("client_fingerprints and trust_scores must have same length")
        if not client_fingerprints:
            return

        slot = self._slots[concept_id]
        cfg = self.config

        # Compute trust-weighted anchor from fingerprint means
        scores = np.asarray(trust_scores, dtype=np.float64)
        total_trust = float(scores.sum())
        if total_trust < 1e-12:
            return

        means = np.stack([fp._mean for fp in client_fingerprints], axis=0)
        weighted_anchor = (scores[:, None] * means).sum(axis=0) / total_trust

        # Append to buffer, trim to W
        slot.anchor_buffer.append(weighted_anchor)
        if len(slot.anchor_buffer) > cfg.trust_buffer_size:
            slot.anchor_buffer = slot.anchor_buffer[-cfg.trust_buffer_size:]

        # Rebuild centroid from exponentially-decayed buffer
        gamma = cfg.trust_decay
        n_buf = len(slot.anchor_buffer)
        weights = np.array(
            [gamma ** (n_buf - 1 - i) for i in range(n_buf)],
            dtype=np.float64,
        )
        total_w = float(weights.sum())
        if total_w < 1e-12:
            return

        buf_array = np.stack(slot.anchor_buffer, axis=0)
        new_mean = (weights[:, None] * buf_array).sum(axis=0) / total_w

        # Update slot's semantic anchor set mean directly
        slot.semantic_anchor_set._mean = new_mean
        slot.support_count += len(client_fingerprints)
        slot.recurrence += float(len(client_fingerprints))
        slot.age = 0
        slot.uncertainty = 1.0 / max(slot.support_count, 1)

        # Track trusted updates for promotion
        slot.trusted_update_count += 1
        if (
            slot.is_provisional
            and slot.trusted_update_count >= cfg.trust_promotion_threshold
        ):
            slot.is_provisional = False

        # Rebuild retrieval key and sync
        previous_key = (
            slot.center_key
            if isinstance(slot.center_key, CompositeRetrievalKey)
            else None
        )
        slot.center_key = build_retrieval_key(
            slot.semantic_anchor_set,
            config=self.config.retrieval_key_config,
            previous=previous_key,
        )
        self._sync_slot(concept_id)

    def batch_bootstrap(
        self,
        fingerprints: list[ConceptFingerprint],
    ) -> list[int]:
        """Bootstrap memory bank from a batch of fingerprints using clustering.

        Parameters
        ----------
        fingerprints : list[ConceptFingerprint]
            Initial fingerprints (e.g. from round 0 clients).

        Returns
        -------
        list[int]
            Cluster assignment for each fingerprint (concept IDs).
        """
        n = len(fingerprints)
        if n == 0:
            return []
        if n == 1:
            result = self.spawn_from_fingerprint(fingerprints[0])
            return [result.new_concept_id]

        # Compute pairwise cosine similarities
        means = np.stack([fp._mean for fp in fingerprints], axis=0)
        norms = np.linalg.norm(means, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = means / norms
        cos_sim = normed @ normed.T
        np.clip(cos_sim, -1.0, 1.0, out=cos_sim)

        # Convert to distance
        cos_dist = 1.0 - cos_sim
        np.fill_diagonal(cos_dist, 0.0)
        cos_dist = np.maximum(cos_dist, 0.0)

        # Find largest gap in sorted pairwise distances for auto threshold
        condensed = squareform(cos_dist, checks=False)
        sorted_dists = np.sort(condensed)
        if len(sorted_dists) > 1:
            gaps = np.diff(sorted_dists)
            gap_idx = int(np.argmax(gaps))
            auto_threshold = (sorted_dists[gap_idx] + sorted_dists[gap_idx + 1]) / 2.0
        else:
            auto_threshold = float(sorted_dists[0]) + 0.01 if len(sorted_dists) > 0 else 0.5

        # Agglomerative clustering
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=max(auto_threshold, 1e-6), criterion="distance")

        # Spawn one slot per cluster from cluster mean
        cluster_ids = sorted(set(labels))
        cluster_to_concept: dict[int, int] = {}
        for cid in cluster_ids:
            members = [i for i, lbl in enumerate(labels) if lbl == cid]
            representative = fingerprints[members[0]]
            result = self.spawn_from_fingerprint(representative)
            concept_id = result.new_concept_id
            cluster_to_concept[cid] = concept_id
            slot = self._slots[concept_id]
            slot.is_provisional = True
            # Absorb remaining cluster members
            for idx in members[1:]:
                self.absorb_fingerprint(concept_id, fingerprints[idx])

        # Build output assignment list
        assignments = [cluster_to_concept[int(lbl)] for lbl in labels]
        return assignments

    def maybe_merge(self) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        ids = list(self._slots.keys())
        consumed: set[int] = set()
        use_legacy_similarity = (
            self.config.retrieval_key_config.mode == "legacy_fingerprint"
        )
        for i in range(len(ids)):
            keep_id = ids[i]
            if keep_id in consumed or keep_id not in self._slots:
                continue
            for j in range(i + 1, len(ids)):
                remove_id = ids[j]
                if remove_id in consumed or remove_id not in self._slots:
                    continue
                keep_slot = self._slots[keep_id]
                remove_slot = self._slots[remove_id]
                if (
                    keep_slot.support_count < self.config.merge_min_support
                    or remove_slot.support_count < self.config.merge_min_support
                ):
                    continue
                semantic_sim = float(
                    keep_slot.semantic_anchor_set.similarity(
                        remove_slot.semantic_anchor_set,
                    )
                )
                if use_legacy_similarity:
                    sim = semantic_sim
                else:
                    key_sim = float(
                        keep_slot.center_key.similarity(remove_slot.center_key)
                    )
                    sim = min(key_sim, semantic_sim)
                if sim > self.config.merge_threshold:
                    _merge_fingerprint_into(
                        keep_slot.semantic_anchor_set,
                        remove_slot.semantic_anchor_set,
                    )
                    keep_slot.center_key = build_retrieval_key(
                        keep_slot.semantic_anchor_set,
                        config=self.config.retrieval_key_config,
                        previous=(
                            keep_slot.center_key
                            if isinstance(keep_slot.center_key, CompositeRetrievalKey)
                            else None
                        ),
                    )
                    keep_slot.support_count += remove_slot.support_count
                    keep_slot.recurrence += remove_slot.recurrence
                    keep_slot.age = min(keep_slot.age, remove_slot.age)
                    keep_slot.uncertainty = min(
                        keep_slot.uncertainty,
                        remove_slot.uncertainty,
                    )
                    if (
                        remove_id in self._model_store
                        and keep_id not in self._model_store
                    ):
                        self._model_store[keep_id] = self._model_store[remove_id]
                        keep_slot.expert_state = self.get_model_params(keep_id)
                    if remove_id in self._signature_store:
                        if keep_id in self._signature_store:
                            count_i = self._signature_counts.get(keep_id, 1.0)
                            count_j = self._signature_counts.get(remove_id, 1.0)
                            total = count_i + count_j
                            self._signature_store[keep_id] = (
                                count_i * self._signature_store[keep_id]
                                + count_j * self._signature_store[remove_id]
                            ) / max(total, 1.0)
                            self._signature_counts[keep_id] = total
                        else:
                            self._signature_store[keep_id] = (
                                self._signature_store[remove_id].copy()
                            )
                            self._signature_counts[keep_id] = (
                                self._signature_counts.get(remove_id, 1.0)
                            )
                    self._delete_slot(remove_id)
                    consumed.add(remove_id)
                    merged.append((keep_id, remove_id))
                    self._sync_slot(keep_id)
        return merged

    def maybe_shrink(self) -> list[int]:
        if self.n_concepts <= 1:
            return []

        to_remove = [
            concept_id
            for concept_id, slot in self._slots.items()
            if slot.semantic_anchor_set.count < self.config.min_count
        ]
        if len(to_remove) >= self.n_concepts:
            best = max(
                self._slots,
                key=lambda concept_id: self._slots[concept_id].semantic_anchor_set.count,
            )
            to_remove = [concept_id for concept_id in to_remove if concept_id != best]

        for concept_id in to_remove:
            self._delete_slot(concept_id)

        return to_remove

    def step(self) -> None:
        self._step_count += 1
        for slot in self._slots.values():
            slot.age += 1
            slot.recurrence *= 0.995
        if self._step_count % self.config.merge_every == 0:
            self.maybe_merge()
        if self._step_count % self.config.shrink_every == 0:
            self.maybe_shrink()

    def _build_slot(self, concept_id: int, fp: ConceptFingerprint) -> MemorySlot:
        new_fp = ConceptFingerprint(
            self.n_features,
            self.n_classes,
            feature_groups=fp.feature_groups,
        )
        _merge_fingerprint_into(new_fp, fp)
        return MemorySlot(
            slot_id=concept_id,
            center_key=build_retrieval_key(
                new_fp,
                config=self.config.retrieval_key_config,
            ),
            semantic_anchor_set=new_fp,
            uncertainty=1.0 / max(int(new_fp.count), 1),
            age=0,
            recurrence=1.0,
            support_count=1,
            trajectory_state={"velocity": 0.0},
        )

    def _sync_slot(self, concept_id: int) -> None:
        slot = self._slots[concept_id]
        self._library[concept_id] = slot.semantic_anchor_set
        if slot.expert_state is not None:
            self._model_store[concept_id] = {
                key: value.copy() for key, value in slot.expert_state.items()
            }

    def _delete_slot(self, concept_id: int) -> None:
        self._slots.pop(concept_id, None)
        self._library.pop(concept_id, None)
        if not self.config.preserve_dormant_models:
            self._model_store.pop(concept_id, None)
        self._signature_store.pop(concept_id, None)
        self._signature_counts.pop(concept_id, None)

    def _find_most_similar(self, fp: ConceptFingerprint) -> int | None:
        if not self._slots:
            return None
        if self.config.retrieval_key_config.mode == "legacy_fingerprint":
            best_id = -1
            best_sim = -1.0
            for concept_id, slot in self._slots.items():
                sim = slot.semantic_anchor_set.similarity(fp)
                if sim > best_sim:
                    best_sim = sim
                    best_id = concept_id
            return best_id
        candidate_key = build_retrieval_key(
            fp,
            config=self.config.retrieval_key_config,
        )
        best_id = -1
        best_sim = -1.0
        for concept_id, slot in self._slots.items():
            sim = slot.center_key.similarity(candidate_key)
            if sim > best_sim:
                best_sim = sim
                best_id = concept_id
        return best_id


def _merge_fingerprint_into(target: ConceptFingerprint, source: ConceptFingerprint) -> None:
    """Merge source fingerprint statistics into target (parallel Welford)."""
    if source.count == 0:
        return
    if target.feature_groups is None and source.feature_groups is not None:
        target._feature_groups = source.feature_groups

    n_a = target._count
    n_b = source._count
    n_ab = n_a + n_b
    if n_ab == 0:
        return

    delta = source._mean - target._mean
    new_mean = (n_a * target._mean + n_b * source._mean) / n_ab
    new_M2 = target._M2 + source._M2 + np.outer(delta, delta) * (n_a * n_b / n_ab)
    new_labels = target._label_counts + source._label_counts

    n_classes = target.n_classes
    new_class_counts = target._class_counts + source._class_counts
    new_class_means = np.zeros_like(target._class_means)
    for class_id in range(n_classes):
        count_a = target._class_counts[class_id]
        count_b = source._class_counts[class_id]
        count_ab = count_a + count_b
        if count_ab > 0:
            new_class_means[class_id] = (
                count_a * target._class_means[class_id]
                + count_b * source._class_means[class_id]
            ) / count_ab

    target._count = n_ab
    target._mean = new_mean
    target._M2 = new_M2
    target._label_counts = new_labels
    target._class_counts = new_class_counts
    target._class_means = new_class_means
