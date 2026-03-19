"""Two-phase communication protocol for FedProTrack.

Phase A: lightweight fingerprint exchange for concept identification.
Phase B: full-model aggregation within identified concept clusters.

This two-phase design reduces communication cost by only performing
expensive model exchange among clients sharing the same concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..baselines.comm_tracker import fingerprint_bytes, model_bytes
from ..concept_tracker.fingerprint import ConceptFingerprint
from ..federation.aggregator import (
    FedAvgAggregator,
    NamespacedExpertAggregator,
    has_namespaced_params,
    merge_param_namespaces,
    split_param_namespaces,
)
from .gibbs import GibbsPosterior, PosteriorAssignment, TransitionPrior
from .memory_bank import DynamicMemoryBank, MemoryBankConfig
from .retrieval_keys import RetrievalKeyConfig


@dataclass
class TwoPhaseConfig:
    """Configuration for the two-phase FedProTrack protocol.

    Parameters
    ----------
    omega : float
        Inverse temperature for Gibbs posterior sharpness.
    kappa : float
        Stickiness for the transition prior.
    novelty_threshold : float
        MAP probability threshold below which a concept is novel.
    loss_novelty_threshold : float
        Absolute loss threshold for novelty detection. When the best
        concept's loss exceeds this value, the observation is considered
        novel. This is critical for escaping the single-concept trap
        where the posterior is always 1.0.
    sticky_dampening : float
        When a client's previous concept has posterior probability >=
        ``sticky_posterior_gate``, the effective loss_novelty_threshold
        is multiplied by this factor. Higher values suppress over-
        spawning during asynchronous drift (alpha > 0) by trusting
        the transition prior's "stay" signal. Default 2.0.
    sticky_posterior_gate : float
        Minimum posterior probability for the previous concept to
        activate sticky dampening. Default 0.3.
    model_loss_weight : float
        Weight for blending model-based loss (classification error)
        into the novelty decision. 0.0 = fingerprint only, 1.0 = model
        loss only. Default 0.3.
    post_spawn_merge : bool
        Whether to run an immediate merge pass after spawning novel
        concepts. This catches duplicate concepts created across
        rounds when clients switch asynchronously. Default True.
    merge_threshold : float
        Similarity threshold for merging concepts in the memory bank.
    merge_min_support : int
        Minimum slot support count required before a concept is eligible
        for merging. Prevents freshly spawned slots from collapsing
        immediately.
    min_count : float
        Minimum fingerprint count for concept survival.
    max_concepts : int
        Hard cap on the number of concepts.
    max_spawn_clusters_per_round : int or None
        Optional cap on how many novel clusters may spawn in a single
        addressing round. When set, novel clusters are ranked by support
        size and novelty gap, and only the strongest clusters spawn while
        the remainder fall back to their MAP existing concept. Default
        None (no cap).
    novelty_hysteresis_rounds : int
        Number of consecutive addressing rounds a client must remain novel
        before a new concept is allowed to spawn. ``1`` reproduces the
        current behaviour (no hysteresis). Values > 1 suppress one-off
        overspawn by falling back to the best existing concept until the
        novelty signal persists.
    merge_every : int
        Run merge maintenance every this many federation rounds.
    shrink_every : int
        Run shrink maintenance every this many federation rounds.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    """

    omega: float = 2.0
    kappa: float = 0.6
    novelty_threshold: float = 0.3
    loss_novelty_threshold: float = 0.05
    sticky_dampening: float = 1.0
    sticky_posterior_gate: float = 0.3
    model_loss_weight: float = 0.0
    post_spawn_merge: bool = True
    merge_threshold: float = 0.98
    merge_min_support: int = 1
    min_count: float = 5.0
    max_concepts: int = 20
    max_spawn_clusters_per_round: int | None = None
    novelty_hysteresis_rounds: int = 1
    merge_every: int = 2
    shrink_every: int = 5
    key_mode: str = "fingerprint"
    key_ema_decay: float = 0.0
    key_style_weight: float = 0.25
    key_semantic_weight: float = 0.30
    key_prototype_weight: float = 0.45
    global_shared_aggregation: bool = True
    entropy_freeze_threshold: float | None = None
    adaptive_addressing: bool = False
    addressing_min_round_interval: int = 1
    addressing_drift_threshold: float = 0.0
    n_features: int = 2
    n_classes: int = 2


@dataclass
class PhaseAResult:
    """Result of Phase A (fingerprint exchange).

    Parameters
    ----------
    assignments : dict[int, int]
        Client ID -> assigned concept ID.
    posteriors : dict[int, PosteriorAssignment]
        Client ID -> full posterior assignment.
    bytes_up : float
        Upload bytes (clients -> server): all fingerprints.
    bytes_down : float
        Download bytes (server -> clients): concept assignments (4 bytes each).
    """

    assignments: dict[int, int]
    posteriors: dict[int, PosteriorAssignment]
    bytes_up: float
    bytes_down: float
    spawned: int = 0
    merged: int = 0
    pruned: int = 0
    addressing_performed: bool = True
    avg_posterior_entropy: float | None = None
    assignment_switch_rate: float | None = None
    avg_clients_per_concept: float | None = None
    singleton_group_ratio: float | None = None
    routing_consistency: float | None = None
    library_size_before: int = 0
    client_fp_losses: dict[int, float] = field(default_factory=dict)
    client_effective_thresholds: dict[int, float] = field(default_factory=dict)
    client_map_probabilities: dict[int, float] = field(default_factory=dict)

    @property
    def total_bytes(self) -> float:
        return self.bytes_up + self.bytes_down


@dataclass
class PhaseBResult:
    """Result of Phase B (model aggregation within concept clusters).

    Parameters
    ----------
    aggregated_params : dict[int, dict[str, np.ndarray]]
        Concept ID -> aggregated model parameters.
    bytes_up : float
        Upload bytes (clients -> server): model parameters.
    bytes_down : float
        Download bytes (server -> clients): aggregated models.
    """

    aggregated_params: dict[int, dict[str, np.ndarray]]
    bytes_up: float
    bytes_down: float

    @property
    def total_bytes(self) -> float:
        return self.bytes_up + self.bytes_down


class TwoPhaseFedProTrack:
    """Server-side coordinator for the two-phase FedProTrack protocol.

    Parameters
    ----------
    config : TwoPhaseConfig
        Protocol configuration.
    """

    def __init__(self, config: TwoPhaseConfig):
        self.config = config

        mb_config = MemoryBankConfig(
            max_concepts=config.max_concepts,
            merge_threshold=config.merge_threshold,
            merge_min_support=config.merge_min_support,
            min_count=config.min_count,
            merge_every=config.merge_every,
            shrink_every=config.shrink_every,
            retrieval_key_config=RetrievalKeyConfig(
                mode=config.key_mode,
                ema_decay=config.key_ema_decay,
                style_weight=config.key_style_weight,
                semantic_weight=config.key_semantic_weight,
                prototype_weight=config.key_prototype_weight,
            ),
        )
        self.memory_bank = DynamicMemoryBank(
            config=mb_config,
            n_features=config.n_features,
            n_classes=config.n_classes,
        )

        self.gibbs = GibbsPosterior(
            omega=config.omega,
            transition_prior=TransitionPrior(kappa=config.kappa),
            novelty_threshold=config.novelty_threshold,
        )

        self._aggregator = FedAvgAggregator()
        self._round: int = 0
        self._last_addressing_round: int = -1
        self._novelty_streaks: dict[int, int] = {}

    def phase_a(
        self,
        client_fingerprints: dict[int, ConceptFingerprint],
        prev_assignments: dict[int, int] | None = None,
        client_model_losses: dict[int, float] | None = None,
    ) -> PhaseAResult:
        """Execute Phase A: lightweight concept identification.

        Uses batch assignment: compute all posteriors first, cluster novel
        clients together, then update the memory bank. This avoids the
        sequential processing bias where early clients always get assigned
        to existing concepts.

        Parameters
        ----------
        client_fingerprints : dict[int, ConceptFingerprint]
            Client ID -> fingerprint built from current data batch.
        prev_assignments : dict[int, int] or None
            Previous round's concept assignments (client_id -> concept_id).
        client_model_losses : dict[int, float] or None
            Client ID -> classification error rate (0=perfect, 1=all wrong).
            Used as a second channel for novelty gating: if a client's model
            performs well (low loss), it suppresses novelty even if fingerprint
            distance is moderate. Default None (fingerprint-only).

        Returns
        -------
        PhaseAResult
        """
        K = len(client_fingerprints)
        library_size_before = self.memory_bank.n_concepts
        fp_bytes = fingerprint_bytes(
            self.config.n_features, self.config.n_classes,
            precision_bits=16, include_global_mean=False,
        )
        bytes_up = K * fp_bytes

        assignments: dict[int, int] = {}
        posteriors: dict[int, PosteriorAssignment] = {}

        if K == 0:
            return PhaseAResult(
                assignments={}, posteriors={},
                bytes_up=0.0, bytes_down=0.0,
                library_size_before=library_size_before,
            )

        if (
            prev_assignments is not None
            and self.config.adaptive_addressing
            and self.memory_bank.n_concepts > 0
        ):
            drift_scores: list[float] = []
            routing_library = self.memory_bank.routing_library
            for client_id, fp in client_fingerprints.items():
                prev_cid = prev_assignments.get(client_id)
                if prev_cid is None or prev_cid not in routing_library:
                    continue
                drift_scores.append(
                    self.gibbs.compute_loss(fp, routing_library[prev_cid])
                )
            mean_drift = float(np.mean(drift_scores)) if drift_scores else 0.0
            if (
                mean_drift <= self.config.addressing_drift_threshold
                and (self._round - self._last_addressing_round)
                < self.config.addressing_min_round_interval
            ):
                frozen_posteriors: dict[int, PosteriorAssignment] = {}
                for client_id, concept_id in prev_assignments.items():
                    frozen_posteriors[client_id] = PosteriorAssignment(
                        probabilities={concept_id: 1.0},
                        map_concept_id=concept_id,
                        is_novel=False,
                        entropy=0.0,
                    )
                return PhaseAResult(
                    assignments=dict(prev_assignments),
                    posteriors=frozen_posteriors,
                    bytes_up=bytes_up,
                    bytes_down=K * 4.0,
                    addressing_performed=False,
                    avg_posterior_entropy=0.0,
                    assignment_switch_rate=0.0,
                    avg_clients_per_concept=_avg_clients_per_concept(prev_assignments),
                    singleton_group_ratio=_singleton_group_ratio(prev_assignments),
                    routing_consistency=1.0,
                    library_size_before=library_size_before,
                )

        # --- Bootstrap: if no concepts yet, create the first from all clients ---
        if self.memory_bank.n_concepts == 0:
            # Spawn first concept from the first client
            first_cid = next(iter(client_fingerprints))
            result = self.memory_bank.spawn_from_fingerprint(
                client_fingerprints[first_cid]
            )
            first_concept_id = result.new_concept_id
            assignment = PosteriorAssignment(
                probabilities={first_concept_id: 1.0},
                map_concept_id=first_concept_id,
                is_novel=True,
                entropy=0.0,
            )
            assignments[first_cid] = first_concept_id
            posteriors[first_cid] = assignment

            # Remove first client from remaining processing
            remaining_fps = {
                cid: fp for cid, fp in client_fingerprints.items()
                if cid != first_cid
            }
        else:
            remaining_fps = dict(client_fingerprints)

        # --- Pass 1: compute posteriors for all remaining clients ---
        novel_clients: list[int] = []  # clients flagged as novel
        assigned_clients: dict[int, int] = {}  # client -> concept_id
        client_fp_losses: dict[int, float] = {}
        client_effective_thresholds: dict[int, float] = {}
        client_map_probabilities: dict[int, float] = {}

        for client_id, fp in remaining_fps.items():
            prev_cid = (
                prev_assignments.get(client_id) if prev_assignments else None
            )
            library = self.memory_bank.routing_library

            assignment = self.gibbs.compute_posterior(fp, library, prev_cid)
            posteriors[client_id] = assignment

            # Check novelty via posterior probability threshold
            is_novel = assignment.is_novel

            # Additional loss-based novelty check (critical for single-concept
            # escape): even if posterior is high (e.g. 1.0 with 1 concept),
            # a high loss means the concept is a poor fit.
            best_concept_key = library[assignment.map_concept_id]
            fp_loss = self.gibbs.compute_loss(fp, best_concept_key)
            # --- Sticky dampening (Fix #1 for async drift) ---
            # When the transition prior says "stay" (high posterior for the
            # previous concept), raise the novelty threshold. This prevents
            # flagging a client as novel just because its per-step fingerprint
            # has moderate noise — the temporal prior is informative.
            #
            # CRITICAL: only apply sticky dampening when there are >= 2
            # concepts in the library. With a single concept, the posterior
            # is trivially 1.0 (no real choice), so stickiness would
            # prevent escaping the single-concept trap on first drift.
            effective_threshold = self.config.loss_novelty_threshold
            n_library = len(library)
            if (
                n_library >= 2
                and prev_cid is not None
                and prev_cid in assignment.probabilities
            ):
                prev_prob = assignment.probabilities[prev_cid]
                if prev_prob >= self.config.sticky_posterior_gate:
                    effective_threshold *= self.config.sticky_dampening
            client_fp_losses[client_id] = float(fp_loss)
            client_effective_thresholds[client_id] = float(effective_threshold)
            client_map_probabilities[client_id] = float(
                assignment.probabilities[assignment.map_concept_id]
            )

            # --- Model-based loss as second channel (Fix #3) ---
            # Model loss acts as a novelty SUPPRESSOR: if the client's
            # model is still predicting well (low error rate), this is
            # strong evidence that the concept hasn't actually changed,
            # even if the fingerprint distance is moderate.
            # It does NOT trigger novelty (that's the fingerprint's job).
            model_suppresses = False
            if (
                client_model_losses is not None
                and client_id in client_model_losses
                and self.config.model_loss_weight > 0.0
            ):
                model_loss = client_model_losses[client_id]
                # If model accuracy is high (loss < weight threshold),
                # suppress fingerprint-based novelty detection
                if model_loss < self.config.model_loss_weight:
                    model_suppresses = True

            if fp_loss > effective_threshold and not model_suppresses:
                is_novel = True

            if (
                self.config.entropy_freeze_threshold is not None
                and prev_cid is not None
                and assignment.entropy >= self.config.entropy_freeze_threshold
                and prev_cid in library
            ):
                is_novel = False
                self._novelty_streaks[client_id] = 0
                assigned_clients[client_id] = prev_cid
                continue

            if is_novel:
                streak = self._novelty_streaks.get(client_id, 0) + 1
                self._novelty_streaks[client_id] = streak
                if int(self.config.novelty_hysteresis_rounds) > 1 and streak < int(
                    self.config.novelty_hysteresis_rounds
                ):
                    assigned_clients[client_id] = assignment.map_concept_id
                    continue
                novel_clients.append(client_id)
            else:
                self._novelty_streaks[client_id] = 0
                assigned_clients[client_id] = assignment.map_concept_id

        # --- Pass 2: cluster novel clients by fingerprint similarity ---
        n_spawned = 0
        n_merged = 0
        n_pruned = 0

        if novel_clients:
            novel_clusters = self._cluster_novel_clients(
                novel_clients, client_fingerprints,
            )
            novel_clusters, capped_clients = self._cap_novel_clusters(
                novel_clusters,
                client_fp_losses=client_fp_losses,
                client_effective_thresholds=client_effective_thresholds,
            )
            for client_id in capped_clients:
                assigned_clients[client_id] = posteriors[client_id].map_concept_id

            # Spawn one new concept per cluster of truly novel clients
            for cluster in novel_clusters:
                # Build a merged fingerprint for the cluster
                representative_fp = client_fingerprints[cluster[0]]
                spawn_result = self.memory_bank.spawn_from_fingerprint(
                    representative_fp
                )
                if not spawn_result.absorbed:
                    n_spawned += 1
                new_cid = spawn_result.new_concept_id
                for client_id in cluster:
                    self._novelty_streaks[client_id] = 0
                    assignments[client_id] = new_cid
                    if client_id != cluster[0]:
                        self.memory_bank.absorb_fingerprint(
                            new_cid, client_fingerprints[client_id]
                        )

            # --- Post-spawn merge (Fix #2 for async drift) ---
            # Immediately check whether newly spawned concepts are similar
            # to existing ones. This catches the cross-round duplication
            # where client A switches at t and spawns B', then client C
            # switches at t+1 and spawns B'' — both represent concept B.
            if self.config.post_spawn_merge:
                merge_pairs = self.memory_bank.maybe_merge()
                n_merged += len(merge_pairs)
                # Remap any assignments whose concept was merged away
                self._remap_merged_assignments(assignments, merge_pairs)
                # Also remap assigned_clients in case their concepts were merged
                self._remap_merged_assignments(assigned_clients, merge_pairs)

        # --- Pass 3: absorb assigned clients into their MAP concepts ---
        for client_id, concept_id in assigned_clients.items():
            assignments[client_id] = concept_id
            self.memory_bank.absorb_fingerprint(
                concept_id, client_fingerprints[client_id]
            )

        # Run memory bank maintenance (merge + shrink on schedule)
        n_before = self.memory_bank.n_concepts
        self.memory_bank.step()
        n_after = self.memory_bank.n_concepts
        # step() may merge or prune concepts
        n_removed_in_step = max(0, n_before - n_after)
        n_pruned += n_removed_in_step
        self._round += 1

        # Download cost: 4 bytes per client (concept ID as int32)
        bytes_down = K * 4.0
        self._last_addressing_round = self._round

        return PhaseAResult(
            assignments=assignments,
            posteriors=posteriors,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
            spawned=n_spawned,
            merged=n_merged,
            pruned=n_pruned,
            addressing_performed=True,
            avg_posterior_entropy=_avg_assignment_entropy(posteriors),
            assignment_switch_rate=_assignment_switch_rate(assignments, prev_assignments),
            avg_clients_per_concept=_avg_clients_per_concept(assignments),
            singleton_group_ratio=_singleton_group_ratio(assignments),
            routing_consistency=_routing_consistency(assignments, prev_assignments),
            library_size_before=library_size_before,
            client_fp_losses=client_fp_losses,
            client_effective_thresholds=client_effective_thresholds,
            client_map_probabilities=client_map_probabilities,
        )

    def _remap_merged_assignments(
        self, assignments: dict[int, int],
        merge_pairs: list[tuple[int, int]],
    ) -> None:
        """Remap client assignments after a merge pass.

        Uses the merge mapping directly: each ``(kept_id, removed_id)``
        pair maps clients assigned to ``removed_id`` to ``kept_id``.

        Parameters
        ----------
        assignments : dict[int, int]
            Client ID -> concept ID (mutated in place).
        merge_pairs : list[tuple[int, int]]
            Pairs of ``(kept_id, removed_id)`` from ``maybe_merge()``.
        """
        if not merge_pairs:
            return

        # Build removed -> kept mapping (chain through transitive merges)
        remap: dict[int, int] = {}
        for kept_id, removed_id in merge_pairs:
            # Follow chain: if kept_id was itself remapped, use its target
            final_kept = kept_id
            while final_kept in remap:
                final_kept = remap[final_kept]
            remap[removed_id] = final_kept

        for client_id, concept_id in assignments.items():
            if concept_id in remap:
                assignments[client_id] = remap[concept_id]

    def _cluster_novel_clients(
        self,
        novel_client_ids: list[int],
        client_fingerprints: dict[int, ConceptFingerprint],
    ) -> list[list[int]]:
        """Cluster novel clients by fingerprint similarity (single-linkage).

        Uses ``merge_threshold`` as the clustering similarity threshold
        (same threshold used for concept merging in the memory bank).
        This ensures that novel clients representing the same new concept
        are grouped together and only one concept is spawned per group.

        Reassignment of novel clients to *existing* memory bank concepts
        is deferred to the post-spawn merge pass, which is more principled
        because it operates on fully-formed concept fingerprints rather
        than noisy single-client observations.

        Parameters
        ----------
        novel_client_ids : list[int]
            Client IDs flagged as novel.
        client_fingerprints : dict[int, ConceptFingerprint]
            All client fingerprints.

        Returns
        -------
        list[list[int]]
            List of clusters, each a list of client IDs.
        """
        if len(novel_client_ids) <= 1:
            return [novel_client_ids] if novel_client_ids else []

        # Use merge_threshold for clustering: two novel clients belong
        # to the same new concept if their fingerprints are similar enough
        # to be merged. This is more permissive than the old
        # (1 - loss_novelty_threshold) and allows clients that switched
        # to the same concept at different times to be clustered together.
        sim_threshold = self.config.merge_threshold

        # Average-linkage greedy clustering
        clusters: list[list[int]] = [[novel_client_ids[0]]]
        for cid in novel_client_ids[1:]:
            fp = client_fingerprints[cid]
            best_cluster_idx = -1
            best_avg_sim = -1.0
            for idx, cluster in enumerate(clusters):
                avg_sim = float(np.mean([
                    fp.similarity(client_fingerprints[member])
                    for member in cluster
                ]))
                if avg_sim >= sim_threshold and avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_cluster_idx = idx
            if best_cluster_idx >= 0:
                clusters[best_cluster_idx].append(cid)
            else:
                clusters.append([cid])

        return clusters

    def _cap_novel_clusters(
        self,
        novel_clusters: list[list[int]],
        *,
        client_fp_losses: dict[int, float],
        client_effective_thresholds: dict[int, float],
    ) -> tuple[list[list[int]], list[int]]:
        """Limit how many novel clusters may spawn in a single round."""
        cap = self.config.max_spawn_clusters_per_round
        if cap is None or len(novel_clusters) <= int(cap):
            return novel_clusters, []

        ranked = sorted(
            novel_clusters,
            key=lambda cluster: (
                -len(cluster),
                -float(np.mean([
                    client_fp_losses[client_id] - client_effective_thresholds[client_id]
                    for client_id in cluster
                ])),
                int(min(cluster)),
            ),
        )
        kept = ranked[:int(cap)]
        capped_clients = [
            client_id
            for cluster in ranked[int(cap):]
            for client_id in cluster
        ]
        return kept, capped_clients

    def phase_b(
        self,
        client_params: dict[int, dict[str, np.ndarray]],
        concept_assignments: dict[int, int],
    ) -> PhaseBResult:
        """Execute Phase B: model aggregation within concept clusters.

        Parameters
        ----------
        client_params : dict[int, dict[str, np.ndarray]]
            Client ID -> model parameters (coef, intercept).
        concept_assignments : dict[int, int]
            Client ID -> concept ID (from Phase A).

        Returns
        -------
        PhaseBResult
        """
        # Group clients by concept
        concept_groups: dict[int, list[int]] = {}
        for client_id, concept_id in concept_assignments.items():
            if concept_id not in concept_groups:
                concept_groups[concept_id] = []
            concept_groups[concept_id].append(client_id)

        # Upload cost: each client sends its model
        bytes_up = sum(
            model_bytes(params) for params in client_params.values()
        )

        # Aggregate within each concept cluster
        aggregated: dict[int, dict[str, np.ndarray]] = {}
        bytes_down = 0.0

        uses_namespaces = any(
            has_namespaced_params(params) for params in client_params.values()
        )

        global_shared: dict[str, np.ndarray] = {}
        if uses_namespaces and self.config.global_shared_aggregation:
            shared_payloads = []
            for params in client_params.values():
                shared, _, other = split_param_namespaces(params)
                if shared:
                    shared_payloads.append(shared)
            if shared_payloads:
                global_shared = self._aggregator.aggregate(shared_payloads)

        for concept_id, client_ids in concept_groups.items():
            group_params = [
                client_params[cid] for cid in client_ids
                if cid in client_params
            ]
            if not group_params:
                continue

            if uses_namespaces:
                local_shared = global_shared
                if not self.config.global_shared_aggregation:
                    shared_payloads = []
                    for params in group_params:
                        shared, _, _ = split_param_namespaces(params)
                        if shared:
                            shared_payloads.append(shared)
                    local_shared = (
                        self._aggregator.aggregate(shared_payloads)
                        if shared_payloads else {}
                    )
                local_payloads = []
                for params in group_params:
                    _, expert, other = split_param_namespaces(params)
                    local_payloads.append(
                        merge_param_namespaces(
                            shared=None,
                            expert=_remap_expert_payload(expert, concept_id),
                            other=other,
                        )
                    )
                local_agg = self._aggregator.aggregate(local_payloads)
                agg = merge_param_namespaces(
                    shared=local_shared if local_shared else None,
                    expert=local_agg,
                    other=None,
                )
            else:
                agg = self._aggregator.aggregate(group_params)
            aggregated[concept_id] = agg

            # Download cost: each client in this cluster gets the aggregated model
            agg_bytes = model_bytes(agg)
            bytes_down += len(client_ids) * agg_bytes

        # Store aggregated models in memory bank for concept-model warm-start
        for concept_id, agg_params in aggregated.items():
            self.memory_bank.store_model_params(concept_id, agg_params)

        return PhaseBResult(
            aggregated_params=aggregated,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
        )

    def phase_b_soft(
        self,
        client_params: dict[int, dict[str, np.ndarray]],
        concept_assignments: dict[int, int],
        posteriors: dict[int, PosteriorAssignment],
    ) -> PhaseBResult:
        """Execute Phase B with posterior-weighted (soft) aggregation.

        Instead of hard MAP assignment (each client contributes 100% to one
        concept), each client's model is distributed across concept clusters
        weighted by its posterior probability. This reduces discrete noise
        when posteriors are ambiguous (MAP < 0.8 in ~80% of cases).

        Parameters
        ----------
        client_params : dict[int, dict[str, np.ndarray]]
            Client ID -> model parameters.
        concept_assignments : dict[int, int]
            Client ID -> MAP concept ID (used for download routing).
        posteriors : dict[int, PosteriorAssignment]
            Client ID -> full posterior distribution over concepts.

        Returns
        -------
        PhaseBResult
        """
        # Upload cost: each client sends its model (same as hard)
        bytes_up = sum(
            model_bytes(params) for params in client_params.values()
        )

        # Collect all concept IDs present in assignments
        concept_ids = list(set(concept_assignments.values()))

        aggregated: dict[int, dict[str, np.ndarray]] = {}
        bytes_down = 0.0
        uses_namespaces = any(
            has_namespaced_params(params) for params in client_params.values()
        )

        if uses_namespaces and self.config.global_shared_aggregation:
            namespaced_aggregator = NamespacedExpertAggregator()
            ordered_client_ids = list(client_params.keys())
            ordered_payloads = [client_params[client_id] for client_id in ordered_client_ids]
            expert_weights: list[dict[int, float]] = []
            for client_id in ordered_client_ids:
                if client_id in posteriors:
                    weights = {
                        int(slot_id): float(weight)
                        for slot_id, weight in posteriors[client_id].probabilities.items()
                        if float(weight) > 0.01
                    }
                else:
                    assigned = concept_assignments.get(client_id)
                    weights = {int(assigned): 1.0} if assigned is not None else {}
                expert_weights.append(weights)

            namespaced_result = namespaced_aggregator.aggregate_namespaced(
                ordered_payloads,
                expert_weights=expert_weights,
            )
            global_shared = namespaced_result.shared_params

            for concept_id in concept_ids:
                expert_payload = namespaced_result.expert_params.get(concept_id)
                if not expert_payload:
                    stored = self.memory_bank.get_model_params(concept_id)
                    if stored is not None:
                        _, stored_expert, _ = split_param_namespaces(stored)
                        expert_payload = stored_expert

                if not global_shared and not expert_payload:
                    continue

                agg = merge_param_namespaces(
                    shared=global_shared if global_shared else None,
                    expert=expert_payload if expert_payload else None,
                    other=None,
                )
                aggregated[concept_id] = agg

                # Download cost: clients assigned to this concept get the model
                n_recipients = sum(
                    1 for cid, c in concept_assignments.items()
                    if c == concept_id
                )
                bytes_down += n_recipients * model_bytes(agg)
        elif uses_namespaces:
            for concept_id in concept_ids:
                weights: dict[int, float] = {}
                for client_id in client_params:
                    if client_id in posteriors:
                        w = posteriors[client_id].probabilities.get(concept_id, 0.0)
                    else:
                        w = 1.0 if concept_assignments.get(client_id) == concept_id else 0.0
                    if w > 0.01:
                        weights[client_id] = w

                if not weights:
                    continue

                total_w = sum(weights.values())
                group_client_ids = list(weights.keys())
                group_weights = [
                    float(weights[client_id]) / total_w for client_id in group_client_ids
                ]

                shared_payloads: list[dict[str, np.ndarray]] = []
                shared_weights: list[float] = []
                expert_payloads: list[dict[str, np.ndarray]] = []
                expert_group_weights: list[float] = []
                for idx, client_id in enumerate(group_client_ids):
                    params = client_params[client_id]
                    shared, expert, _ = split_param_namespaces(params)
                    if shared:
                        shared_payloads.append(shared)
                        shared_weights.append(group_weights[idx])
                    slot_expert = {
                        key: value.copy()
                        for key, value in expert.items()
                        if key.startswith(f"expert.{concept_id}.")
                    }
                    if slot_expert:
                        expert_payloads.append(slot_expert)
                        expert_group_weights.append(group_weights[idx])

                local_shared = (
                    self._aggregator.aggregate(shared_payloads, shared_weights)
                    if shared_payloads else {}
                )
                local_expert = (
                    self._aggregator.aggregate(expert_payloads, expert_group_weights)
                    if expert_payloads else {}
                )
                if not local_shared and not local_expert:
                    stored = self.memory_bank.get_model_params(concept_id)
                    if stored is not None:
                        shared, expert, _ = split_param_namespaces(stored)
                        local_shared = shared
                        local_expert = {
                            key: value.copy()
                            for key, value in expert.items()
                            if key.startswith(f"expert.{concept_id}.")
                        }
                if not local_shared and not local_expert:
                    continue

                agg = merge_param_namespaces(
                    shared=local_shared if local_shared else None,
                    expert=local_expert if local_expert else None,
                    other=None,
                )
                aggregated[concept_id] = agg

                n_recipients = sum(
                    1 for cid, c in concept_assignments.items()
                    if c == concept_id
                )
                bytes_down += n_recipients * model_bytes(agg)
        else:
            for concept_id in concept_ids:
                # Gather posterior weights from all clients for this concept
                weights: dict[int, float] = {}
                for client_id in client_params:
                    if client_id in posteriors:
                        w = posteriors[client_id].probabilities.get(concept_id, 0.0)
                    else:
                        # Fallback: hard assignment
                        w = 1.0 if concept_assignments.get(client_id) == concept_id else 0.0
                    if w > 0.01:  # skip negligible contributions
                        weights[client_id] = w

                if not weights:
                    continue

                total_w = sum(weights.values())
                group_payloads = [client_params[client_id] for client_id in weights]
                group_weights = [float(weight) / total_w for weight in weights.values()]
                agg = self._aggregator.aggregate(group_payloads, group_weights)

                aggregated[concept_id] = agg

                # Download cost: clients assigned to this concept get the model
                n_recipients = sum(
                    1 for cid, c in concept_assignments.items()
                    if c == concept_id
                )
                bytes_down += n_recipients * model_bytes(agg)

        # Store aggregated models in memory bank
        for concept_id, agg_params in aggregated.items():
            self.memory_bank.store_model_params(concept_id, agg_params)

        return PhaseBResult(
            aggregated_params=aggregated,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
        )


def _avg_assignment_entropy(
    posteriors: dict[int, PosteriorAssignment],
) -> float | None:
    if not posteriors:
        return None
    return float(np.mean([assignment.entropy for assignment in posteriors.values()]))


def _remap_expert_payload(
    expert_payload: dict[str, np.ndarray],
    target_concept_id: int,
) -> dict[str, np.ndarray]:
    """Rename a client's expert payload so it contributes to one slot.

    Local models export their active expert under the slot they trained on.
    During Phase B we need every contributor to aggregate into the target
    concept's expert namespace for that round.
    """
    remapped: dict[str, np.ndarray] = {}
    for key, value in expert_payload.items():
        if not key.startswith("expert."):
            remapped[key] = value.copy()
            continue
        parts = key.split(".", 2)
        if len(parts) < 3:
            remapped[key] = value.copy()
            continue
        remapped[f"expert.{target_concept_id}.{parts[2]}"] = value.copy()
    return remapped


def _assignment_switch_rate(
    assignments: dict[int, int],
    prev_assignments: dict[int, int] | None,
) -> float | None:
    if not assignments or not prev_assignments:
        return None
    switches = 0
    total = 0
    for client_id, concept_id in assignments.items():
        if client_id not in prev_assignments:
            continue
        switches += int(prev_assignments[client_id] != concept_id)
        total += 1
    if total == 0:
        return None
    return float(switches / total)


def _avg_clients_per_concept(assignments: dict[int, int]) -> float | None:
    if not assignments:
        return None
    counts: dict[int, int] = {}
    for concept_id in assignments.values():
        counts[concept_id] = counts.get(concept_id, 0) + 1
    return float(np.mean(list(counts.values()))) if counts else None


def _singleton_group_ratio(assignments: dict[int, int]) -> float | None:
    if not assignments:
        return None
    counts: dict[int, int] = {}
    for concept_id in assignments.values():
        counts[concept_id] = counts.get(concept_id, 0) + 1
    if not counts:
        return None
    singletons = sum(1 for count in counts.values() if count == 1)
    return float(singletons / len(counts))


def _routing_consistency(
    assignments: dict[int, int],
    prev_assignments: dict[int, int] | None,
) -> float | None:
    switch_rate = _assignment_switch_rate(assignments, prev_assignments)
    if switch_rate is None:
        return None
    return float(1.0 - switch_rate)

