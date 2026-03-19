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
from ..federation.aggregator import FedAvgAggregator
from .gibbs import GibbsPosterior, PosteriorAssignment, TransitionPrior
from .memory_bank import DynamicMemoryBank, MemoryBankConfig


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
    min_count : float
        Minimum fingerprint count for concept survival.
    max_concepts : int
        Hard cap on the number of concepts.
    merge_every : int
        Run merge maintenance every this many federation rounds.
    shrink_every : int
        Run shrink maintenance every this many federation rounds.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    model_signature_weight : float
        Weight for mixing lightweight model-signature similarity into
        Phase A routing. 0.0 keeps the original fingerprint-only routing.
    model_signature_dim : int
        Dimensionality of the lightweight model signature uploaded in
        Phase A when ``model_signature_weight > 0``.
    prototype_ot_weight : float
        Weight for mixing an OT-style prototype similarity derived from
        class-conditional fingerprint means into Phase A routing.
    prototype_ot_reg : float
        Entropic regularization used by the lightweight Sinkhorn solver
        for prototype OT.
    prototype_ot_iters : int
        Number of Sinkhorn iterations for prototype OT.
    update_ot_weight : float
        Weight for mixing a lightweight OT similarity over projected
        classifier rows into Phase A routing.
    update_ot_dim : int
        Projection dimension for each classifier row in update OT.
    labelwise_proto_weight : float
        Weight for mixing a label-wise data-to-model routing term that
        matches client batch prototypes against concept classifier rows.
    labelwise_proto_dim : int
        Projection dimension for each label-wise batch prototype.
    prototype_alignment_mix : float
        After Phase B aggregation, blend each class row toward the concept's
        class-prototype direction. This adds a lightweight prototype-aware
        sharing bias without changing the wire format.
    prototype_alignment_early_rounds : int
        Number of earliest federation rounds that should use the stronger
        ``prototype_alignment_early_mix`` instead of the steady-state mix.
    prototype_alignment_early_mix : float
        Prototype-alignment mix used during the first
        ``prototype_alignment_early_rounds`` federation rounds.
    prototype_prealign_early_rounds : int
        Number of earliest federation rounds that should pre-align each
        client model toward the concept prototypes before aggregation.
    prototype_prealign_early_mix : float
        Pre-alignment mix applied to each client model during the first
        ``prototype_prealign_early_rounds`` federation rounds.
    prototype_subgroup_early_rounds : int
        Number of earliest federation rounds that should first split a
        concept group into up to two predictive subgroups and align each
        subgroup toward its own aggregated prototypes before concept-level
        aggregation.
    prototype_subgroup_early_mix : float
        Prototype-alignment mix used for subgroup-aware pre-alignment
        during the first ``prototype_subgroup_early_rounds`` federation
        rounds.
    prototype_subgroup_min_clients : int
        Minimum number of clients assigned to the same concept before the
        subgroup-aware pre-alignment path is attempted.
    prototype_subgroup_similarity_gate : float
        Minimum farthest-pair dissimilarity required before a concept group
        is split into predictive subgroups. Higher values make subgroup
        pre-alignment more conservative.
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
    min_count: float = 5.0
    max_concepts: int = 20
    merge_every: int = 2
    shrink_every: int = 5
    n_features: int = 2
    n_classes: int = 2
    model_signature_weight: float = 0.0
    model_signature_dim: int = 8
    prototype_ot_weight: float = 0.0
    prototype_ot_reg: float = 0.1
    prototype_ot_iters: int = 30
    update_ot_weight: float = 0.0
    update_ot_dim: int = 4
    labelwise_proto_weight: float = 0.0
    labelwise_proto_dim: int = 4
    prototype_alignment_mix: float = 0.0
    prototype_alignment_early_rounds: int = 0
    prototype_alignment_early_mix: float = 0.0
    prototype_prealign_early_rounds: int = 0
    prototype_prealign_early_mix: float = 0.0
    prototype_subgroup_early_rounds: int = 0
    prototype_subgroup_early_mix: float = 0.0
    prototype_subgroup_min_clients: int = 3
    prototype_subgroup_similarity_gate: float = 0.8


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
            min_count=config.min_count,
            merge_every=config.merge_every,
            shrink_every=config.shrink_every,
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

    def phase_a(
        self,
        client_fingerprints: dict[int, ConceptFingerprint],
        prev_assignments: dict[int, int] | None = None,
        client_model_losses: dict[int, float] | None = None,
        client_model_signatures: dict[int, np.ndarray] | None = None,
        client_update_signatures: dict[int, np.ndarray] | None = None,
        client_batch_prototype_signatures: dict[int, np.ndarray] | None = None,
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
        fp_bytes = fingerprint_bytes(
            self.config.n_features, self.config.n_classes,
            precision_bits=16, include_global_mean=False,
        )
        bytes_up = K * fp_bytes
        if (
            client_model_signatures is not None
            and self.config.model_signature_weight > 0.0
        ):
            bytes_up += K * self.config.model_signature_dim * 2.0
        if (
            client_update_signatures is not None
            and self.config.update_ot_weight > 0.0
        ):
            bytes_up += K * self.config.n_classes * self.config.update_ot_dim * 2.0
        if (
            client_batch_prototype_signatures is not None
            and self.config.labelwise_proto_weight > 0.0
        ):
            bytes_up += K * self.config.n_classes * self.config.labelwise_proto_dim * 2.0

        assignments: dict[int, int] = {}
        posteriors: dict[int, PosteriorAssignment] = {}

        if K == 0:
            return PhaseAResult(
                assignments={}, posteriors={},
                bytes_up=0.0, bytes_down=0.0,
            )

        # --- Bootstrap: if no concepts yet, create the first from all clients ---
        if self.memory_bank.n_concepts == 0:
            # Spawn first concept from the first client
            first_cid = next(iter(client_fingerprints))
            result = self.memory_bank.spawn_from_fingerprint(
                client_fingerprints[first_cid]
            )
            first_concept_id = result.new_concept_id
            if (
                client_model_signatures is not None
                and first_cid in client_model_signatures
            ):
                self.memory_bank.absorb_signature(
                    first_concept_id,
                    client_model_signatures[first_cid],
                )
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

        for client_id, fp in remaining_fps.items():
            prev_cid = (
                prev_assignments.get(client_id) if prev_assignments else None
            )
            library = self.memory_bank.concept_library
            concept_losses = {
                cid: self._routing_loss(
                    fp,
                    cid,
                    client_model_signatures.get(client_id)
                    if client_model_signatures is not None
                    else None,
                    client_update_signatures.get(client_id)
                    if client_update_signatures is not None
                    else None,
                    client_batch_prototype_signatures.get(client_id)
                    if client_batch_prototype_signatures is not None
                    else None,
                )
                for cid in library
            }
            assignment = self.gibbs.compute_posterior_from_losses(
                concept_losses,
                prev_concept_id=prev_cid,
            )
            posteriors[client_id] = assignment

            # Check novelty via posterior probability threshold
            is_novel = assignment.is_novel

            # Additional loss-based novelty check (critical for single-concept
            # escape): even if posterior is high (e.g. 1.0 with 1 concept),
            # a high loss means the concept is a poor fit.
            routed_loss = concept_losses[assignment.map_concept_id]

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

            if routed_loss > effective_threshold and not model_suppresses:
                is_novel = True

            if is_novel:
                novel_clients.append(client_id)
            else:
                assigned_clients[client_id] = assignment.map_concept_id

        # --- Pass 2: cluster novel clients by fingerprint similarity ---
        n_spawned = 0
        n_merged = 0
        n_pruned = 0

        if novel_clients:
            novel_clusters = self._cluster_novel_clients(
                novel_clients,
                client_fingerprints,
                client_model_signatures=client_model_signatures,
                client_update_signatures=client_update_signatures,
                client_batch_prototype_signatures=client_batch_prototype_signatures,
            )

            # Spawn one new concept per cluster of truly novel clients
            for cluster in novel_clusters:
                representative_id = self._select_cluster_representative(
                    cluster,
                    client_fingerprints,
                    client_model_signatures=client_model_signatures,
                    client_update_signatures=client_update_signatures,
                    client_batch_prototype_signatures=client_batch_prototype_signatures,
                )
                representative_fp = client_fingerprints[representative_id]
                spawn_result = self.memory_bank.spawn_from_fingerprint(
                    representative_fp
                )
                if not spawn_result.absorbed:
                    n_spawned += 1
                new_cid = spawn_result.new_concept_id
                for client_id in cluster:
                    assignments[client_id] = new_cid
                    if client_id != representative_id:
                        self.memory_bank.absorb_fingerprint(
                            new_cid, client_fingerprints[client_id]
                        )
                    if (
                        client_model_signatures is not None
                        and client_id in client_model_signatures
                    ):
                        self.memory_bank.absorb_signature(
                            new_cid,
                            client_model_signatures[client_id],
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
            if (
                client_model_signatures is not None
                and client_id in client_model_signatures
            ):
                self.memory_bank.absorb_signature(
                    concept_id,
                    client_model_signatures[client_id],
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

        return PhaseAResult(
            assignments=assignments,
            posteriors=posteriors,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
            spawned=n_spawned,
            merged=n_merged,
            pruned=n_pruned,
        )

    def _routing_loss(
        self,
        observation_fp: ConceptFingerprint,
        concept_id: int,
        client_signature: np.ndarray | None,
        client_update_signature: np.ndarray | None,
        client_batch_prototype_signature: np.ndarray | None,
    ) -> float:
        """Compute the hybrid Phase A routing loss for one concept."""
        concept_fp = self.memory_bank.get_fingerprint(concept_id)
        if concept_fp is None:
            raise KeyError(f"Concept {concept_id} not in memory bank")
        fp_loss = self.gibbs.compute_loss(observation_fp, concept_fp)
        weighted_terms: list[tuple[float, float]] = []
        consumed_weight = 0.0

        sig_weight = self.config.model_signature_weight
        if sig_weight > 0.0 and client_signature is not None:
            concept_signature = self.memory_bank.get_signature(concept_id)
            if concept_signature is not None:
                sig_sim = _vector_similarity(client_signature, concept_signature)
                weighted_terms.append((sig_weight, 1.0 - sig_sim))
                consumed_weight += sig_weight

        ot_weight = self.config.prototype_ot_weight
        if ot_weight > 0.0:
            proto_sim = _prototype_ot_similarity(
                observation_fp,
                concept_fp,
                reg=self.config.prototype_ot_reg,
                n_iters=self.config.prototype_ot_iters,
            )
            weighted_terms.append((ot_weight, 1.0 - proto_sim))
            consumed_weight += ot_weight

        update_ot_weight = self.config.update_ot_weight
        if update_ot_weight > 0.0 and client_update_signature is not None:
            concept_params = self.memory_bank.get_model_params(concept_id)
            if concept_params is not None:
                concept_update_signature = _project_classifier_row_signatures(
                    concept_params,
                    n_features=self.config.n_features,
                    n_classes=self.config.n_classes,
                    output_dim=self.config.update_ot_dim,
                    seed=0,
                )
                update_sim = _signature_transport_similarity(
                    client_update_signature,
                    concept_update_signature,
                    observation_fp.label_distribution,
                    concept_fp.label_distribution,
                    reg=0.1,
                    n_iters=30,
                )
                weighted_terms.append((update_ot_weight, 1.0 - update_sim))
                consumed_weight += update_ot_weight

        labelwise_weight = self.config.labelwise_proto_weight
        if labelwise_weight > 0.0 and client_batch_prototype_signature is not None:
            concept_params = self.memory_bank.get_model_params(concept_id)
            if concept_params is not None:
                concept_row_signature = _project_classifier_row_signatures(
                    concept_params,
                    n_features=self.config.n_features,
                    n_classes=self.config.n_classes,
                    output_dim=self.config.labelwise_proto_dim,
                    seed=0,
                )
                labelwise_sim = _signature_transport_similarity(
                    client_batch_prototype_signature,
                    concept_row_signature,
                    observation_fp.label_distribution,
                    concept_fp.label_distribution,
                    reg=0.1,
                    n_iters=30,
                )
                weighted_terms.append((labelwise_weight, 1.0 - labelwise_sim))
                consumed_weight += labelwise_weight

        fp_weight = max(0.0, 1.0 - consumed_weight)
        weighted_terms.append((fp_weight if fp_weight > 0.0 else 1.0, fp_loss))

        total_weight = sum(weight for weight, _ in weighted_terms)
        return float(
            sum(weight * loss for weight, loss in weighted_terms) / max(total_weight, 1e-12)
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
        client_model_signatures: dict[int, np.ndarray] | None = None,
        client_update_signatures: dict[int, np.ndarray] | None = None,
        client_batch_prototype_signatures: dict[int, np.ndarray] | None = None,
    ) -> list[list[int]]:
        """Cluster novel clients by hybrid similarity (average linkage).

        Uses the same predictive signals already active in Phase A routing
        when grouping clients that were flagged as novel. This keeps spawn
        clustering consistent with the routing objective instead of falling
        back to fingerprint-only similarity.

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
        client_model_signatures : dict[int, np.ndarray] or None
            Optional lightweight model signatures for hybrid routing.
        client_update_signatures : dict[int, np.ndarray] or None
            Optional projected per-class update signatures for update-OT
            routing.

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
            best_cluster_idx = -1
            best_avg_sim = -1.0
            for idx, cluster in enumerate(clusters):
                avg_sim = float(np.mean([
                    self._novel_pair_similarity(
                        cid,
                        member,
                        client_fingerprints,
                        client_model_signatures=client_model_signatures,
                        client_update_signatures=client_update_signatures,
                        client_batch_prototype_signatures=client_batch_prototype_signatures,
                    )
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

    def _select_cluster_representative(
        self,
        cluster: list[int],
        client_fingerprints: dict[int, ConceptFingerprint],
        client_model_signatures: dict[int, np.ndarray] | None = None,
        client_update_signatures: dict[int, np.ndarray] | None = None,
        client_batch_prototype_signatures: dict[int, np.ndarray] | None = None,
    ) -> int:
        """Select the most central client inside a novel cluster."""
        if len(cluster) <= 1:
            return cluster[0]

        best_client = cluster[0]
        best_score = -np.inf
        for cid in cluster:
            sims = [
                self._novel_pair_similarity(
                    cid,
                    other,
                    client_fingerprints,
                    client_model_signatures=client_model_signatures,
                    client_update_signatures=client_update_signatures,
                    client_batch_prototype_signatures=client_batch_prototype_signatures,
                )
                for other in cluster
                if other != cid
            ]
            score = float(np.mean(sims)) if sims else 0.0
            if score > best_score:
                best_score = score
                best_client = cid
        return best_client

    def _novel_pair_similarity(
        self,
        client_a: int,
        client_b: int,
        client_fingerprints: dict[int, ConceptFingerprint],
        *,
        client_model_signatures: dict[int, np.ndarray] | None = None,
        client_update_signatures: dict[int, np.ndarray] | None = None,
        client_batch_prototype_signatures: dict[int, np.ndarray] | None = None,
    ) -> float:
        """Compute pairwise hybrid similarity for novel-client grouping."""
        fp_a = client_fingerprints[client_a]
        fp_b = client_fingerprints[client_b]
        weighted_terms: list[tuple[float, float]] = []
        consumed_weight = 0.0

        sig_weight = self.config.model_signature_weight
        if (
            sig_weight > 0.0
            and client_model_signatures is not None
            and client_a in client_model_signatures
            and client_b in client_model_signatures
        ):
            sig_sim = _vector_similarity(
                client_model_signatures[client_a],
                client_model_signatures[client_b],
            )
            weighted_terms.append((sig_weight, sig_sim))
            consumed_weight += sig_weight

        ot_weight = self.config.prototype_ot_weight
        if ot_weight > 0.0:
            proto_sim = _prototype_ot_similarity(
                fp_a,
                fp_b,
                reg=self.config.prototype_ot_reg,
                n_iters=self.config.prototype_ot_iters,
            )
            weighted_terms.append((ot_weight, proto_sim))
            consumed_weight += ot_weight

        update_ot_weight = self.config.update_ot_weight
        if (
            update_ot_weight > 0.0
            and client_update_signatures is not None
            and client_a in client_update_signatures
            and client_b in client_update_signatures
        ):
            update_sim = _signature_transport_similarity(
                client_update_signatures[client_a],
                client_update_signatures[client_b],
                fp_a.label_distribution,
                fp_b.label_distribution,
                reg=0.1,
                n_iters=30,
            )
            weighted_terms.append((update_ot_weight, update_sim))
            consumed_weight += update_ot_weight

        labelwise_weight = self.config.labelwise_proto_weight
        if (
            labelwise_weight > 0.0
            and client_batch_prototype_signatures is not None
            and client_a in client_batch_prototype_signatures
            and client_b in client_batch_prototype_signatures
        ):
            labelwise_sim = _signature_transport_similarity(
                client_batch_prototype_signatures[client_a],
                client_batch_prototype_signatures[client_b],
                fp_a.label_distribution,
                fp_b.label_distribution,
                reg=0.1,
                n_iters=30,
            )
            weighted_terms.append((labelwise_weight, labelwise_sim))
            consumed_weight += labelwise_weight

        fp_weight = max(0.0, 1.0 - consumed_weight)
        fp_sim = fp_a.similarity(fp_b)
        weighted_terms.append((fp_weight if fp_weight > 0.0 else 1.0, fp_sim))

        total_weight = sum(weight for weight, _ in weighted_terms)
        return float(
            sum(weight * sim for weight, sim in weighted_terms) / max(total_weight, 1e-12)
        )

    def _resolve_alignment_mix(
        self,
        *,
        base_mix: float,
        early_rounds: int,
        early_mix: float,
    ) -> float:
        """Resolve steady-state vs early-round prototype mix."""
        mix = base_mix
        if early_rounds > 0 and self._round <= early_rounds:
            mix = early_mix
        return mix

    def _align_params_to_concept_prototypes(
        self,
        concept_id: int,
        params: dict[str, np.ndarray],
        *,
        mix_override: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Blend aggregated classifier rows toward concept prototype directions."""
        mix = mix_override
        if mix is None:
            mix = self._resolve_alignment_mix(
                base_mix=self.config.prototype_alignment_mix,
                early_rounds=self.config.prototype_alignment_early_rounds,
                early_mix=self.config.prototype_alignment_early_mix,
            )
        if mix <= 0.0:
            return params
        if "coef" not in params or "intercept" not in params:
            return params

        concept_fp = self.memory_bank.get_fingerprint(concept_id)
        if concept_fp is None:
            return params

        return self._align_params_to_prototype_stats(
            params,
            concept_fp.class_means,
            concept_fp.label_distribution,
            mix=mix,
        )

    def _align_params_to_prototype_stats(
        self,
        params: dict[str, np.ndarray],
        class_prototypes: np.ndarray,
        class_mass: np.ndarray,
        *,
        mix: float,
    ) -> dict[str, np.ndarray]:
        """Blend classifier rows toward externally provided prototype stats."""
        if mix <= 0.0:
            return params
        if "coef" not in params or "intercept" not in params:
            return params

        rows, bias = _split_linear_classifier_params(
            params,
            n_features=self.config.n_features,
            n_classes=self.config.n_classes,
        )

        for label in range(self.config.n_classes):
            if label >= len(class_mass) or class_mass[label] <= 1e-8:
                continue
            prototype = np.asarray(class_prototypes[label], dtype=np.float64)
            proto_norm = float(np.linalg.norm(prototype))
            if proto_norm <= 1e-12:
                continue
            row = rows[label]
            row_scale = max(float(np.linalg.norm(row)), 1.0)
            aligned = prototype / proto_norm * row_scale
            rows[label] = (1.0 - mix) * row + mix * aligned

        return _merge_linear_classifier_params(
            rows,
            bias,
            n_features=self.config.n_features,
            n_classes=self.config.n_classes,
        )

    def _estimate_prototype_stats_from_fingerprints(
        self,
        fingerprints: list[ConceptFingerprint],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate class prototypes and class mass from a client subgroup."""
        if not fingerprints:
            return None

        class_mass = np.zeros(self.config.n_classes, dtype=np.float64)
        class_prototypes = np.zeros(
            (self.config.n_classes, self.config.n_features),
            dtype=np.float64,
        )
        for fp in fingerprints:
            label_mass = fp.label_distribution * max(float(fp.count), 1.0)
            class_mass += label_mass
            class_prototypes += label_mass[:, None] * fp.class_means

        valid = class_mass > 1e-8
        if not np.any(valid):
            return None
        class_prototypes[valid] /= class_mass[valid, None]
        return class_prototypes, class_mass

    def _split_clients_by_predictive_similarity(
        self,
        client_ids: list[int],
        client_fingerprints: dict[int, ConceptFingerprint],
    ) -> list[list[int]]:
        """Split a concept group into up to two predictive subgroups."""
        if len(client_ids) <= 2:
            return [client_ids]

        sim = np.ones((len(client_ids), len(client_ids)), dtype=np.float64)
        min_sim = np.inf
        seed_i = 0
        seed_j = 1
        for i in range(len(client_ids)):
            for j in range(i + 1, len(client_ids)):
                pair_sim = _prototype_ot_similarity(
                    client_fingerprints[client_ids[i]],
                    client_fingerprints[client_ids[j]],
                    reg=self.config.prototype_ot_reg,
                    n_iters=self.config.prototype_ot_iters,
                )
                sim[i, j] = pair_sim
                sim[j, i] = pair_sim
                if pair_sim < min_sim:
                    min_sim = pair_sim
                    seed_i, seed_j = i, j

        if min_sim >= self.config.prototype_subgroup_similarity_gate:
            return [client_ids]

        group_a = [client_ids[seed_i]]
        group_b = [client_ids[seed_j]]
        for idx, client_id in enumerate(client_ids):
            if idx in {seed_i, seed_j}:
                continue
            if sim[idx, seed_i] >= sim[idx, seed_j]:
                group_a.append(client_id)
            else:
                group_b.append(client_id)

        return [group for group in (group_a, group_b) if group]

    def _subgroup_prealign_params(
        self,
        client_ids: list[int],
        client_params: dict[int, dict[str, np.ndarray]],
        client_fingerprints: dict[int, ConceptFingerprint] | None,
        *,
        mix: float,
    ) -> list[dict[str, np.ndarray]] | None:
        """Pre-align params to subgroup prototypes without changing downloads."""
        if mix <= 0.0 or client_fingerprints is None:
            return None

        eligible_ids = [
            client_id for client_id in client_ids
            if client_id in client_params and client_id in client_fingerprints
        ]
        if len(eligible_ids) < self.config.prototype_subgroup_min_clients:
            return None

        subgroups = self._split_clients_by_predictive_similarity(
            eligible_ids,
            client_fingerprints,
        )
        if len(subgroups) <= 1:
            return None

        subgroup_params: dict[int, dict[str, np.ndarray]] = {}
        for subgroup in subgroups:
            stats = self._estimate_prototype_stats_from_fingerprints(
                [client_fingerprints[client_id] for client_id in subgroup]
            )
            if stats is None:
                for client_id in subgroup:
                    subgroup_params[client_id] = client_params[client_id]
                continue
            class_prototypes, class_mass = stats
            for client_id in subgroup:
                subgroup_params[client_id] = self._align_params_to_prototype_stats(
                    client_params[client_id],
                    class_prototypes,
                    class_mass,
                    mix=mix,
                )

        return [
            subgroup_params.get(client_id, client_params[client_id])
            for client_id in client_ids
            if client_id in client_params
        ]

    def phase_b(
        self,
        client_params: dict[int, dict[str, np.ndarray]],
        concept_assignments: dict[int, int],
        client_fingerprints: dict[int, ConceptFingerprint] | None = None,
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

        for concept_id, client_ids in concept_groups.items():
            group_params = [
                client_params[cid] for cid in client_ids
                if cid in client_params
            ]
            if not group_params:
                continue

            subgroup_mix = self._resolve_alignment_mix(
                base_mix=0.0,
                early_rounds=self.config.prototype_subgroup_early_rounds,
                early_mix=self.config.prototype_subgroup_early_mix,
            )
            subgroup_params = self._subgroup_prealign_params(
                client_ids,
                client_params,
                client_fingerprints,
                mix=subgroup_mix,
            )
            if subgroup_params is not None:
                group_params = subgroup_params

            prealign_mix = self._resolve_alignment_mix(
                base_mix=0.0,
                early_rounds=self.config.prototype_prealign_early_rounds,
                early_mix=self.config.prototype_prealign_early_mix,
            )
            if prealign_mix > 0.0 and subgroup_params is None:
                group_params = [
                    self._align_params_to_concept_prototypes(
                        concept_id,
                        params,
                        mix_override=prealign_mix,
                    )
                    for params in group_params
                ]

            agg = self._aggregator.aggregate(group_params)
            agg = self._align_params_to_concept_prototypes(concept_id, agg)
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
        client_fingerprints: dict[int, ConceptFingerprint] | None = None,
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

            subgroup_mix = self._resolve_alignment_mix(
                base_mix=0.0,
                early_rounds=self.config.prototype_subgroup_early_rounds,
                early_mix=self.config.prototype_subgroup_early_mix,
            )
            weighted_client_ids = list(weights)
            subgroup_params: dict[int, dict[str, np.ndarray]] = {}
            subgroup_param_list = self._subgroup_prealign_params(
                weighted_client_ids,
                client_params,
                client_fingerprints,
                mix=subgroup_mix,
            )
            if subgroup_param_list is not None:
                subgroup_params = {
                    client_id: subgroup_param_list[idx]
                    for idx, client_id in enumerate(
                        [cid for cid in weighted_client_ids if cid in client_params]
                    )
                }

            prealign_mix = self._resolve_alignment_mix(
                base_mix=0.0,
                early_rounds=self.config.prototype_prealign_early_rounds,
                early_mix=self.config.prototype_prealign_early_mix,
            )

            # Weighted average of model parameters
            first_cid = next(iter(weights))
            agg: dict[str, np.ndarray] = {
                key: np.zeros_like(arr)
                for key, arr in client_params[first_cid].items()
            }
            for client_id, w in weights.items():
                params = subgroup_params.get(client_id, client_params[client_id])
                if prealign_mix > 0.0 and not subgroup_params:
                    params = self._align_params_to_concept_prototypes(
                        concept_id,
                        params,
                        mix_override=prealign_mix,
                    )
                for key in agg:
                    agg[key] += (w / total_w) * params[key]

            agg = self._align_params_to_concept_prototypes(concept_id, agg)
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


def _vector_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity mapped to [0, 1] for lightweight routing vectors."""
    vec_a = np.asarray(a, dtype=np.float64).reshape(-1)
    vec_b = np.asarray(b, dtype=np.float64).reshape(-1)
    if vec_a.size == 0 or vec_b.size == 0:
        return 0.0
    if vec_a.size != vec_b.size:
        raise ValueError(
            f"Signature size mismatch: {vec_a.size} vs {vec_b.size}",
        )
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 1e-12:
        return 0.0
    cosine = float(np.dot(vec_a, vec_b) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return 0.5 * (cosine + 1.0)


def _prototype_pair_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Prototype similarity that mixes angle and displacement cues."""
    cosine_sim = _vector_similarity(a, b)
    diff = np.asarray(a, dtype=np.float64).reshape(-1) - np.asarray(b, dtype=np.float64).reshape(-1)
    mse = float(np.mean(diff ** 2)) if diff.size > 0 else 0.0
    mse_sim = float(np.exp(-mse))
    return 0.5 * cosine_sim + 0.5 * mse_sim


def _classifier_rows_from_params(
    params: dict[str, np.ndarray],
    *,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Convert project parameter dicts into per-class classifier rows."""
    coef = np.asarray(params["coef"], dtype=np.float64).reshape(-1)
    intercept = np.asarray(params["intercept"], dtype=np.float64).reshape(-1)
    if n_classes == 2:
        if coef.size == n_features and intercept.size == 1:
            row = coef.reshape(1, n_features)
            bias = intercept.reshape(1, 1)
            rows = np.concatenate(
                [
                    np.concatenate([-row, -bias], axis=1),
                    np.concatenate([row, bias], axis=1),
                ],
                axis=0,
            )
            return rows
        if coef.size == 2 * n_features and intercept.size == 2:
            return np.concatenate(
                [
                    coef.reshape(2, n_features),
                    intercept.reshape(2, 1),
                ],
                axis=1,
            )
        raise ValueError(
            f"Binary classifier params incompatible with n_features={n_features}: "
            f"coef={coef.size}, intercept={intercept.size}",
        )

    expected_coef = n_classes * n_features
    if coef.size != expected_coef or intercept.size != n_classes:
        raise ValueError(
            f"Expected coef/intercept sizes {expected_coef}/{n_classes}, "
            f"got {coef.size}/{intercept.size}",
        )
    return np.concatenate(
        [
            coef.reshape(n_classes, n_features),
            intercept.reshape(n_classes, 1),
        ],
        axis=1,
    )


def _split_linear_classifier_params(
    params: dict[str, np.ndarray],
    *,
    n_features: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split project params into dense per-class rows and biases."""
    rows = _classifier_rows_from_params(
        params,
        n_features=n_features,
        n_classes=n_classes,
    )
    return rows[:, :-1].copy(), rows[:, -1].copy()


def _merge_linear_classifier_params(
    rows: np.ndarray,
    bias: np.ndarray,
    *,
    n_features: int,
    n_classes: int,
) -> dict[str, np.ndarray]:
    """Merge dense per-class rows and biases back into project params."""
    arr_rows = np.asarray(rows, dtype=np.float64)
    arr_bias = np.asarray(bias, dtype=np.float64).reshape(-1)

    if n_classes == 2:
        if arr_rows.shape != (2, n_features) or arr_bias.shape != (2,):
            raise ValueError(
                f"Binary rows/bias must have shapes {(2, n_features)} and (2,), "
                f"got {arr_rows.shape} and {arr_bias.shape}",
            )
        return {
            "coef": arr_rows[1].reshape(-1).copy(),
            "intercept": np.array([arr_bias[1]], dtype=np.float64),
        }

    if arr_rows.shape != (n_classes, n_features) or arr_bias.shape != (n_classes,):
        raise ValueError(
            f"Expected rows/bias shapes {(n_classes, n_features)} and {(n_classes,)}, "
            f"got {arr_rows.shape} and {arr_bias.shape}",
        )
    return {
        "coef": arr_rows.reshape(-1).copy(),
        "intercept": arr_bias.copy(),
    }


def _project_classifier_row_signatures(
    params: dict[str, np.ndarray],
    *,
    n_features: int,
    n_classes: int,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Project each classifier row into a lightweight signature space."""
    if output_dim < 1:
        raise ValueError(f"output_dim must be >= 1, got {output_dim}")
    rows = _classifier_rows_from_params(
        params,
        n_features=n_features,
        n_classes=n_classes,
    )
    row_dim = rows.shape[1]
    rng = np.random.default_rng(seed + 1543 * output_dim + 31 * row_dim + 7 * n_classes)
    projector = rng.standard_normal((row_dim, output_dim)) / np.sqrt(output_dim)
    signatures = rows @ projector
    norms = np.linalg.norm(signatures, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (signatures / norms).astype(np.float64, copy=False)


def _prototype_ot_similarity(
    source_fp: ConceptFingerprint,
    target_fp: ConceptFingerprint,
    *,
    reg: float = 0.1,
    n_iters: int = 30,
    min_mass: float = 1e-8,
) -> float:
    """Compute a lightweight OT similarity between fingerprint prototypes.

    Uses class-conditional means as prototypes and class distributions as
    transport masses. Pairwise prototype costs mix cosine similarity with
    mean-squared displacement, then are converted to an entropic OT distance
    via Sinkhorn iterations.
    """
    p = np.asarray(source_fp.label_distribution, dtype=np.float64)
    q = np.asarray(target_fp.label_distribution, dtype=np.float64)
    src_idx = np.where(p > min_mass)[0]
    tgt_idx = np.where(q > min_mass)[0]
    if src_idx.size == 0 or tgt_idx.size == 0:
        return 0.0

    p = p[src_idx]
    q = q[tgt_idx]
    p = p / max(float(p.sum()), min_mass)
    q = q / max(float(q.sum()), min_mass)
    src_proto = source_fp.class_means[src_idx]
    tgt_proto = target_fp.class_means[tgt_idx]

    cost = np.zeros((src_idx.size, tgt_idx.size), dtype=np.float64)
    for i in range(src_idx.size):
        for j in range(tgt_idx.size):
            cost[i, j] = 1.0 - _prototype_pair_similarity(
                src_proto[i],
                tgt_proto[j],
            )

    transport_cost = _sinkhorn_transport_cost(
        cost,
        p,
        q,
        reg=reg,
        n_iters=n_iters,
    )
    return float(np.clip(1.0 - transport_cost, 0.0, 1.0))


def _signature_transport_similarity(
    source_signatures: np.ndarray,
    target_signatures: np.ndarray,
    source_masses: np.ndarray,
    target_masses: np.ndarray,
    *,
    reg: float = 0.1,
    n_iters: int = 30,
    min_mass: float = 1e-8,
) -> float:
    """Compute OT similarity between two sets of projected row signatures."""
    p = np.asarray(source_masses, dtype=np.float64).reshape(-1)
    q = np.asarray(target_masses, dtype=np.float64).reshape(-1)
    src = np.asarray(source_signatures, dtype=np.float64)
    tgt = np.asarray(target_signatures, dtype=np.float64)
    src_idx = np.where(p > min_mass)[0]
    tgt_idx = np.where(q > min_mass)[0]
    if src_idx.size == 0 or tgt_idx.size == 0:
        return 0.0
    p = p[src_idx]
    q = q[tgt_idx]
    p = p / max(float(p.sum()), min_mass)
    q = q / max(float(q.sum()), min_mass)
    src = src[src_idx]
    tgt = tgt[tgt_idx]
    cost = np.zeros((src.shape[0], tgt.shape[0]), dtype=np.float64)
    for i in range(src.shape[0]):
        for j in range(tgt.shape[0]):
            cost[i, j] = 1.0 - _prototype_pair_similarity(src[i], tgt[j])
    transport_cost = _sinkhorn_transport_cost(cost, p, q, reg=reg, n_iters=n_iters)
    return float(np.clip(1.0 - transport_cost, 0.0, 1.0))


def _sinkhorn_transport_cost(
    cost: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    *,
    reg: float = 0.1,
    n_iters: int = 30,
) -> float:
    """Compute an entropic OT cost with a small Sinkhorn solver."""
    if reg <= 0.0:
        raise ValueError(f"reg must be > 0, got {reg}")
    kernel = np.exp(-np.asarray(cost, dtype=np.float64) / reg)
    kernel = np.maximum(kernel, 1e-12)
    u = np.ones_like(p, dtype=np.float64)
    v = np.ones_like(q, dtype=np.float64)
    for _ in range(max(n_iters, 1)):
        Kv = kernel @ v
        Kv = np.maximum(Kv, 1e-12)
        u = p / Kv
        KTu = kernel.T @ u
        KTu = np.maximum(KTu, 1e-12)
        v = q / KTu
    plan = (u[:, None] * kernel) * v[None, :]
    return float(np.sum(plan * cost))

