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
    """

    omega: float = 10.0
    kappa: float = 0.6
    novelty_threshold: float = 0.3
    loss_novelty_threshold: float = 0.1
    sticky_dampening: float = 1.5
    sticky_posterior_gate: float = 0.3
    model_loss_weight: float = 0.3
    post_spawn_merge: bool = True
    merge_threshold: float = 0.90
    min_count: float = 5.0
    max_concepts: int = 20
    merge_every: int = 2
    shrink_every: int = 5
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
        fp_bytes = fingerprint_bytes(self.config.n_features, self.config.n_classes)
        bytes_up = K * fp_bytes

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

            assignment = self.gibbs.compute_posterior(fp, library, prev_cid)
            posteriors[client_id] = assignment

            # Check novelty via posterior probability threshold
            is_novel = assignment.is_novel

            # Additional loss-based novelty check (critical for single-concept
            # escape): even if posterior is high (e.g. 1.0 with 1 concept),
            # a high loss means the concept is a poor fit.
            best_concept_fp = library[assignment.map_concept_id]
            fp_loss = self.gibbs.compute_loss(fp, best_concept_fp)

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

            if fp_loss > effective_threshold and not model_suppresses:
                is_novel = True

            if is_novel:
                novel_clients.append(client_id)
            else:
                assigned_clients[client_id] = assignment.map_concept_id

        # --- Pass 2: cluster novel clients by fingerprint similarity ---
        if novel_clients:
            novel_clusters = self._cluster_novel_clients(
                novel_clients, client_fingerprints,
            )

            # Spawn one new concept per cluster of truly novel clients
            for cluster in novel_clusters:
                # Build a merged fingerprint for the cluster
                representative_fp = client_fingerprints[cluster[0]]
                spawn_result = self.memory_bank.spawn_from_fingerprint(
                    representative_fp
                )
                new_cid = spawn_result.new_concept_id
                for client_id in cluster:
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
                self.memory_bank.maybe_merge()
                # Remap any assignments whose concept was merged away
                self._remap_merged_assignments(assignments)

        # --- Pass 3: absorb assigned clients into their MAP concepts ---
        for client_id, concept_id in assigned_clients.items():
            assignments[client_id] = concept_id
            self.memory_bank.absorb_fingerprint(
                concept_id, client_fingerprints[client_id]
            )

        # Run memory bank maintenance
        self.memory_bank.step()
        self._round += 1

        # Download cost: 4 bytes per client (concept ID as int32)
        bytes_down = K * 4.0

        return PhaseAResult(
            assignments=assignments,
            posteriors=posteriors,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
        )

    def _remap_merged_assignments(
        self, assignments: dict[int, int],
    ) -> None:
        """Remap client assignments after a merge pass.

        If a concept was merged away (no longer in the memory bank),
        reassign clients to the surviving concept that absorbed it.

        Parameters
        ----------
        assignments : dict[int, int]
            Client ID -> concept ID (mutated in place).
        """
        live_ids = set(self.memory_bank._library.keys())
        stale_clients = [
            cid for cid, concept_id in assignments.items()
            if concept_id not in live_ids
        ]
        if not stale_clients:
            return

        # For each stale assignment, find the best surviving concept
        for client_id in stale_clients:
            best_id = None
            best_sim = -1.0
            for concept_id, cfp in self.memory_bank._library.items():
                # Use the library directly (no copy needed for read)
                sim = cfp.count  # prefer higher-count concepts as merge targets
                if sim > best_sim:
                    best_sim = sim
                    best_id = concept_id
            if best_id is not None:
                assignments[client_id] = best_id

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

        # Single-linkage greedy clustering
        clusters: list[list[int]] = [[novel_client_ids[0]]]
        for cid in novel_client_ids[1:]:
            fp = client_fingerprints[cid]
            merged = False
            for cluster in clusters:
                rep_fp = client_fingerprints[cluster[0]]
                sim = fp.similarity(rep_fp)
                if sim >= sim_threshold:
                    cluster.append(cid)
                    merged = True
                    break
            if not merged:
                clusters.append([cid])

        return clusters

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

        for concept_id, client_ids in concept_groups.items():
            group_params = [
                client_params[cid] for cid in client_ids
                if cid in client_params
            ]
            if not group_params:
                continue

            agg = self._aggregator.aggregate(group_params)
            aggregated[concept_id] = agg

            # Download cost: each client in this cluster gets the aggregated model
            agg_bytes = model_bytes(agg)
            bytes_down += len(client_ids) * agg_bytes

        return PhaseBResult(
            aggregated_params=aggregated,
            bytes_up=bytes_up,
            bytes_down=bytes_down,
        )

    def _cluster_novel_clients(
        self,
        novel_clients: list[tuple[int, ConceptFingerprint]],
    ) -> list[list[tuple[int, ConceptFingerprint]]]:
        """Group novel clients into clusters by fingerprint similarity.

        Uses single-linkage clustering: novel clients whose pairwise
        similarity exceeds ``merge_threshold`` are placed in the same
        cluster, spawning one concept per cluster instead of one per
        client.

        Parameters
        ----------
        novel_clients : list of (client_id, ConceptFingerprint)

        Returns
        -------
        list of clusters, each a list of (client_id, ConceptFingerprint)
        """
        n = len(novel_clients)
        if n <= 1:
            return [novel_clients]

        # Union-Find for clustering
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        threshold = 1.0 - self.config.loss_novelty_threshold
        for i in range(n):
            for j in range(i + 1, n):
                sim = novel_clients[i][1].similarity(novel_clients[j][1])
                if sim >= threshold:
                    union(i, j)

        clusters: dict[int, list[tuple[int, ConceptFingerprint]]] = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(novel_clients[i])

        return list(clusters.values())
