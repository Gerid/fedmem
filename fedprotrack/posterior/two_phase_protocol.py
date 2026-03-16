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
        MAP probability threshold below which a concept is novel
        (used by GibbsPosterior).
    loss_novelty_threshold : float
        Loss threshold for novelty detection in Phase A. A client whose
        best-matching concept has loss above this value is spawned as a
        new concept. Lower values are more sensitive to concept differences.
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

    omega: float = 1.0
    kappa: float = 0.8
    novelty_threshold: float = 0.3
    loss_novelty_threshold: float = 0.08
    merge_threshold: float = 0.85
    min_count: float = 5.0
    max_concepts: int = 20
    merge_every: int = 5
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
    ) -> PhaseAResult:
        """Execute Phase A: lightweight concept identification.

        Each client uploads a fingerprint. The server uses the Gibbs
        posterior to assign concept identities and updates the memory bank.

        Parameters
        ----------
        client_fingerprints : dict[int, ConceptFingerprint]
            Client ID -> fingerprint built from current data batch.
        prev_assignments : dict[int, int] or None
            Previous round's concept assignments (client_id -> concept_id).

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
                assignments={}, posteriors={}, bytes_up=0.0, bytes_down=0.0,
            )

        # Bootstrap: if memory bank is empty, seed it from the first client
        if self.memory_bank.n_concepts == 0:
            first_id = next(iter(client_fingerprints))
            first_fp = client_fingerprints[first_id]
            result = self.memory_bank.spawn_from_fingerprint(first_fp)
            assignments[first_id] = result.new_concept_id
            posteriors[first_id] = PosteriorAssignment(
                probabilities={result.new_concept_id: 1.0},
                map_concept_id=result.new_concept_id,
                is_novel=True,
                entropy=0.0,
            )

        # --- Pass 1: compute all posteriors against CURRENT memory bank ---
        pending: list[tuple[int, ConceptFingerprint, PosteriorAssignment]] = []
        for client_id, fp in client_fingerprints.items():
            if client_id in assignments:
                continue  # already assigned during bootstrap
            prev_cid = prev_assignments.get(client_id) if prev_assignments else None
            assignment = self.gibbs.compute_posterior(
                fp, self.memory_bank.concept_library, prev_cid,
            )
            pending.append((client_id, fp, assignment))

        # --- Pass 2: identify novel clients via pairwise comparison ---
        # Clients whose MAP probability is low AND whose fingerprint is
        # dissimilar from the best-matching concept are spawned as new.
        novel_clients: list[tuple[int, ConceptFingerprint]] = []
        absorb_clients: list[tuple[int, ConceptFingerprint, PosteriorAssignment]] = []

        for client_id, fp, assignment in pending:
            best_loss = self.gibbs.compute_loss(
                fp, self.memory_bank.concept_library[assignment.map_concept_id],
            )
            loss_novel = best_loss > self.config.loss_novelty_threshold

            if assignment.is_novel or loss_novel:
                novel_clients.append((client_id, fp))
                posteriors[client_id] = assignment
            else:
                absorb_clients.append((client_id, fp, assignment))
                posteriors[client_id] = assignment

        # --- Pass 3: cluster novel clients, then spawn one concept per cluster ---
        if novel_clients:
            novel_clusters = self._cluster_novel_clients(novel_clients)
            for cluster in novel_clusters:
                # Spawn concept from first client, absorb the rest
                first_cid, first_fp = cluster[0]
                result = self.memory_bank.spawn_from_fingerprint(first_fp)
                new_id = result.new_concept_id
                assignments[first_cid] = new_id
                for client_id, fp in cluster[1:]:
                    self.memory_bank.absorb_fingerprint(new_id, fp)
                    assignments[client_id] = new_id

        # --- Pass 4: assign non-novel clients, then batch-absorb ---
        for client_id, fp, assignment in absorb_clients:
            assignments[client_id] = assignment.map_concept_id

        # Batch absorb after all assignments are decided
        for client_id, fp, assignment in absorb_clients:
            self.memory_bank.absorb_fingerprint(assignment.map_concept_id, fp)

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
