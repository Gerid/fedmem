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

    omega: float = 5.0
    kappa: float = 0.8
    novelty_threshold: float = 0.3
    loss_novelty_threshold: float = 0.1
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
            best_loss = self.gibbs.compute_loss(fp, best_concept_fp)
            if best_loss > self.config.loss_novelty_threshold:
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
            # Spawn one new concept per cluster
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

    def _cluster_novel_clients(
        self,
        novel_client_ids: list[int],
        client_fingerprints: dict[int, ConceptFingerprint],
    ) -> list[list[int]]:
        """Cluster novel clients by fingerprint similarity (single-linkage).

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

        # Similarity threshold for clustering: two novel clients belong
        # to the same new concept if their fingerprints are similar enough.
        sim_threshold = 1.0 - self.config.loss_novelty_threshold

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
