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
    spawn_pressure_damping : float
        When > 0, progressively raises the effective loss novelty threshold
        as the number of active concepts approaches ``max_concepts``.
        The scaling factor is ``1 + damping * (n_concepts / max_concepts)^2``.
        This creates back-pressure against spawning when the concept budget
        is filling up, forcing the system to reuse existing concepts rather
        than creating spurious new ones. Default 0.0 (disabled).
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
    merge_min_support: int = 1
    min_count: float = 5.0
    max_concepts: int = 20
    max_spawn_clusters_per_round: int | None = None
    novelty_hysteresis_rounds: int = 1
    spawn_pressure_damping: float = 0.0
    merge_every: int = 2
    shrink_every: int = 5
    key_mode: str = "legacy_fingerprint"
    key_ema_decay: float = 0.0
    key_style_weight: float = 0.25
    key_semantic_weight: float = 0.30
    key_prototype_weight: float = 0.45
    global_shared_aggregation: bool = False
    entropy_freeze_threshold: float | None = None
    adaptive_addressing: bool = False
    addressing_min_round_interval: int = 1
    addressing_drift_threshold: float = 0.0
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
    min_agg_group_size: int = 1
    cross_time_ema: float = 1.0
    concept_global_mix: float = 0.0
    concept_delta_mode: bool = False
    enable_trust_estimation: bool = False
    trust_buffer_size: int = 5
    trust_decay: float = 0.7
    trust_promotion_threshold: int = 2
    # DRCT (Dual-Rank Concept Tracking) Stein shrinkage
    drct_shrinkage: bool = False
    drct_d_eff_ratio: float = 0.9
    drct_min_concepts: int = 2
    # Warmup window during which λ̂ is forced to 1 (pure FedAvg aggregation).
    # Low-SNR datasets (e.g. CIFAR-10) suffer from noisy early σ_B² estimates
    # that pull λ̂ away from 1; fixing λ̂=1 for the first W rounds lets concept
    # statistics stabilize before empirical-Bayes shrinkage kicks in.
    drct_warmup_rounds: int = 0
    # SNR-gated shrinkage: when between-concept signal is weak relative to
    # within-concept noise, the Stein estimate of σ_B² is unreliable and
    # shrinkage collapses per-concept models into FedAvg. Gate λ by the
    # observed SNR r = σ_B² / (σ²·d_eff/n̄_k); set λ=0 when r < threshold.
    drct_snr_gate: bool = True
    drct_snr_threshold: float = 1.0
    # EMA smoothing of σ_B² / σ² across rounds. Single-round estimates are
    # noisy in low-SNR regimes; EMA stabilises them assuming drift is slow.
    # beta=0 disables smoothing (use instantaneous estimate).
    drct_sigma_ema_beta: float = 0.0
    # Ablation knobs for OT concept discovery. Setting to the ablation value
    # recovers the pre-fix behaviour; defaults preserve the current path.
    # The drct_force_ambient_d_eff knob was removed in the v2 submission: the
    # ablation confirmed it produces bit-identical results in the Scheme C
    # recurrence regime, so keeping it as a configurable knob added code
    # complexity without any behavioural degree of freedom. See the paper's
    # three-fix ablation table for the empirical basis of this removal.
    ot_affinity_scale: str = "local"  # ablation: "median" (fix 1b off)
    ot_eigengap_method: str = "last_significant"  # ablation: "argmax" (fix 1a off)


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
            enable_trust_estimation=config.enable_trust_estimation,
            trust_buffer_size=config.trust_buffer_size,
            trust_decay=config.trust_decay,
            trust_promotion_threshold=config.trust_promotion_threshold,
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
        self._latest_global_fedavg: dict[str, np.ndarray] | None = None
        # Per-round DRCT shrinkage diagnostics (populated by _apply_drct_shrinkage)
        self.drct_lambda_log: list[dict[str, object]] = []
        self._drct_sigma2_ema: float | None = None
        self._drct_sigma_B2_ema: float | None = None

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
        library_size_before = self.memory_bank.n_concepts
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
        client_fp_losses: dict[int, float] = {}
        client_effective_thresholds: dict[int, float] = {}
        client_map_probabilities: dict[int, float] = {}

        for client_id, fp in remaining_fps.items():
            prev_cid = (
                prev_assignments.get(client_id) if prev_assignments else None
            )
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
                for cid in self.memory_bank.concept_library
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
            fp_loss = routed_loss
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
            library = self.memory_bank.concept_library
            n_library = len(library)
            if (
                n_library >= 2
                and prev_cid is not None
                and prev_cid in assignment.probabilities
            ):
                prev_prob = assignment.probabilities[prev_cid]
                if prev_prob >= self.config.sticky_posterior_gate:
                    effective_threshold *= self.config.sticky_dampening

            # --- Spawn pressure damping ---
            # As concept count grows toward max_concepts, raise the
            # effective novelty threshold to create back-pressure against
            # spawning.  This prevents the "6 concepts for 4 true" scenario
            # on real data where fingerprint noise triggers spurious spawns.
            if (
                self.config.spawn_pressure_damping > 0.0
                and self.config.max_concepts > 0
                and n_library >= 2
            ):
                fill_ratio = n_library / self.config.max_concepts
                pressure = 1.0 + self.config.spawn_pressure_damping * fill_ratio ** 2
                effective_threshold *= pressure

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

            if routed_loss > effective_threshold and not model_suppresses:
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
                novel_clients,
                client_fingerprints,
                client_model_signatures=client_model_signatures,
                client_update_signatures=client_update_signatures,
                client_batch_prototype_signatures=client_batch_prototype_signatures,
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
                    self._novelty_streaks[client_id] = 0
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
        if self.config.key_mode == "legacy_fingerprint":
            fp_loss = self.gibbs.compute_loss(observation_fp, concept_fp)
        else:
            routing_entry = self.memory_bank.routing_library.get(concept_id)
            if routing_entry is None:
                raise KeyError(f"Concept {concept_id} has no routing key")
            fp_loss = self.gibbs.compute_loss(observation_fp, routing_entry)
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

    # ------------------------------------------------------------------
    # DRCT Stein shrinkage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_params(params: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten a param dict into a single 1-D vector."""
        return np.concatenate([v.ravel() for v in params.values()])

    def _apply_drct_shrinkage(
        self,
        aggregated: dict[int, dict[str, np.ndarray]],
        global_fedavg: dict[str, np.ndarray],
        concept_groups: dict[int, list[int]],
        client_params: dict[int, dict[str, np.ndarray]],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Apply DRCT Stein shrinkage: blend each concept toward global.

        λ_k = σ² · d_eff / n_k / (σ² · d_eff / n_k + σ_B²)

        Parameters
        ----------
        aggregated : dict[int, dict[str, np.ndarray]]
            Concept ID -> aggregated model parameters.
        global_fedavg : dict[str, np.ndarray]
            Global FedAvg parameters (shrinkage target).
        concept_groups : dict[int, list[int]]
            Concept ID -> list of client IDs.
        client_params : dict[int, dict[str, np.ndarray]]
            Client ID -> model parameters.

        Returns
        -------
        dict[int, dict[str, np.ndarray]]
            Shrunk concept parameters.
        """
        # Build per-concept upload lists
        concept_uploads: dict[int, list[dict[str, np.ndarray]]] = {}
        for cid, cids in concept_groups.items():
            uploads = [client_params[k] for k in cids if k in client_params]
            if uploads:
                concept_uploads[cid] = uploads

        if len(concept_uploads) < self.config.drct_min_concepts:
            return aggregated

        # Warmup: force λ̂=1 (pure FedAvg) for the first W rounds so noisy
        # early σ_B² estimates cannot pull concept models apart before
        # between-concept structure is resolved.
        if self._round <= self.config.drct_warmup_rounds:
            self.drct_lambda_log.append(
                {"round": self._round, "lambda_mean": 1.0, "warmup": True}
            )
            return {
                cid: {k: global_fedavg[k].copy() for k in agg}
                for cid, agg in aggregated.items()
            }

        # Estimate d_eff: effective dimensionality of between-concept structure.
        # Using participation ratio of concept mean SVD singular values:
        #   d_eff = (Σ s_i)² / Σ s_i²
        # This measures how many dimensions the concept means actually differ
        # in, rather than using the (vastly overestimated) ambient dimension.
        flatten = TwoPhaseFedProTrack._flatten_params
        flat_concepts = np.stack([flatten(w) for w in aggregated.values()])
        d = flat_concepts.shape[1]

        if flat_concepts.shape[0] >= 2:
            centered = flat_concepts - flat_concepts.mean(axis=0)
            svs = np.linalg.svd(centered, compute_uv=False)
            svs = svs[svs > 1e-10]
            if len(svs) > 0:
                sum_s = float(np.sum(svs))
                sum_s2 = float(np.sum(svs ** 2))
                d_eff = max(1.0, sum_s ** 2 / sum_s2) if sum_s2 > 0 else 1.0
            else:
                d_eff = 1.0
        else:
            d_eff = d * self.config.drct_d_eff_ratio

        # σ² — within-concept noise variance
        sigma2 = self._estimate_sigma2(concept_uploads, aggregated)

        # σ_B² — between-concept variance (bias-corrected)
        sigma_B2 = self._estimate_sigma_B2(aggregated, sigma2, concept_uploads)

        # EMA smoothing of variance estimates across rounds. Single-round
        # σ_B² is noisy when concepts are not yet well-separated; the EMA
        # assumes true between-concept structure drifts slowly.
        beta = float(self.config.drct_sigma_ema_beta)
        if beta > 0.0:
            if self._drct_sigma2_ema is None:
                self._drct_sigma2_ema = sigma2
                self._drct_sigma_B2_ema = sigma_B2
            else:
                self._drct_sigma2_ema = beta * self._drct_sigma2_ema + (1.0 - beta) * sigma2
                self._drct_sigma_B2_ema = beta * self._drct_sigma_B2_ema + (1.0 - beta) * sigma_B2
            sigma2 = self._drct_sigma2_ema
            sigma_B2 = self._drct_sigma_B2_ema

        # SNR gate: if between-concept signal is dominated by within-concept
        # noise, the Stein estimate is unreliable — fall back to λ=0 (trust
        # clustering) rather than the default λ→1 (collapse to FedAvg).
        if self.config.drct_snr_gate:
            n_bar = np.mean([
                float(len(concept_groups.get(cid, [])))
                for cid in aggregated
            ]) if aggregated else 1.0
            within_term = sigma2 * d_eff / max(n_bar, 1.0)
            snr = sigma_B2 / within_term if within_term > 1e-30 else 0.0
            if snr < float(self.config.drct_snr_threshold):
                self.drct_lambda_log.append({
                    "round": self._round,
                    "n_concepts": len(aggregated),
                    "sigma2": sigma2,
                    "sigma_B2": sigma_B2,
                    "d_eff": d_eff,
                    "snr": snr,
                    "snr_gate": True,
                    "lambda_mean": 0.0,
                })
                return {cid: {k: v.copy() for k, v in agg.items()}
                        for cid, agg in aggregated.items()}

        # Per-concept shrinkage
        result: dict[int, dict[str, np.ndarray]] = {}
        per_concept_lambdas: dict[int, float] = {}
        for cid, agg in aggregated.items():
            n_k = float(len(concept_groups.get(cid, [])))
            if n_k <= 0:
                result[cid] = agg
                per_concept_lambdas[cid] = 0.0
                continue
            # λ = σ²·d_eff / n_k / (σ²·d_eff / n_k + σ_B²)
            variance_term = sigma2 * d_eff / n_k
            denom = variance_term + sigma_B2
            lam = float(np.clip(variance_term / denom, 0.0, 1.0)) if denom > 1e-30 else 0.5
            per_concept_lambdas[cid] = lam
            result[cid] = {
                key: (1.0 - lam) * agg[key] + lam * global_fedavg[key]
                for key in agg
            }

        lam_values = list(per_concept_lambdas.values())
        self.drct_lambda_log.append({
            "round": self._round,
            "n_concepts": len(aggregated),
            "sigma2": sigma2,
            "sigma_B2": sigma_B2,
            "d_eff": d_eff,
            "lambda_mean": float(np.mean(lam_values)) if lam_values else 0.0,
            "lambda_std": float(np.std(lam_values)) if lam_values else 0.0,
            "per_concept": dict(per_concept_lambdas),
        })
        return result

    @staticmethod
    def _estimate_sigma2(
        concept_uploads: dict[int, list[dict[str, np.ndarray]]],
        concept_means: dict[int, dict[str, np.ndarray]],
    ) -> float:
        """Estimate within-concept noise variance from client model dispersion."""
        total_var = 0.0
        total_dim = 0
        total_count = 0
        flatten = TwoPhaseFedProTrack._flatten_params
        for cid, uploads in concept_uploads.items():
            if len(uploads) < 2 or cid not in concept_means:
                continue
            mean_w = flatten(concept_means[cid])
            for w in uploads:
                diff = flatten(w) - mean_w
                total_var += np.sum(diff ** 2)
                total_dim += diff.size
            total_count += len(uploads) - 1
        if total_count == 0 or total_dim == 0:
            return 1e-4
        return float(total_var / total_dim)

    @staticmethod
    def _estimate_sigma_B2(
        concept_weights: dict[int, dict[str, np.ndarray]],
        sigma2: float,
        concept_uploads: dict[int, list[dict[str, np.ndarray]]],
    ) -> float:
        """Estimate between-concept variance from concept-level weight means."""
        if len(concept_weights) < 2:
            return 0.0
        flatten = TwoPhaseFedProTrack._flatten_params
        flat = [flatten(w) for w in concept_weights.values()]
        stacked = np.stack(flat)
        global_mean = stacked.mean(axis=0)
        d = stacked.shape[1]
        C = len(concept_weights)
        spread = np.sum((stacked - global_mean) ** 2) / ((C - 1) * d)
        # Bias correction: subtract estimation noise
        n_concept_avg = np.mean([
            float(len(concept_uploads.get(cid, [])))
            for cid in concept_weights
        ])
        correction = sigma2 / n_concept_avg if n_concept_avg > 0 else 0.0
        return float(max(spread - correction, 0.0))

    def _store_aggregated_models(
        self,
        aggregated: dict[int, dict[str, np.ndarray]],
        global_fedavg: dict[str, np.ndarray] | None,
    ) -> None:
        """Store aggregated models in memory bank.

        Applies cross-time EMA blending, concept-global interpolation,
        and concept-delta storage according to config.

        Parameters
        ----------
        aggregated : dict[int, dict[str, np.ndarray]]
            Concept ID -> aggregated model parameters. Modified in place
            to reflect any blending applied before storage.
        global_fedavg : dict[str, np.ndarray] or None
            Global FedAvg model (average of all clients this round).
        """
        ema = self.config.cross_time_ema
        cgm = self.config.concept_global_mix
        delta_mode = self.config.concept_delta_mode

        # Track latest global FedAvg for delta-mode reconstruction
        if global_fedavg is not None:
            self._latest_global_fedavg = {
                key: arr.copy() for key, arr in global_fedavg.items()
            }

        for concept_id, agg_params in aggregated.items():
            to_store = agg_params

            # Step 1: Concept-global interpolation — blend concept model
            # with global FedAvg to inherit regularization from all clients
            if cgm > 0.0 and global_fedavg is not None:
                to_store = {
                    key: (1.0 - cgm) * to_store[key] + cgm * global_fedavg[key]
                    for key in to_store
                    if key in global_fedavg
                }
                # Keep any keys only in to_store
                for key in agg_params:
                    if key not in global_fedavg:
                        to_store[key] = agg_params[key]
                # The blended model is also what clients receive
                aggregated[concept_id] = to_store

            # Step 2: Cross-time EMA blending
            if ema < 1.0:
                stored = self.memory_bank.get_model_params(concept_id)
                if stored is not None:
                    blended = {
                        key: ema * to_store[key] + (1.0 - ema) * stored[key]
                        for key in to_store
                        if key in stored
                    }
                    for key in to_store:
                        if key not in stored:
                            blended[key] = to_store[key]
                    to_store = blended
                    aggregated[concept_id] = blended

            # Step 3: Delta mode — store concept_model - global_model
            # so that at recall time we reconstruct with the latest global
            if delta_mode and global_fedavg is not None:
                delta = {
                    key: to_store[key] - global_fedavg[key]
                    for key in to_store
                    if key in global_fedavg
                }
                for key in to_store:
                    if key not in global_fedavg:
                        delta[key] = to_store[key]
                self.memory_bank.store_model_params(concept_id, delta)
            else:
                self.memory_bank.store_model_params(concept_id, to_store)

    def recall_concept_model(
        self,
        concept_id: int,
    ) -> dict[str, np.ndarray] | None:
        """Recall a concept model from the memory bank.

        When ``concept_delta_mode`` is enabled, reconstructs the full model
        from the stored delta and the latest global FedAvg:
        ``model = latest_global + stored_delta``.

        Parameters
        ----------
        concept_id : int
            Concept slot ID.

        Returns
        -------
        dict[str, np.ndarray] or None
            Full model parameters, or None if not stored.
        """
        stored = self.memory_bank.get_model_params(concept_id)
        if stored is None:
            return None
        if (
            self.config.concept_delta_mode
            and self._latest_global_fedavg is not None
        ):
            reconstructed = {
                key: self._latest_global_fedavg[key] + stored[key]
                for key in stored
                if key in self._latest_global_fedavg
            }
            for key in stored:
                if key not in self._latest_global_fedavg:
                    reconstructed[key] = stored[key]
            return reconstructed
        return stored

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

        # Pre-compute global FedAvg for min-group-size fallback blend,
        # concept-global interpolation, concept-delta mode, and DRCT shrinkage
        min_gs = self.config.min_agg_group_size
        needs_global = (
            (min_gs > 1)
            or (self.config.concept_global_mix > 0.0)
            or self.config.concept_delta_mode
            or self.config.drct_shrinkage
        )
        global_fedavg: dict[str, np.ndarray] | None = None
        if needs_global and not uses_namespaces:
            all_params = list(client_params.values())
            if len(all_params) >= 2:
                global_fedavg = self._aggregator.aggregate(all_params)

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
                subgroup_params = self._subgroup_prealign_params(
                    client_ids,
                    client_params,
                    client_fingerprints,
                    mix=self._resolve_alignment_mix(
                        base_mix=0.0,
                        early_rounds=self.config.prototype_subgroup_early_rounds,
                        early_mix=self.config.prototype_subgroup_early_mix,
                    ),
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

                # Min-group-size fallback: blend with global FedAvg
                group_size = len(group_params)
                if (
                    global_fedavg is not None
                    and group_size < min_gs
                    and group_size > 0
                ):
                    blend_w = group_size / min_gs
                    agg = {
                        key: blend_w * agg[key] + (1.0 - blend_w) * global_fedavg[key]
                        for key in agg
                    }

            aggregated[concept_id] = agg

            # Download cost: each client in this cluster gets the aggregated model
            agg_bytes = model_bytes(agg)
            bytes_down += len(client_ids) * agg_bytes

        # DRCT Stein shrinkage: blend each concept toward global FedAvg
        # λ = σ²·d_eff / n_k / (σ²·d_eff / n_k + σ_B²)
        if (
            self.config.drct_shrinkage
            and global_fedavg is not None
            and len(aggregated) >= self.config.drct_min_concepts
        ):
            aggregated = self._apply_drct_shrinkage(
                aggregated, global_fedavg, concept_groups, client_params,
            )

        # Store aggregated models in memory bank for concept-model warm-start
        # with cross-time EMA blending, concept-global interpolation, and
        # concept-delta mode
        self._store_aggregated_models(aggregated, global_fedavg)

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
        global_fedavg_soft: dict[str, np.ndarray] | None = None
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
            # Pre-compute global FedAvg for min-group-size fallback blend
            min_gs = self.config.min_agg_group_size
            needs_global_soft = (
                (min_gs > 1)
                or (self.config.concept_global_mix > 0.0)
                or self.config.concept_delta_mode
                or self.config.drct_shrinkage
            )
            global_fedavg_soft: dict[str, np.ndarray] | None = None
            if needs_global_soft:
                all_params = list(client_params.values())
                if len(all_params) >= 2:
                    global_fedavg_soft = self._aggregator.aggregate(all_params)

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
                weighted_client_ids = list(weights)
                subgroup_params: dict[int, dict[str, np.ndarray]] = {}
                subgroup_param_list = self._subgroup_prealign_params(
                    weighted_client_ids,
                    client_params,
                    client_fingerprints,
                    mix=self._resolve_alignment_mix(
                        base_mix=0.0,
                        early_rounds=self.config.prototype_subgroup_early_rounds,
                        early_mix=self.config.prototype_subgroup_early_mix,
                    ),
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

                # Min-group-size fallback: blend with global FedAvg
                # For soft aggregation, effective group size = total posterior weight
                effective_size = total_w
                if (
                    global_fedavg_soft is not None
                    and effective_size < min_gs
                    and effective_size > 0
                ):
                    blend_w = effective_size / min_gs
                    agg = {
                        key: blend_w * agg[key] + (1.0 - blend_w) * global_fedavg_soft[key]
                        for key in agg
                    }

                aggregated[concept_id] = agg

                # Download cost: clients assigned to this concept get the model
                n_recipients = sum(
                    1 for cid, c in concept_assignments.items()
                    if c == concept_id
                )
                bytes_down += n_recipients * model_bytes(agg)

        # DRCT Stein shrinkage for soft aggregation path
        if (
            self.config.drct_shrinkage
            and global_fedavg_soft is not None
            and len(aggregated) >= self.config.drct_min_concepts
        ):
            # Build concept groups from MAP assignments for DRCT
            soft_concept_groups: dict[int, list[int]] = {}
            for cid, concept_id in concept_assignments.items():
                soft_concept_groups.setdefault(concept_id, []).append(cid)
            aggregated = self._apply_drct_shrinkage(
                aggregated, global_fedavg_soft, soft_concept_groups, client_params,
            )

        # Store aggregated models in memory bank with cross-time EMA blending,
        # concept-global interpolation, and concept-delta mode
        self._store_aggregated_models(aggregated, global_fedavg_soft)

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

