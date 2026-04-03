"""Gibbs posterior concept assignment — probabilistic concept identity inference.

Implements the core FedProTrack inference mechanism:

    p(z_t^{(i)} = k) ∝ exp{-ω · ℓ(o, m_k)} · p(z_t^{(i)} = k | z_{t-1}^{(i)})

where:
  - z_t^{(i)} is the concept assignment for client i at time t
  - k indexes over known concepts in the memory bank
  - ω (omega) is an inverse-temperature controlling posterior sharpness
  - ℓ(o, m_k) is the loss of observation o under concept memory m_k
  - p(z_t | z_{t-1}) is the transition prior from a sticky factorial HMM

This replaces the hard-threshold concept identification in ConceptTracker
with a full posterior distribution over concept identities, enabling
soft aggregation and principled uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..concept_tracker.fingerprint import ConceptFingerprint
from .retrieval_keys import SupportsSimilarity


@dataclass
class PosteriorAssignment:
    """Result of a Gibbs posterior concept assignment step.

    Parameters
    ----------
    probabilities : dict[int, float]
        Mapping from concept_id to posterior probability p(z=k|data).
    map_concept_id : int
        Maximum a-posteriori concept ID (argmax of probabilities).
    is_novel : bool
        Whether the MAP assignment is to a newly spawned concept.
    entropy : float
        Shannon entropy of the posterior distribution (nats).
    log_likelihood : dict[int, float]
        Per-concept log-likelihood contribution -ω·ℓ(o, m_k).
    """

    probabilities: dict[int, float]
    map_concept_id: int
    is_novel: bool
    entropy: float
    log_likelihood: dict[int, float] = field(default_factory=dict)


class TransitionPrior:
    """Sticky factorial HMM transition prior for concept persistence.

    Models the tendency of concepts to persist over time (stickiness)
    while allowing transitions to any known concept.

    The transition probability is:

        p(z_t = k | z_{t-1} = j) =
            kappa                   if k == j  (self-transition / sticky)
            (1 - kappa) / (K - 1)   if k != j  (transition to other concept)

    where kappa is the stickiness parameter and K is the number of
    known concepts.

    Parameters
    ----------
    kappa : float
        Stickiness parameter in (0, 1). Higher values make concepts
        more likely to persist. Default 0.8.
    """

    def __init__(self, kappa: float = 0.8):
        if not 0.0 < kappa < 1.0:
            raise ValueError(
                f"kappa must be in (0, 1), got {kappa}"
            )
        self.kappa = kappa

    def log_transition(
        self,
        prev_concept_id: int | None,
        target_concept_id: int,
        known_concept_ids: list[int],
    ) -> float:
        """Compute log transition probability.

        Parameters
        ----------
        prev_concept_id : int or None
            Previous concept assignment. None if this is the first step
            (uniform prior is used).
        target_concept_id : int
            Candidate concept to transition to.
        known_concept_ids : list of int
            All currently known concept IDs.

        Returns
        -------
        float
            Log probability of transitioning to target_concept_id.

        Raises
        ------
        ValueError
            If known_concept_ids is empty.
        """
        K = len(known_concept_ids)
        if K == 0:
            raise ValueError("known_concept_ids must be non-empty")

        if prev_concept_id is None:
            # Uniform prior at t=0
            return -np.log(K)

        if K == 1:
            # Only one concept: must stay
            return 0.0

        if target_concept_id == prev_concept_id:
            return np.log(self.kappa)
        else:
            return np.log((1.0 - self.kappa) / (K - 1))


class GibbsPosterior:
    """Gibbs posterior inference for concept identity.

    Computes the posterior distribution over concept identities using
    the Gibbs posterior formulation, which combines a loss-based
    likelihood with a transition prior.

    The loss function is based on the negative log-similarity between
    an observation fingerprint and each concept's stored fingerprint,
    providing a natural bridge to the existing ConceptFingerprint
    infrastructure.

    Parameters
    ----------
    omega : float
        Inverse temperature (posterior sharpness). Higher values make
        the posterior more peaked around the best-matching concept.
        Must be > 0. Default 1.0.
    transition_prior : TransitionPrior or None
        Transition prior for temporal coherence. If None, a default
        TransitionPrior(kappa=0.8) is created.
    novelty_threshold : float
        If the MAP concept's posterior probability is below this
        threshold, the observation is classified as a novel concept.
        Default 0.3.
    """

    def __init__(
        self,
        omega: float = 1.0,
        transition_prior: TransitionPrior | None = None,
        novelty_threshold: float = 0.3,
    ):
        if omega <= 0:
            raise ValueError(f"omega must be > 0, got {omega}")
        if not 0.0 < novelty_threshold < 1.0:
            raise ValueError(
                f"novelty_threshold must be in (0, 1), got {novelty_threshold}"
            )
        self.omega = omega
        self.transition_prior = transition_prior or TransitionPrior()
        self.novelty_threshold = novelty_threshold

    def compute_loss(
        self,
        observation_fp: ConceptFingerprint,
        concept_fp: SupportsSimilarity | ConceptFingerprint,
    ) -> float:
        """Compute the loss ℓ(o, m_k) between an observation and a concept.

        Uses 1 - similarity as the loss function, mapping the [0, 1]
        similarity score to a [0, 1] loss.

        Parameters
        ----------
        observation_fp : ConceptFingerprint
            Fingerprint of the current observation batch.
        concept_fp : ConceptFingerprint
            Stored fingerprint of concept k.

        Returns
        -------
        float
            Loss in [0, 1]. Lower means better match.
        """
        try:
            sim = float(concept_fp.similarity(observation_fp))
        except Exception:
            sim = float(observation_fp.similarity(concept_fp))
        return 1.0 - sim

    def compute_posterior(
        self,
        observation_fp: ConceptFingerprint,
        concept_library: dict[int, SupportsSimilarity | ConceptFingerprint],
        prev_concept_id: int | None = None,
    ) -> PosteriorAssignment:
        """Compute the full Gibbs posterior over concept identities.

        Implements:
            p(z=k) ∝ exp{-ω · ℓ(o, m_k)} · p(z=k | z_{t-1})

        Parameters
        ----------
        observation_fp : ConceptFingerprint
            Fingerprint built from the current observation batch.
        concept_library : dict[int, ConceptFingerprint]
            Mapping from concept_id to stored concept fingerprint.
        prev_concept_id : int or None
            Previous concept assignment (None for first step).

        Returns
        -------
        PosteriorAssignment
            Full posterior distribution and derived quantities.

        Raises
        ------
        ValueError
            If concept_library is empty.
        """
        if not concept_library:
            raise ValueError("concept_library must be non-empty")

        concept_losses = {
            cid: self.compute_loss(observation_fp, concept_fp)
            for cid, concept_fp in concept_library.items()
        }
        return self.compute_posterior_from_losses(
            concept_losses,
            prev_concept_id=prev_concept_id,
        )

    def compute_posterior_from_losses(
        self,
        concept_losses: dict[int, float],
        prev_concept_id: int | None = None,
    ) -> PosteriorAssignment:
        """Compute the full Gibbs posterior from external per-concept losses.

        Parameters
        ----------
        concept_losses : dict[int, float]
            Concept ID -> loss value in [0, 1]. Lower means a better match.
        prev_concept_id : int or None
            Previous concept assignment (None for first step).

        Returns
        -------
        PosteriorAssignment
            Full posterior distribution and derived quantities.

        Raises
        ------
        ValueError
            If concept_losses is empty.
        """
        if not concept_losses:
            raise ValueError("concept_losses must be non-empty")

        concept_ids = list(concept_losses.keys())
        log_posteriors: dict[int, float] = {}
        log_likelihoods: dict[int, float] = {}

        for cid in concept_ids:
            loss = float(concept_losses[cid])
            log_lik = -self.omega * loss
            log_likelihoods[cid] = log_lik

            log_prior = self.transition_prior.log_transition(
                prev_concept_id, cid, concept_ids,
            )
            log_posteriors[cid] = log_lik + log_prior

        # Normalize via log-sum-exp for numerical stability
        log_vals = np.array([log_posteriors[cid] for cid in concept_ids])
        log_Z = _log_sum_exp(log_vals)

        probabilities: dict[int, float] = {}
        for i, cid in enumerate(concept_ids):
            probabilities[cid] = float(np.exp(log_vals[i] - log_Z))

        map_cid = max(probabilities, key=probabilities.get)  # type: ignore[arg-type]
        map_prob = probabilities[map_cid]

        probs_arr = np.array(list(probabilities.values()))
        probs_arr = np.clip(probs_arr, 1e-15, 1.0)
        entropy = float(-np.sum(probs_arr * np.log(probs_arr)))

        return PosteriorAssignment(
            probabilities=probabilities,
            map_concept_id=map_cid,
            is_novel=map_prob < self.novelty_threshold,
            entropy=entropy,
            log_likelihood=log_likelihoods,
        )

    def soft_weights(
        self,
        assignment: PosteriorAssignment,
        concept_ids: list[int] | None = None,
    ) -> dict[int, float]:
        """Extract soft aggregation weights from a posterior assignment.

        These weights can be used for soft model aggregation, where a
        client's model update is distributed across concept clusters
        proportionally to the posterior.

        Parameters
        ----------
        assignment : PosteriorAssignment
            Result from compute_posterior.
        concept_ids : list of int, optional
            Subset of concept IDs to include. If None, all concepts
            in the assignment are included.

        Returns
        -------
        dict[int, float]
            Mapping from concept_id to aggregation weight (sums to 1).
        """
        if concept_ids is None:
            return dict(assignment.probabilities)

        weights = {
            cid: assignment.probabilities.get(cid, 0.0)
            for cid in concept_ids
        }
        total = sum(weights.values())
        if total > 0:
            weights = {cid: w / total for cid, w in weights.items()}
        return weights


def calibrate_omega(
    concept_losses: dict[int, dict[int, float]],
    target_confidence: float = 0.9,
) -> float:
    """Estimate omega from observed fingerprint loss statistics.

    Computes a label-free omega that makes the posterior assign at least
    ``target_confidence`` to the best-matching concept for a typical
    observation.  The rule is:

        omega = -log(target_confidence / (1 - target_confidence) * (K-1))
                / (min_loss - median_other_loss)

    where ``min_loss`` is the loss to the best-matching concept and
    ``median_other_loss`` is the median loss to all other concepts,
    averaged over observations.

    Parameters
    ----------
    concept_losses : dict[int, dict[int, float]]
        Mapping ``observation_id -> {concept_id: loss}``.  Typically
        collected during the first few federation rounds.
    target_confidence : float
        Desired posterior probability on the best concept (default 0.9).

    Returns
    -------
    float
        Calibrated omega, clipped to [1.0, 100.0].
    """
    if not concept_losses:
        return 1.0
    gaps: list[float] = []
    K_vals: list[int] = []
    for losses in concept_losses.values():
        if len(losses) < 2:
            continue
        sorted_losses = sorted(losses.values())
        best = sorted_losses[0]
        others = sorted_losses[1:]
        median_other = float(np.median(others))
        gap = median_other - best
        if gap > 1e-8:
            gaps.append(gap)
            K_vals.append(len(losses))
    if not gaps:
        return 1.0
    mean_gap = float(np.mean(gaps))
    K = int(np.median(K_vals))
    # Solve: exp(-omega * best) / (exp(-omega * best) + (K-1)*exp(-omega * med))
    #      = target_confidence
    # => omega * (med - best) = log(target / (1-target) * (K-1))
    ratio = target_confidence / (1.0 - target_confidence) * max(K - 1, 1)
    omega = float(np.log(ratio) / mean_gap)
    return float(np.clip(omega, 1.0, 100.0))


def _log_sum_exp(log_vals: np.ndarray) -> float:
    """Numerically stable log-sum-exp.

    Parameters
    ----------
    log_vals : np.ndarray
        Array of log values.

    Returns
    -------
    float
        log(sum(exp(log_vals))).
    """
    max_val = float(np.max(log_vals))
    if not np.isfinite(max_val):
        return max_val
    return max_val + float(np.log(np.sum(np.exp(log_vals - max_val))))
