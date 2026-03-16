"""Concept matrix generation for federated concept drift.

The concept matrix M is a (K, T) integer array where M[k, t] is the concept ID
for client k at time step t. Controlled by:
  - rho: recurrence frequency (concept pool size = T / rho)
  - alpha: asynchrony level (0 = synchronous, 1 = fully asynchronous)
"""

from __future__ import annotations

import numpy as np


def _generate_markov_sequence(
    T: int, n_concepts: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a length-T Markov chain over n_concepts states.

    At each step, with probability p_switch, transitions to a uniformly random
    *different* concept; otherwise stays. p_switch is tuned so that the expected
    number of distinct segments ≈ n_concepts.
    """
    # Expected number of transitions = n_concepts - 1
    # p_switch per step = (n_concepts - 1) / (T - 1) to get ~n_concepts segments
    if n_concepts >= T:
        # Every step is a different concept
        return rng.permutation(T).astype(np.int32)

    p_switch = min(1.0, (n_concepts - 1) / (T - 1))

    seq = np.zeros(T, dtype=np.int32)
    seq[0] = rng.integers(0, n_concepts)

    for t in range(1, T):
        if rng.random() < p_switch:
            # Switch to a different concept
            candidates = [c for c in range(n_concepts) if c != seq[t - 1]]
            seq[t] = rng.choice(candidates)
        else:
            seq[t] = seq[t - 1]

    return seq


def generate_concept_matrix(
    K: int,
    T: int,
    n_concepts: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    """Generate a (K, T) concept assignment matrix.

    Parameters
    ----------
    K : int
        Number of clients.
    T : int
        Number of time steps.
    n_concepts : int
        Size of the shared concept pool.
    alpha : float
        Asynchrony level in [0, 1].
        0 = fully synchronous (all clients share the same drift schedule).
        1 = fully asynchronous (independent Markov chains per client).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    matrix : np.ndarray of shape (K, T) with dtype int32
        Concept ID assignments.
    """
    rng = np.random.default_rng(seed)

    # Generate reference (synchronous) drift schedule
    reference = _generate_markov_sequence(T, n_concepts, rng)

    matrix = np.zeros((K, T), dtype=np.int32)

    for k in range(K):
        if alpha == 0.0:
            matrix[k] = reference.copy()
        elif alpha == 1.0:
            matrix[k] = _generate_markov_sequence(T, n_concepts, rng)
        else:
            # Interpolation: each client runs its own chain, but at each step
            # follows the reference with probability (1-alpha) or makes an
            # independent transition with probability alpha.
            independent = _generate_markov_sequence(T, n_concepts, rng)
            mask = rng.random(T) < alpha
            matrix[k] = np.where(mask, independent, reference)

    return matrix
