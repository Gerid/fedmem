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


def _fix_column(
    col: np.ndarray,
    min_group_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Redistribute assignments in one time-step column.

    Ensures every active concept has at least ``min_group_size`` clients.
    When no donor is available, merges the smallest group into the largest.

    Parameters
    ----------
    col : np.ndarray of shape (K,)
        Concept assignments for all clients at one time step.
    min_group_size : int
        Minimum number of clients per active concept.
    rng : np.random.Generator
        Used for tie-breaking when choosing which client to reassign.

    Returns
    -------
    np.ndarray of shape (K,)
        Adjusted assignments.
    """
    col = col.copy()
    K = len(col)

    for _ in range(K):
        concepts, counts = np.unique(col, return_counts=True)
        small_mask = counts < min_group_size
        if not small_mask.any():
            break

        small_order = np.argsort(counts[small_mask])
        small_concepts = concepts[small_mask][small_order]

        large_mask = counts > min_group_size
        if not large_mask.any():
            largest_concept = concepts[np.argmax(counts)]
            smallest_concept = small_concepts[0]
            col[col == smallest_concept] = largest_concept
            continue

        target_concept = small_concepts[0]
        large_concepts = concepts[large_mask]
        large_counts = counts[large_mask]
        donor_concept = large_concepts[np.argmax(large_counts)]

        donor_clients = np.flatnonzero(col == donor_concept)
        victim = rng.choice(donor_clients)
        col[victim] = target_concept

    return col


def generate_concept_matrix_low_singleton(
    K: int,
    T: int,
    n_concepts: int,
    alpha: float,
    seed: int,
    min_group_size: int = 1,
) -> np.ndarray:
    """Generate a (K, T) concept matrix with bounded singleton ratio.

    Wraps :func:`generate_concept_matrix` with column-wise post-processing
    to ensure every active concept at each time step has at least
    ``min_group_size`` clients.  When ``min_group_size <= 1``, behaviour
    is identical to the original generator.

    Parameters
    ----------
    K, T, n_concepts, alpha, seed
        Same as :func:`generate_concept_matrix`.
    min_group_size : int
        Minimum number of clients per active concept at each time step.

    Returns
    -------
    np.ndarray of shape (K, T), dtype int32
    """
    matrix = generate_concept_matrix(
        K=K, T=T, n_concepts=n_concepts, alpha=alpha, seed=seed,
    )

    if min_group_size <= 1:
        return matrix

    rng = np.random.default_rng(seed + 999_999)

    for t in range(T):
        matrix[:, t] = _fix_column(matrix[:, t], min_group_size, rng)

    return matrix.astype(np.int32)
