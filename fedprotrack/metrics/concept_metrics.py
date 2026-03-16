from __future__ import annotations

import numpy as np

from .hungarian import align_predictions


def concept_re_id_accuracy(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute concept re-identification accuracy after Hungarian alignment.

    Parameters
    ----------
    ground_truth : np.ndarray
        Shape (K, T), dtype int.  Ground-truth concept IDs.
    predicted : np.ndarray
        Shape (K, T), dtype int.  Predicted concept IDs.

    Returns
    -------
    global_accuracy : float
        Mean accuracy over all (k, t) cells.
    per_client : np.ndarray
        Shape (K,), dtype float64.  Per-client accuracy (mean over T).
    per_timestep : np.ndarray
        Shape (T,), dtype float64.  Per-timestep accuracy (mean over K).
    """
    ground_truth = np.asarray(ground_truth, dtype=np.int32)
    predicted = np.asarray(predicted, dtype=np.int32)

    aligned, _ = align_predictions(ground_truth, predicted)

    match = (aligned == ground_truth).astype(np.float64)  # (K, T)

    global_accuracy: float = float(match.mean())
    per_client: np.ndarray = match.mean(axis=1)    # mean over T → shape (K,)
    per_timestep: np.ndarray = match.mean(axis=0)  # mean over K → shape (T,)

    return global_accuracy, per_client, per_timestep


def assignment_entropy(
    soft_assignments: np.ndarray | None,
    predicted: np.ndarray,
    n_concepts: int,
) -> float:
    """Compute mean entropy of the concept assignment distribution.

    When *soft_assignments* are available the per-cell entropy is computed
    directly from the probability vectors.  Otherwise the marginal distribution
    across clients at each time step is used as a proxy.

    Parameters
    ----------
    soft_assignments : np.ndarray or None
        Shape (K, T, C), dtype float64.  Soft assignment probabilities.  If
        ``None``, hard predictions are used instead.
    predicted : np.ndarray
        Shape (K, T), dtype int.  Hard predicted concept IDs (used when
        *soft_assignments* is ``None``).
    n_concepts : int
        Number of distinct concepts C.  Used only when *soft_assignments* is
        ``None`` to define the support of the marginal distribution.

    Returns
    -------
    float
        Mean entropy (nats) over all evaluation cells.
    """
    eps = 1e-12

    if soft_assignments is not None:
        # soft_assignments: (K, T, C)
        p = np.asarray(soft_assignments, dtype=np.float64)
        # Clamp to avoid log(0).
        p = np.clip(p, eps, None)
        # Entropy per (k, t) cell: shape (K, T)
        H = -np.sum(p * np.log(p), axis=-1)
        return float(H.mean())

    # Fallback: marginal distribution over clients at each time step.
    predicted = np.asarray(predicted, dtype=np.int32)
    K, T = predicted.shape
    entropies = np.empty(T, dtype=np.float64)
    for t in range(T):
        counts = np.zeros(n_concepts, dtype=np.float64)
        for k in range(K):
            cid = int(predicted[k, t])
            if 0 <= cid < n_concepts:
                counts[cid] += 1.0
        p = counts / (counts.sum() + eps)
        p = np.clip(p, eps, None)
        entropies[t] = float(-np.sum(p * np.log(p)))
    return float(entropies.mean())


def wrong_memory_reuse_rate(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Fraction of (k, t) cells where the aligned prediction is incorrect.

    This equals ``1 − concept_re_id_accuracy`` but is provided as a
    semantically distinct function: a high value indicates that a client is
    frequently loading the wrong memory/model snapshot.

    Parameters
    ----------
    ground_truth : np.ndarray
        Shape (K, T), dtype int.
    predicted : np.ndarray
        Shape (K, T), dtype int.

    Returns
    -------
    float
        Wrong-memory reuse rate in ``[0, 1]``.
    """
    ground_truth = np.asarray(ground_truth, dtype=np.int32)
    predicted = np.asarray(predicted, dtype=np.int32)

    aligned, _ = align_predictions(ground_truth, predicted)

    wrong = (aligned != ground_truth).astype(np.float64)
    return float(wrong.mean())
