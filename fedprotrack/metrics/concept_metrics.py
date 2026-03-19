from __future__ import annotations

import numpy as np

from .hungarian import align_predictions


def _aligned_predictions(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """Return Hungarian-aligned predictions as int32."""
    ground_truth = np.asarray(ground_truth, dtype=np.int32)
    predicted = np.asarray(predicted, dtype=np.int32)
    aligned, _ = align_predictions(ground_truth, predicted)
    return aligned.astype(np.int32)


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
    aligned = _aligned_predictions(ground_truth, predicted)
    wrong = (aligned != ground_truth).astype(np.float64)
    return float(wrong.mean())


def assignment_switch_rate(predicted: np.ndarray) -> float:
    """Compute how often hard concept assignments change over time.

    Parameters
    ----------
    predicted : np.ndarray
        Shape (K, T), dtype int.

    Returns
    -------
    float
        Mean fraction of client transitions where the assignment changed.
        Returns 0.0 when ``T < 2``.
    """
    predicted = np.asarray(predicted, dtype=np.int32)
    if predicted.ndim != 2:
        raise ValueError(
            f"predicted must be 2-D (K, T), got shape {predicted.shape}"
        )
    if predicted.shape[1] < 2:
        return 0.0
    switches = predicted[:, 1:] != predicted[:, :-1]
    return float(switches.mean())


def avg_clients_per_concept(predicted: np.ndarray) -> float:
    """Compute the mean active group size across all timesteps.

    Parameters
    ----------
    predicted : np.ndarray
        Shape (K, T), dtype int.

    Returns
    -------
    float
        Average number of clients per active concept group.
    """
    predicted = np.asarray(predicted, dtype=np.int32)
    if predicted.ndim != 2:
        raise ValueError(
            f"predicted must be 2-D (K, T), got shape {predicted.shape}"
        )

    active_group_sizes: list[float] = []
    for t in range(predicted.shape[1]):
        _, counts = np.unique(predicted[:, t], return_counts=True)
        active_group_sizes.extend(float(c) for c in counts)
    if not active_group_sizes:
        return 0.0
    return float(np.mean(active_group_sizes))


def singleton_group_ratio(predicted: np.ndarray) -> float:
    """Compute the fraction of active groups that contain a single client.

    Parameters
    ----------
    predicted : np.ndarray
        Shape (K, T), dtype int.

    Returns
    -------
    float
        Ratio of singleton active groups in ``[0, 1]``.
    """
    predicted = np.asarray(predicted, dtype=np.int32)
    if predicted.ndim != 2:
        raise ValueError(
            f"predicted must be 2-D (K, T), got shape {predicted.shape}"
        )

    singleton = 0
    active = 0
    for t in range(predicted.shape[1]):
        _, counts = np.unique(predicted[:, t], return_counts=True)
        active += len(counts)
        singleton += int(np.sum(counts == 1))
    if active == 0:
        return 0.0
    return float(singleton / active)


def memory_reuse_rate(
    ground_truth: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Estimate how often the aligned prediction reuses a seen concept.

    Reuse is counted when a client's aligned predicted concept at time ``t``
    has appeared previously in that client's own aligned trajectory.

    Parameters
    ----------
    ground_truth : np.ndarray
        Shape (K, T), dtype int. Used only for Hungarian alignment.
    predicted : np.ndarray
        Shape (K, T), dtype int.

    Returns
    -------
    float
        Fraction of client-time cells with recurrent aligned predictions.
        Returns 0.0 when ``T < 2``.
    """
    aligned = _aligned_predictions(ground_truth, predicted)
    K, T = aligned.shape
    if T < 2:
        return 0.0

    reuse_hits = 0
    total_cells = K * max(T - 1, 0)
    for k in range(K):
        seen = {int(aligned[k, 0])}
        for t in range(1, T):
            cid = int(aligned[k, t])
            if cid in seen:
                reuse_hits += 1
            seen.add(cid)
    if total_cells == 0:
        return 0.0
    return float(reuse_hits / total_cells)


def routing_consistency(
    soft_assignments: np.ndarray | None,
    predicted: np.ndarray,
) -> float:
    """Measure temporal smoothness of routing decisions.

    With soft assignments this is ``1 - TV(p_t, p_{t-1})`` averaged over all
    client transitions. Without soft assignments it reduces to the fraction of
    unchanged hard assignments.

    Parameters
    ----------
    soft_assignments : np.ndarray or None
        Shape (K, T, C), dtype float64.
    predicted : np.ndarray
        Shape (K, T), dtype int.

    Returns
    -------
    float
        Routing consistency in ``[0, 1]``. Higher is smoother.
    """
    predicted = np.asarray(predicted, dtype=np.int32)
    if predicted.ndim != 2:
        raise ValueError(
            f"predicted must be 2-D (K, T), got shape {predicted.shape}"
        )
    if predicted.shape[1] < 2:
        return 1.0

    if soft_assignments is None:
        return float(1.0 - assignment_switch_rate(predicted))

    soft = np.asarray(soft_assignments, dtype=np.float64)
    if soft.ndim != 3:
        raise ValueError(
            f"soft_assignments must be 3-D (K, T, C), got shape {soft.shape}"
        )
    if soft.shape[:2] != predicted.shape:
        raise ValueError(
            "soft_assignments leading dimensions must match predicted: "
            f"{soft.shape[:2]} vs {predicted.shape}"
        )

    prev = soft[:, :-1, :]
    cur = soft[:, 1:, :]
    tv = 0.5 * np.abs(cur - prev).sum(axis=-1)
    score = 1.0 - np.clip(tv, 0.0, 1.0)
    return float(score.mean())
