from __future__ import annotations

"""Lightweight similarity helpers for routing-based concept assignment.

This module provides small, dependency-free utilities that can support
FedProTrack Phase A routing without coupling the protocol to a specific
distance family. The core pieces are:

* Sinkhorn transport cost / plan computation
* Prototype cost matrix construction
* Prototype transport similarity helpers

The helpers operate on NumPy arrays and are intentionally generic so the
main protocol can combine them with fingerprint, model-signature, or
update-signature objects.
"""

import numpy as np


def normalize_mass(mass: np.ndarray, *, name: str = "mass") -> np.ndarray:
    """Normalize a non-negative mass vector to sum to one.

    Parameters
    ----------
    mass : np.ndarray
        One-dimensional non-negative vector.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        Normalized probability vector.
    """
    vec = np.asarray(mass, dtype=np.float64).reshape(-1)
    if vec.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(~np.isfinite(vec)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(vec < 0.0):
        raise ValueError(f"{name} must be non-negative")
    total = float(vec.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return vec / total


def pairwise_euclidean_cost(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compute a pairwise Euclidean cost matrix."""
    src = _as_2d_array(source, name="source")
    tgt = _as_2d_array(target, name="target")
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"source and target must share feature dimension, got "
            f"{src.shape[1]} and {tgt.shape[1]}",
        )
    diff = src[:, None, :] - tgt[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def pairwise_cosine_cost(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compute a cosine-distance cost matrix in [0, 2]."""
    src = _as_2d_array(source, name="source")
    tgt = _as_2d_array(target, name="target")
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"source and target must share feature dimension, got "
            f"{src.shape[1]} and {tgt.shape[1]}",
        )
    src_norm = np.linalg.norm(src, axis=1, keepdims=True)
    tgt_norm = np.linalg.norm(tgt, axis=1, keepdims=True)
    src_norm = np.maximum(src_norm, 1e-12)
    tgt_norm = np.maximum(tgt_norm, 1e-12)
    cosine = (src / src_norm) @ (tgt / tgt_norm).T
    cosine = np.clip(cosine, -1.0, 1.0)
    return 1.0 - cosine


def prototype_cost_matrix(
    source_prototypes: np.ndarray,
    target_prototypes: np.ndarray,
    *,
    metric: str = "euclidean",
) -> np.ndarray:
    """Build a pairwise cost matrix between prototype sets.

    Parameters
    ----------
    source_prototypes : np.ndarray
        Shape (n_source, d).
    target_prototypes : np.ndarray
        Shape (n_target, d).
    metric : str
        One of ``"euclidean"``, ``"sqeuclidean"``, or ``"cosine"``.

    Returns
    -------
    np.ndarray
        Shape (n_source, n_target).
    """
    src = _as_2d_array(source_prototypes, name="source_prototypes")
    tgt = _as_2d_array(target_prototypes, name="target_prototypes")
    if src.shape[1] != tgt.shape[1]:
        raise ValueError(
            f"prototype feature dimensions must match, got "
            f"{src.shape[1]} and {tgt.shape[1]}",
        )

    diff = src[:, None, :] - tgt[None, :, :]
    if metric == "euclidean":
        return np.sqrt(np.sum(diff ** 2, axis=2))
    if metric == "sqeuclidean":
        return np.sum(diff ** 2, axis=2)
    if metric == "cosine":
        return pairwise_cosine_cost(src, tgt)
    raise ValueError(
        "metric must be one of {'euclidean', 'sqeuclidean', 'cosine'}, "
        f"got {metric!r}",
    )


def sinkhorn_transport_plan(
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Compute an entropic OT plan with a basic Sinkhorn solver.

    Parameters
    ----------
    cost : np.ndarray
        Transport cost matrix of shape (n_source, n_target).
    source_mass : np.ndarray
        Non-negative source mass vector.
    target_mass : np.ndarray
        Non-negative target mass vector.
    reg : float
        Entropic regularization strength. Must be > 0.
    max_iter : int
        Maximum Sinkhorn iterations.
    tol : float
        Early stopping threshold on scaling updates.
    epsilon : float
        Numerical floor for divisions and the Gibbs kernel.

    Returns
    -------
    np.ndarray
        Transport plan of shape (n_source, n_target).
    """
    cost_arr = _as_2d_array(cost, name="cost")
    if reg <= 0.0:
        raise ValueError(f"reg must be > 0, got {reg}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol < 0.0:
        raise ValueError(f"tol must be >= 0, got {tol}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    p = normalize_mass(source_mass, name="source_mass")
    q = normalize_mass(target_mass, name="target_mass")
    if cost_arr.shape != (p.size, q.size):
        raise ValueError(
            f"cost shape {cost_arr.shape} does not match masses "
            f"{p.size} x {q.size}",
        )

    kernel = np.exp(-cost_arr / reg)
    kernel = np.maximum(kernel, epsilon)
    u = np.ones_like(p)
    v = np.ones_like(q)

    for _ in range(max_iter):
        prev_u = u.copy()
        prev_v = v.copy()

        Kv = kernel @ v
        Kv = np.maximum(Kv, epsilon)
        u = p / Kv

        KTu = kernel.T @ u
        KTu = np.maximum(KTu, epsilon)
        v = q / KTu

        delta_u = float(np.max(np.abs(u - prev_u)))
        delta_v = float(np.max(np.abs(v - prev_v)))
        if max(delta_u, delta_v) <= tol:
            break

    plan = (u[:, None] * kernel) * v[None, :]
    return np.maximum(plan, 0.0)


def sinkhorn_transport_cost(
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
    epsilon: float = 1e-12,
) -> float:
    """Compute the OT cost under the Sinkhorn plan."""
    plan = sinkhorn_transport_plan(
        cost,
        source_mass,
        target_mass,
        reg=reg,
        max_iter=max_iter,
        tol=tol,
        epsilon=epsilon,
    )
    cost_arr = _as_2d_array(cost, name="cost")
    if plan.shape != cost_arr.shape:
        raise RuntimeError("Sinkhorn plan shape mismatch")
    return float(np.sum(plan * cost_arr))


def sinkhorn_transport_similarity(
    cost: np.ndarray,
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    *,
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
    epsilon: float = 1e-12,
    temperature: float = 1.0,
) -> float:
    """Convert a Sinkhorn transport cost into a bounded similarity score."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    transport_cost = sinkhorn_transport_cost(
        cost,
        source_mass,
        target_mass,
        reg=reg,
        max_iter=max_iter,
        tol=tol,
        epsilon=epsilon,
    )
    return float(np.exp(-transport_cost / temperature))


def prototype_transport_cost(
    source_prototypes: np.ndarray,
    target_prototypes: np.ndarray,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    *,
    metric: str = "euclidean",
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
    epsilon: float = 1e-12,
) -> float:
    """Compute OT cost between two prototype sets."""
    cost = prototype_cost_matrix(
        source_prototypes,
        target_prototypes,
        metric=metric,
    )
    if source_mass is None:
        source_mass = np.ones(cost.shape[0], dtype=np.float64)
    if target_mass is None:
        target_mass = np.ones(cost.shape[1], dtype=np.float64)
    return sinkhorn_transport_cost(
        cost,
        source_mass,
        target_mass,
        reg=reg,
        max_iter=max_iter,
        tol=tol,
        epsilon=epsilon,
    )


def prototype_transport_similarity(
    source_prototypes: np.ndarray,
    target_prototypes: np.ndarray,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    *,
    metric: str = "euclidean",
    reg: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
    epsilon: float = 1e-12,
    temperature: float = 1.0,
) -> float:
    """Convert prototype OT cost into a bounded similarity score."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    transport_cost = prototype_transport_cost(
        source_prototypes,
        target_prototypes,
        source_mass,
        target_mass,
        metric=metric,
        reg=reg,
        max_iter=max_iter,
        tol=tol,
        epsilon=epsilon,
    )
    return float(np.exp(-transport_cost / temperature))


def _as_2d_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
