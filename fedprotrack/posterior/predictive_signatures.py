from __future__ import annotations

"""Predictive signature helpers for FedProTrack routing research.

These utilities are intentionally lightweight and self-contained so they can
be reused by Phase A routing experiments without pulling in the rest of the
protocol stack. They cover three common objects:

* classifier-row signatures from model parameters
* local update-delta signatures from two parameter snapshots
* batch prototype summaries / projected prototype signatures
"""

import numpy as np


def _validate_projection_args(input_dim: int, output_dim: int) -> None:
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}")
    if output_dim < 1:
        raise ValueError(f"output_dim must be >= 1, got {output_dim}")


def _projection_matrix(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    """Build a deterministic Gaussian projection matrix."""
    _validate_projection_args(input_dim, output_dim)
    rng = np.random.default_rng(seed + 997 * input_dim + 17 * output_dim)
    return rng.standard_normal((input_dim, output_dim), dtype=np.float64) / np.sqrt(
        float(output_dim)
    )


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with zero-row preservation."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def classifier_rows_from_params(
    params: dict[str, np.ndarray],
    *,
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Convert a classifier parameter dict into per-class row vectors.

    The returned matrix has shape ``(n_rows, n_features + 1)`` where the last
    column stores the bias term. For binary models stored with a single row,
    we materialize the symmetric two-row representation.
    """
    if "coef" not in params or "intercept" not in params:
        raise KeyError("params must contain 'coef' and 'intercept'")
    coef = np.asarray(params["coef"], dtype=np.float64).reshape(-1)
    intercept = np.asarray(params["intercept"], dtype=np.float64).reshape(-1)

    if n_classes == 2:
        if coef.size == n_features and intercept.size == 1:
            row = np.concatenate([coef.reshape(1, n_features), intercept.reshape(1, 1)], axis=1)
            return np.vstack([-row, row])
        if coef.size == 2 * n_features and intercept.size == 2:
            return np.concatenate(
                [
                    coef.reshape(2, n_features),
                    intercept.reshape(2, 1),
                ],
                axis=1,
            )
        raise ValueError(
            "Binary classifier params must have shapes "
            f"({n_features},)/(1,) or ({2 * n_features},)/(2,), "
            f"got coef={coef.shape}, intercept={intercept.shape}",
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


def project_classifier_row_signatures(
    params: dict[str, np.ndarray],
    *,
    n_features: int,
    n_classes: int,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Project per-class classifier rows into a compact signature space."""
    rows = classifier_rows_from_params(params, n_features=n_features, n_classes=n_classes)
    proj = _projection_matrix(rows.shape[1], output_dim, seed)
    return _normalize_rows(rows @ proj)


def project_update_delta_signatures(
    current_params: dict[str, np.ndarray],
    previous_params: dict[str, np.ndarray],
    *,
    n_features: int,
    n_classes: int,
    output_dim: int,
    seed: int,
) -> np.ndarray:
    """Project the parameter delta between two classifier snapshots."""
    current_rows = classifier_rows_from_params(
        current_params,
        n_features=n_features,
        n_classes=n_classes,
    )
    previous_rows = classifier_rows_from_params(
        previous_params,
        n_features=n_features,
        n_classes=n_classes,
    )
    if current_rows.shape != previous_rows.shape:
        raise ValueError(
            f"Row-matrix shape mismatch: {current_rows.shape} vs {previous_rows.shape}",
        )
    proj = _projection_matrix(current_rows.shape[1], output_dim, seed)
    return _normalize_rows((current_rows - previous_rows) @ proj)


def extract_batch_prototypes(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-class means and counts from a batch.

    Returns
    -------
    prototypes : np.ndarray
        Shape ``(n_classes, n_features)``. Missing classes are zero-filled.
    counts : np.ndarray
        Shape ``(n_classes,)`` with raw sample counts.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}")
    if len(X) == 0:
        raise ValueError("X and y must be non-empty")

    if n_classes is None:
        n_classes = int(y.max()) + 1
    if n_classes < 1:
        raise ValueError(f"n_classes must be >= 1, got {n_classes}")

    prototypes = np.zeros((n_classes, X.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.float64)
    for cls in range(n_classes):
        mask = y == cls
        count = int(mask.sum())
        counts[cls] = float(count)
        if count > 0:
            prototypes[cls] = X[mask].mean(axis=0)
    return prototypes, counts


def project_batch_prototype_signatures(
    X: np.ndarray,
    y: np.ndarray,
    *,
    output_dim: int,
    seed: int,
    n_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project batch prototypes into a compact signature space.

    The returned tuple is ``(signatures, counts)`` where ``signatures`` has
    shape ``(n_classes, output_dim)`` and ``counts`` are the raw class counts.
    """
    prototypes, counts = extract_batch_prototypes(X, y, n_classes=n_classes)
    mass = counts.reshape(-1, 1)
    mass_scale = max(float(mass.sum()), 1.0)
    rows = np.concatenate([prototypes, mass / mass_scale], axis=1)
    proj = _projection_matrix(rows.shape[1], output_dim, seed)
    return _normalize_rows(rows @ proj), counts
