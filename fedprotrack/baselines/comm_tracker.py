from __future__ import annotations

"""Communication cost accounting for federated baselines.

Pure accounting utilities — no network I/O is performed. All functions
return byte counts as ``float`` for consistent arithmetic with budget
metrics elsewhere in the pipeline.
"""

import numpy as np


def model_bytes(params: dict[str, np.ndarray], precision_bits: int = 32) -> float:
    """Compute bytes required to transmit a model parameter dict.

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Mapping from parameter name to numpy array.
    precision_bits : int
        Bit-width used for each scalar element (default 32 for float32).

    Returns
    -------
    float
        Total transmission size in bytes.

    Raises
    ------
    ValueError
        If ``precision_bits`` is not a positive integer.
    """
    if precision_bits <= 0:
        raise ValueError(f"precision_bits must be > 0, got {precision_bits}")
    if not params:
        return 0.0
    total_elements = sum(arr.size for arr in params.values())
    return float(total_elements * precision_bits / 8)


def prototype_bytes(
    prototypes: dict[int, np.ndarray], precision_bits: int = 32
) -> float:
    """Compute bytes required to transmit a per-class prototype dict.

    Parameters
    ----------
    prototypes : dict[int, np.ndarray]
        Mapping from class label (int) to mean feature vector of shape
        ``(n_features,)``.
    precision_bits : int
        Bit-width used for each scalar element (default 32).

    Returns
    -------
    float
        Total transmission size in bytes.

    Raises
    ------
    ValueError
        If ``precision_bits`` is not a positive integer.
    """
    if precision_bits <= 0:
        raise ValueError(f"precision_bits must be > 0, got {precision_bits}")
    if not prototypes:
        return 0.0
    total_elements = sum(vec.size for vec in prototypes.values())
    return float(total_elements * precision_bits / 8)


def fingerprint_bytes(
    n_features: int, n_classes: int, precision_bits: int = 32,
    *, include_global_mean: bool = True,
) -> float:
    """Compute bytes required to transmit one ConceptFingerprint summary.

    The transmitted payload consists of the label distribution vector
    ``(n_classes,)`` and the per-class conditional means matrix
    ``(n_classes, n_features)``.  Optionally includes the global running
    mean vector ``(n_features,)`` (default: yes, for backward compat).

    Parameters
    ----------
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of class labels tracked by the fingerprint.
    precision_bits : int
        Bit-width used for each scalar element (default 32).
    include_global_mean : bool
        Whether to include the global mean in the payload (default True).

    Returns
    -------
    float
        Total transmission size in bytes.

    Raises
    ------
    ValueError
        If ``precision_bits`` is not positive, or if ``n_features`` or
        ``n_classes`` are not positive integers.
    """
    if precision_bits <= 0:
        raise ValueError(f"precision_bits must be > 0, got {precision_bits}")
    if n_features <= 0:
        raise ValueError(f"n_features must be > 0, got {n_features}")
    if n_classes <= 0:
        raise ValueError(f"n_classes must be > 0, got {n_classes}")
    total_elements = n_classes + n_classes * n_features
    if include_global_mean:
        total_elements += n_features
    return float(total_elements * precision_bits / 8)
