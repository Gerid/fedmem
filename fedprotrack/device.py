from __future__ import annotations

"""Device management for GPU/CPU tensor operations.

Provides a singleton-style device reference so all modules share the same
device (GPU when available, CPU otherwise) without redundant detection.
"""

import os

import torch

_GPU_THRESHOLD = 8192


def get_device(n_params: int = 0) -> torch.device:
    """Return the best available device, respecting model size and env overrides.

    Parameters
    ----------
    n_params : int
        Number of model parameters.  Models smaller than ``_GPU_THRESHOLD``
        (8 192) are placed on CPU to avoid GPU kernel-launch overhead.

    Environment variables
    ---------------------
    FEDPROTRACK_FORCE_CPU : str
        If set to ``"1"``, always return CPU regardless of GPU availability.
    FEDPROTRACK_GPU_THRESHOLD : str
        Override the default 8 192 threshold.  Set to ``"0"`` to always prefer
        GPU (restoring the original behaviour).

    Returns
    -------
    torch.device
        ``cuda`` if a CUDA-capable GPU is available and the model is large
        enough, otherwise ``cpu``.
    """
    if os.environ.get("FEDPROTRACK_FORCE_CPU", "") == "1":
        return torch.device("cpu")

    threshold = int(os.environ.get("FEDPROTRACK_GPU_THRESHOLD", str(_GPU_THRESHOLD)))

    if n_params > 0 and n_params < threshold:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(
    x,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a numpy array or list to a torch tensor on the target device.

    Parameters
    ----------
    x : array-like
        Input data (numpy array, list, or scalar).
    dtype : torch.dtype
        Target dtype. Default ``torch.float32``.
    device : torch.device or None
        Target device. If None, uses ``get_device()``.

    Returns
    -------
    torch.Tensor
    """
    if device is None:
        device = get_device()
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype).to(device)


def to_numpy(x: torch.Tensor):
    """Convert a torch tensor back to numpy (detached, on CPU).

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    numpy.ndarray
    """
    return x.detach().cpu().numpy()
