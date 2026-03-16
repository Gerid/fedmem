from __future__ import annotations

"""Device management for GPU/CPU tensor operations.

Provides a singleton-style device reference so all modules share the same
device (GPU when available, CPU otherwise) without redundant detection.
"""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU).

    Returns
    -------
    torch.device
        ``cuda`` if a CUDA-capable GPU is available, otherwise ``cpu``.
    """
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
