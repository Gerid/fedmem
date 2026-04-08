from __future__ import annotations

"""Model factory for creating classifier instances by type name.

Provides a single ``create_model`` entry-point used by baseline runners
so that the model architecture can be swapped without touching each
baseline's Client class.
"""

from .torch_linear import TorchLinearClassifier

MODEL_REGISTRY: dict[str, type] = {
    "linear": TorchLinearClassifier,
}


def _ensure_cnn_registered() -> None:
    """Lazily register CNN models to avoid hard torch/torchvision dependency at import."""
    if "small_cnn" not in MODEL_REGISTRY:
        try:
            from .cnn import TorchSmallCNN

            MODEL_REGISTRY["small_cnn"] = TorchSmallCNN
        except ImportError:
            pass


def create_model(
    model_type: str,
    n_features: int,
    n_classes: int,
    *,
    seed: int = 42,
    **kwargs,
) -> TorchLinearClassifier:
    """Create a model instance by type name.

    Parameters
    ----------
    model_type : str
        One of ``"linear"``, ``"small_cnn"``.
    n_features : int
        Input feature dimension (used by linear; passed but may be ignored
        by CNN variants that infer shape from the image).
    n_classes : int
        Number of output classes.
    seed : int
        Random seed for initialisation. Default 42.
    **kwargs
        Extra keyword arguments forwarded to the model constructor
        (e.g. ``lr``, ``n_epochs``, ``input_channels``, ``image_size``).

    Returns
    -------
    object
        Model instance with ``fit / predict / get_params / set_params``
        interface.

    Raises
    ------
    ValueError
        If *model_type* is not in the registry.
    """
    _ensure_cnn_registered()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type](n_features, n_classes, seed=seed, **kwargs)
