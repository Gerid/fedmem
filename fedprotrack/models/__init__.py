from __future__ import annotations

"""PyTorch-based models for federated learning on GPU."""

from .cnn import TorchSmallCNN
from .factory import create_model
from .torch_feature_adapter import TorchFeatureAdapterClassifier
from .torch_factorized_adapter import TorchFactorizedAdapterClassifier
from .torch_linear import TorchLinearClassifier

__all__ = [
    "TorchLinearClassifier",
    "TorchFeatureAdapterClassifier",
    "TorchFactorizedAdapterClassifier",
    "TorchSmallCNN",
    "create_model",
]
