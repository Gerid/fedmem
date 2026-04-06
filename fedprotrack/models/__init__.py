from __future__ import annotations

"""PyTorch-based models for federated learning on GPU."""

from .cnn import TorchMobileNetV2, TorchSmallCNN
from .torch_feature_adapter import TorchFeatureAdapterClassifier
from .torch_factorized_adapter import TorchFactorizedAdapterClassifier
from .torch_linear import TorchLinearClassifier

__all__ = [
    "TorchLinearClassifier",
    "TorchFeatureAdapterClassifier",
    "TorchFactorizedAdapterClassifier",
    "TorchSmallCNN",
    "TorchMobileNetV2",
]
