from __future__ import annotations

"""End-to-end trainable CNN models for federated learning on raw images.

Provides ``TorchSmallCNN`` (a lightweight 2-conv architecture) and
``TorchMobileNetV2`` (a torchvision MobileNetV2 wrapper) with the same
parameter-dict interface used by the federation layer, so that aggregators
and ``comm_tracker`` remain unchanged.

Input convention
----------------
All models in this module expect raw images of shape ``(N, 3, 32, 32)``
with pixel values in ``[0, 1]`` (float32), **not** pre-extracted features.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn

from ..device import get_device, to_numpy, to_tensor


# ======================================================================
# SmallCNN backbone
# ======================================================================

class _SmallCNNNet(nn.Module):
    """Conv2d(3,32) -> ReLU -> MaxPool -> Conv2d(32,64) -> ReLU -> MaxPool -> FC(128) -> ReLU -> FC(num_classes).

    Designed for 32x32 RGB inputs (CIFAR-style).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N, 3, 32, 32)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(N, num_classes)``.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ======================================================================
# TorchSmallCNN  -- federation-compatible wrapper
# ======================================================================

class TorchSmallCNN:
    """End-to-end trainable small CNN for 32x32 RGB images.

    Architecture::

        Conv2d(3, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Linear(64*8*8, 128) -> ReLU -> Linear(128, num_classes)

    Exposes the same ``fit`` / ``predict`` / ``get_params`` / ``set_params``
    interface as ``TorchLinearClassifier`` so it can be used as a drop-in
    replacement in the FedProTrack pipeline.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    lr : float
        Learning rate for SGD.  Default 0.01.
    n_epochs : int
        Number of epochs per ``fit`` call.  Default 5.
    batch_size : int
        Mini-batch size for SGD training.  Default 64.
    seed : int
        Random seed for reproducibility.  Default 0.
    device : torch.device or None
        Target device.  If None, auto-detect via ``get_device()``.
    """

    def __init__(
        self,
        n_classes: int,
        lr: float = 0.01,
        n_epochs: int = 5,
        batch_size: int = 64,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._seed = seed

        # Parameter count for device heuristic
        # Conv1: 3*32*3*3+32 = 896, Conv2: 32*64*3*3+64 = 18496,
        # FC1: 64*8*8*128+128 = 524416, FC2: 128*n_classes+n_classes
        n_params = 896 + 18496 + 524416 + 128 * n_classes + n_classes
        self.device = device or get_device(n_params=n_params)

        torch.manual_seed(seed)
        self._net = _SmallCNNNet(n_classes).to(self.device)
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=lr)
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on a batch of images for ``n_epochs``.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)`` with values in ``[0, 1]``.
        y : np.ndarray
            Labels of shape ``(N,)`` with integer class indices.
        """
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        n = X_t.size(0)

        self._net.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                self._optimizer.zero_grad()
                logits = self._net(X_t[idx])
                loss = nn.functional.cross_entropy(logits, y_t[idx])
                loss.backward()
                self._optimizer.step()
        self._fitted = True

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Single-epoch incremental fit.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.
        y : np.ndarray
            Labels of shape ``(N,)``.
        """
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        n = X_t.size(0)

        self._net.train()
        perm = torch.randperm(n, device=self.device)
        for start in range(0, n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            self._optimizer.zero_grad()
            logits = self._net(X_t[idx])
            loss = nn.functional.cross_entropy(logits, y_t[idx])
            loss.backward()
            self._optimizer.step()
        self._fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.

        Returns
        -------
        np.ndarray
            Predicted labels of shape ``(N,)`` with dtype int64.
        """
        if not self._fitted:
            return np.zeros(len(X), dtype=np.int64)
        X_t = to_tensor(X, device=self.device)
        self._net.eval()
        logits = self._net(X_t)
        return to_numpy(logits.argmax(dim=-1)).astype(np.int64)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.

        Returns
        -------
        np.ndarray
            Probabilities of shape ``(N, n_classes)``.
        """
        if not self._fitted:
            return np.full(
                (len(X), self.n_classes), 1.0 / self.n_classes, dtype=np.float32,
            )
        X_t = to_tensor(X, device=self.device)
        self._net.eval()
        logits = self._net(X_t)
        probs = torch.softmax(logits, dim=-1)
        return to_numpy(probs).astype(np.float32)

    @torch.no_grad()
    def predict_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean cross-entropy loss on a batch.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.
        y : np.ndarray
            Labels of shape ``(N,)``.

        Returns
        -------
        float
            Mean loss value.
        """
        if not self._fitted:
            return float("inf")
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        self._net.eval()
        logits = self._net(X_t)
        return float(nn.functional.cross_entropy(logits, y_t).item())

    # ------------------------------------------------------------------
    # Parameter I/O  (federation boundary: dict[str, np.ndarray])
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, np.ndarray]:
        """Export model parameters as numpy arrays (state_dict serialization).

        Returns
        -------
        dict[str, np.ndarray]
            Keys are ``nn.Module`` state-dict keys; values are flattened
            numpy arrays.
        """
        return {
            key: to_numpy(val).copy()
            for key, val in self._net.state_dict().items()
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Load model parameters from numpy arrays.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Must contain the same keys as ``get_params()`` output.
        """
        state = {}
        ref_state = self._net.state_dict()
        for key, ref_val in ref_state.items():
            arr = params[key]
            state[key] = torch.as_tensor(arr, dtype=ref_val.dtype).reshape(
                ref_val.shape,
            ).to(self.device)
        self._net.load_state_dict(state)
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self.lr)
        self._fitted = True

    def blend_params(
        self, other_params: dict[str, np.ndarray], alpha: float = 0.5,
    ) -> None:
        """Blend current params with external params: new = alpha * other + (1-alpha) * self.

        Parameters
        ----------
        other_params : dict[str, np.ndarray]
            External model parameters.
        alpha : float
            Weight of ``other_params``.  Default 0.5.
        """
        own = self.get_params()
        blended = {
            key: alpha * np.asarray(other_params[key], dtype=np.float64).reshape(
                own[key].shape,
            ) + (1.0 - alpha) * own[key]
            for key in own
        }
        self.set_params(blended)

    def clone_fresh(self, seed: int | None = None) -> TorchSmallCNN:
        """Create a fresh (untrained) copy with the same architecture.

        Parameters
        ----------
        seed : int or None
            New seed.  If None, uses ``self._seed + 1``.

        Returns
        -------
        TorchSmallCNN
        """
        return TorchSmallCNN(
            n_classes=self.n_classes,
            lr=self.lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            seed=seed if seed is not None else self._seed + 1,
            device=self.device,
        )


# ======================================================================
# MobileNetV2 backbone wrapper
# ======================================================================

class TorchMobileNetV2:
    """End-to-end trainable MobileNetV2 for 32x32 RGB images.

    Wraps ``torchvision.models.mobilenet_v2`` with the same federation
    interface as ``TorchSmallCNN``.  The classifier head is replaced to
    match ``n_classes``.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    pretrained : bool
        Whether to load ImageNet-pretrained weights.  Default False
        (end-to-end training from scratch, matching competitor setups).
    lr : float
        Learning rate for SGD.  Default 0.01.
    n_epochs : int
        Number of epochs per ``fit`` call.  Default 5.
    batch_size : int
        Mini-batch size for SGD training.  Default 64.
    seed : int
        Random seed.  Default 0.
    device : torch.device or None
        Target device.  If None, auto-detect via ``get_device()``.
    """

    def __init__(
        self,
        n_classes: int,
        pretrained: bool = False,
        lr: float = 0.01,
        n_epochs: int = 5,
        batch_size: int = 64,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        self.n_classes = n_classes
        self.pretrained = pretrained
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self._seed = seed

        torch.manual_seed(seed)

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self._net = mobilenet_v2(weights=weights)

        # Replace the classifier head for our number of classes
        in_features = self._net.classifier[1].in_features
        self._net.classifier[1] = nn.Linear(in_features, n_classes)

        # Adapt first conv for 32x32 images: use stride=1 instead of 2
        # to avoid shrinking the spatial dims too aggressively
        old_conv = self._net.features[0][0]
        self._net.features[0][0] = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,
            padding=old_conv.padding,
            bias=False,
        )

        n_params = sum(p.numel() for p in self._net.parameters())
        self.device = device or get_device(n_params=n_params)
        self._net = self._net.to(self.device)
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=lr)
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on a batch of images for ``n_epochs``.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)`` with values in ``[0, 1]``.
        y : np.ndarray
            Labels of shape ``(N,)``.
        """
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        n = X_t.size(0)

        self._net.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                self._optimizer.zero_grad()
                logits = self._net(X_t[idx])
                loss = nn.functional.cross_entropy(logits, y_t[idx])
                loss.backward()
                self._optimizer.step()
        self._fitted = True

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Single-epoch incremental fit.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.
        y : np.ndarray
            Labels of shape ``(N,)``.
        """
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        n = X_t.size(0)

        self._net.train()
        perm = torch.randperm(n, device=self.device)
        for start in range(0, n, self.batch_size):
            idx = perm[start : start + self.batch_size]
            self._optimizer.zero_grad()
            logits = self._net(X_t[idx])
            loss = nn.functional.cross_entropy(logits, y_t[idx])
            loss.backward()
            self._optimizer.step()
        self._fitted = True

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.

        Returns
        -------
        np.ndarray
            Predicted labels of shape ``(N,)`` with dtype int64.
        """
        if not self._fitted:
            return np.zeros(len(X), dtype=np.int64)
        X_t = to_tensor(X, device=self.device)
        self._net.eval()
        logits = self._net(X_t)
        return to_numpy(logits.argmax(dim=-1)).astype(np.int64)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.

        Returns
        -------
        np.ndarray
            Probabilities of shape ``(N, n_classes)``.
        """
        if not self._fitted:
            return np.full(
                (len(X), self.n_classes), 1.0 / self.n_classes, dtype=np.float32,
            )
        X_t = to_tensor(X, device=self.device)
        self._net.eval()
        logits = self._net(X_t)
        probs = torch.softmax(logits, dim=-1)
        return to_numpy(probs).astype(np.float32)

    @torch.no_grad()
    def predict_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean cross-entropy loss on a batch.

        Parameters
        ----------
        X : np.ndarray
            Images of shape ``(N, 3, 32, 32)``.
        y : np.ndarray
            Labels of shape ``(N,)``.

        Returns
        -------
        float
            Mean loss value.
        """
        if not self._fitted:
            return float("inf")
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        self._net.eval()
        logits = self._net(X_t)
        return float(nn.functional.cross_entropy(logits, y_t).item())

    # ------------------------------------------------------------------
    # Parameter I/O  (federation boundary: dict[str, np.ndarray])
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, np.ndarray]:
        """Export model parameters as numpy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are ``nn.Module`` state-dict keys; values are numpy arrays.
        """
        return {
            key: to_numpy(val).copy()
            for key, val in self._net.state_dict().items()
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Load model parameters from numpy arrays.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Must contain the same keys as ``get_params()`` output.
        """
        state = {}
        ref_state = self._net.state_dict()
        for key, ref_val in ref_state.items():
            arr = params[key]
            state[key] = torch.as_tensor(arr, dtype=ref_val.dtype).reshape(
                ref_val.shape,
            ).to(self.device)
        self._net.load_state_dict(state)
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self.lr)
        self._fitted = True

    def blend_params(
        self, other_params: dict[str, np.ndarray], alpha: float = 0.5,
    ) -> None:
        """Blend current params with external params: new = alpha * other + (1-alpha) * self.

        Parameters
        ----------
        other_params : dict[str, np.ndarray]
            External model parameters.
        alpha : float
            Weight of ``other_params``.  Default 0.5.
        """
        own = self.get_params()
        blended = {
            key: alpha * np.asarray(other_params[key], dtype=np.float64).reshape(
                own[key].shape,
            ) + (1.0 - alpha) * own[key]
            for key in own
        }
        self.set_params(blended)

    def clone_fresh(self, seed: int | None = None) -> TorchMobileNetV2:
        """Create a fresh (untrained) copy with the same architecture.

        Parameters
        ----------
        seed : int or None
            New seed.  If None, uses ``self._seed + 1``.

        Returns
        -------
        TorchMobileNetV2
        """
        return TorchMobileNetV2(
            n_classes=self.n_classes,
            pretrained=self.pretrained,
            lr=self.lr,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            seed=seed if seed is not None else self._seed + 1,
            device=self.device,
        )
