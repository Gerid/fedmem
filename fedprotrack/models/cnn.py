from __future__ import annotations

"""PyTorch small CNN classifier for 32x32 images (e.g. CIFAR-100).

Provides the same ``fit / predict / get_params / set_params`` interface as
``TorchLinearClassifier`` so that it can be used as a drop-in replacement
in all baseline runners.
"""

import numpy as np
import torch
import torch.nn as nn

from ..device import get_device, to_numpy, to_tensor


class _SmallCNNNet(nn.Module):
    """2-conv CNN for 32x32 images (~545K params at n_classes=100)."""

    def __init__(self, input_channels: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After two 2x2 max-pools on 32x32 input: 64 * 8 * 8 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TorchSmallCNN:
    """GPU-accelerated 2-conv CNN for 32x32 image classification.

    Provides the same parameter-dict interface (``get_params / set_params``)
    as ``TorchLinearClassifier`` so that federation aggregators work
    transparently.

    Parameters
    ----------
    n_features : int
        Ignored for CNN (kept for API compatibility with the factory).
    n_classes : int
        Number of output classes.
    lr : float
        Learning rate for SGD. Default 0.01.
    n_epochs : int
        Number of epochs per ``fit`` call. Default 5.
    seed : int
        Random seed for reproducibility. Default 0.
    device : torch.device or None
        Device to run on. If None, auto-detect via ``get_device()``.
    input_channels : int
        Number of input channels (3 for RGB). Default 3.
    image_size : int
        Spatial size of the input (height = width). Default 32.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        lr: float = 0.01,
        n_epochs: int = 5,
        seed: int = 0,
        device: torch.device | None = None,
        input_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self._seed = seed
        self.input_channels = input_channels
        self.image_size = image_size
        self.device = device or get_device(n_params=600_000)

        torch.manual_seed(seed)

        self._net = _SmallCNNNet(input_channels, n_classes).to(self.device)
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=lr)
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _to_image_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Reshape flat or image arrays into (N, C, H, W) tensors."""
        X_t = to_tensor(X, device=self.device)
        if X_t.ndim == 2:
            # Flat features: reshape to (N, C, H, W)
            X_t = X_t.view(-1, self.input_channels, self.image_size, self.image_size)
        elif X_t.ndim == 3:
            # (N, H, W) -- single-channel, add channel dim
            X_t = X_t.unsqueeze(1)
        return X_t

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on a batch (full re-train from current weights).

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, C*H*W) or (n_samples, C, H, W).
        y : np.ndarray of shape (n_samples,)
        """
        X_t = self._to_image_tensor(X)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)

        self._net.train()
        for _ in range(self.n_epochs):
            self._optimizer.zero_grad()
            logits = self._net(X_t)
            loss = nn.functional.cross_entropy(logits, y_t)
            loss.backward()
            self._optimizer.step()
        self._fitted = True

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Single-epoch incremental fit.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray
        """
        X_t = self._to_image_tensor(X)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)

        self._net.train()
        self._optimizer.zero_grad()
        logits = self._net(X_t)
        loss = nn.functional.cross_entropy(logits, y_t)
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

        Returns
        -------
        np.ndarray of shape (n_samples,) with dtype int64
        """
        if not self._fitted:
            return np.zeros(len(X), dtype=np.int64)

        X_t = self._to_image_tensor(X)
        self._net.eval()
        logits = self._net(X_t)
        preds = logits.argmax(dim=-1)
        return to_numpy(preds).astype(np.int64)

    @torch.no_grad()
    def predict_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean loss on a batch.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        float
        """
        if not self._fitted:
            return float("inf")

        X_t = self._to_image_tensor(X)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)

        self._net.eval()
        logits = self._net(X_t)
        loss = nn.functional.cross_entropy(logits, y_t)
        return float(loss.item())

    # ------------------------------------------------------------------
    # Parameter I/O  (federation boundary: dict[str, np.ndarray])
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, np.ndarray]:
        """Export all model parameters as a flat dict of numpy arrays.

        Returns
        -------
        dict[str, np.ndarray]
        """
        params: dict[str, np.ndarray] = {}
        for name, param in self._net.named_parameters():
            params[name] = to_numpy(param.data).flatten().copy()
        return params

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Load model parameters from numpy arrays.

        Parameters
        ----------
        params : dict[str, np.ndarray]
        """
        with torch.no_grad():
            for name, param in self._net.named_parameters():
                if name in params:
                    arr = params[name].reshape(param.shape)
                    param.copy_(to_tensor(arr, device=self.device))
        self._fitted = True

        # Rebuild optimizer so it tracks the correct parameter tensors
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self.lr)

    def blend_params(self, other_params: dict[str, np.ndarray], alpha: float = 0.5) -> None:
        """Blend current params with external params: new = alpha * other + (1-alpha) * self.

        Parameters
        ----------
        other_params : dict[str, np.ndarray]
        alpha : float
        """
        own = self.get_params()
        blended = {
            k: alpha * np.asarray(other_params[k], dtype=np.float64).reshape(own[k].shape)
            + (1 - alpha) * own[k]
            for k in own
        }
        self.set_params(blended)

    def clone_fresh(self, seed: int | None = None) -> TorchSmallCNN:
        """Create a fresh (untrained) copy with same architecture.

        Parameters
        ----------
        seed : int or None

        Returns
        -------
        TorchSmallCNN
        """
        return TorchSmallCNN(
            n_features=self.n_features,
            n_classes=self.n_classes,
            lr=self.lr,
            n_epochs=self.n_epochs,
            seed=seed if seed is not None else self._seed + 1,
            device=self.device,
            input_channels=self.input_channels,
            image_size=self.image_size,
        )
