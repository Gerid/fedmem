from __future__ import annotations

"""PyTorch linear classifier — GPU-accelerated drop-in for sklearn SGD/LogReg.

Provides the same parameter-dict interface (``{"coef": ndarray, "intercept": ndarray}``)
used by the federation layer so that aggregators and comm_tracker remain unchanged.
"""

import numpy as np
import torch
import torch.nn as nn

from ..device import get_device, to_numpy, to_tensor


class TorchLinearClassifier:
    """GPU-accelerated logistic regression with SGD.

    Replaces sklearn's ``SGDClassifier`` / ``LogisticRegression`` for
    local client training and prediction, running on the best available
    device (CUDA when present).

    Parameters
    ----------
    n_features : int
        Input feature dimensionality.
    n_classes : int
        Number of output classes.
    lr : float
        Learning rate for SGD. Default 0.1.
    n_epochs : int
        Number of epochs per ``fit`` / ``partial_fit`` call. Default 5.
    seed : int
        Random seed for reproducibility. Default 0.
    device : torch.device or None
        Device to run on. If None, auto-detect via ``get_device()``.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        lr: float = 0.1,
        n_epochs: int = 5,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self._seed = seed
        self.device = device or get_device()

        torch.manual_seed(seed)

        # Binary classification uses 1 output row, multi-class uses n_classes
        self._n_out = 1 if n_classes == 2 else n_classes
        self._linear = nn.Linear(n_features, self._n_out).to(self.device)
        self._optimizer = torch.optim.SGD(self._linear.parameters(), lr=lr)
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on a batch (full re-train from current weights).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)

        self._linear.train()
        for _ in range(self.n_epochs):
            self._optimizer.zero_grad()
            logits = self._linear(X_t)
            if self._n_out == 1:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), y_t.float(),
                )
            else:
                loss = nn.functional.cross_entropy(logits, y_t)
            loss.backward()
            self._optimizer.step()
        self._fitted = True

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Single-epoch incremental fit (like sklearn's partial_fit).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)

        self._linear.train()
        self._optimizer.zero_grad()
        logits = self._linear(X_t)
        if self._n_out == 1:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), y_t.float(),
            )
        else:
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
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,) with dtype int64
        """
        if not self._fitted:
            return np.zeros(len(X), dtype=np.int64)

        X_t = to_tensor(X, device=self.device)
        self._linear.eval()
        logits = self._linear(X_t)

        if self._n_out == 1:
            preds = (logits.squeeze(-1) > 0).long()
        else:
            preds = logits.argmax(dim=-1)

        return to_numpy(preds).astype(np.int64)

    @torch.no_grad()
    def predict_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean loss on a batch (for model selection like IFCA).

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        float
            Mean loss value.
        """
        if not self._fitted:
            return float("inf")

        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)

        self._linear.eval()
        logits = self._linear(X_t)

        if self._n_out == 1:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), y_t.float(),
            )
        else:
            loss = nn.functional.cross_entropy(logits, y_t)
        return float(loss.item())

    # ------------------------------------------------------------------
    # Parameter I/O  (federation boundary: dict[str, np.ndarray])
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, np.ndarray]:
        """Export model parameters as numpy arrays.

        Returns
        -------
        dict[str, np.ndarray]
            ``{"coef": shape (n_out * n_features,),
              "intercept": shape (n_out,)}``.
        """
        coef = to_numpy(self._linear.weight.data).flatten().copy()
        intercept = to_numpy(self._linear.bias.data).flatten().copy()
        return {"coef": coef, "intercept": intercept}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Load model parameters from numpy arrays.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Must contain ``"coef"`` and ``"intercept"``.
        """
        coef = params["coef"].reshape(self._n_out, self.n_features)
        intercept = params["intercept"].reshape(self._n_out)

        with torch.no_grad():
            self._linear.weight.copy_(to_tensor(coef, device=self.device))
            self._linear.bias.copy_(to_tensor(intercept, device=self.device))
        self._fitted = True

        # Rebuild optimizer so it tracks the correct parameter tensors
        self._optimizer = torch.optim.SGD(
            self._linear.parameters(), lr=self.lr,
        )

    def blend_params(self, other_params: dict[str, np.ndarray], alpha: float = 0.5) -> None:
        """Blend current params with external params: new = alpha * other + (1-alpha) * self.

        Parameters
        ----------
        other_params : dict[str, np.ndarray]
        alpha : float
            Weight of other_params. Default 0.5.
        """
        own = self.get_params()
        blended = {
            k: alpha * other_params[k] + (1 - alpha) * own[k]
            for k in own
        }
        self.set_params(blended)

    @property
    def coef_(self) -> np.ndarray:
        """Sklearn-compatible coef_ accessor (numpy, on CPU)."""
        return to_numpy(self._linear.weight.data).copy()

    @property
    def intercept_(self) -> np.ndarray:
        """Sklearn-compatible intercept_ accessor (numpy, on CPU)."""
        return to_numpy(self._linear.bias.data).copy()

    def clone_fresh(self, seed: int | None = None) -> TorchLinearClassifier:
        """Create a fresh (untrained) copy with same architecture.

        Parameters
        ----------
        seed : int or None
            New seed. If None, uses ``self._seed + 1``.

        Returns
        -------
        TorchLinearClassifier
        """
        return TorchLinearClassifier(
            n_features=self.n_features,
            n_classes=self.n_classes,
            lr=self.lr,
            n_epochs=self.n_epochs,
            seed=seed if seed is not None else self._seed + 1,
            device=self.device,
        )
