from __future__ import annotations

"""FedProx baseline: FedAvg with proximal regularisation.

FedProx (Li et al., MLSys 2020) adds a proximal term to the local
objective: ``mu / 2 * ||w - w_global||^2``, which limits the amount
each client's local model can drift from the latest global model.
This is a standard personalised-FL baseline used across many papers
including FedCCFA.

Communication budget:
  - Upload: model params (same as FedAvg).
  - Download: global model to all clients.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..device import get_device, to_numpy, to_tensor
from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Deep-copy a parameter dict."""
    return {k: v.copy() for k, v in params.items()}


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _extract_dims(dataset: DriftDataset) -> tuple[int, int, int, int]:
    """Return (K, T, n_features, n_classes) from a DriftDataset."""
    K = dataset.config.K
    T = dataset.config.T
    X0, _ = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        labels.update(int(v) for v in np.unique(y))
    n_classes = max(labels) + 1
    return K, T, n_features, n_classes


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FedProxUpload:
    """Data uploaded by one FedProx client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    model_params : dict[str, np.ndarray]
        Locally updated model parameters.
    n_samples : int
        Number of training samples seen so far.
    """

    client_id: int
    model_params: dict[str, np.ndarray]
    n_samples: int


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FedProxClient:
    """Client for the FedProx baseline.

    Local training minimises:
        ``CE(w; D_local) + mu / 2 * ||w - w_global||^2``

    Parameters
    ----------
    client_id : int
        Unique identifier.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    mu : float
        Proximal regularisation strength.  Default 0.01.
    lr : float
        Local SGD learning rate.  Default 0.01.
    n_epochs : int
        Local training epochs per round.  Default 5.
    seed : int
        Random seed.  Default 0.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        mu: float = 0.01,
        lr: float = 0.01,
        n_epochs: int = 5,
        seed: int = 0,
    ) -> None:
        if mu < 0.0:
            raise ValueError(f"mu must be >= 0, got {mu}")

        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.mu = mu
        self._lr = lr
        self._n_epochs = n_epochs
        self._seed = seed

        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._global_params: dict[str, np.ndarray] = {}
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Training with proximal term
    # ------------------------------------------------------------------

    def _fit_with_prox(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train with proximal regularisation.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)
        """
        device = self._model.device
        X_t = to_tensor(X, device=device)
        y_t = to_tensor(y, dtype=torch.long, device=device)

        linear = self._model._linear
        n_out = self._model._n_out
        optimizer = self._model._optimizer

        # Snapshot of the global model weights for the proximal term
        global_weight = to_tensor(
            self._global_params["coef"].reshape(n_out, self.n_features),
            device=device,
        )
        global_bias = to_tensor(
            self._global_params["intercept"].reshape(n_out),
            device=device,
        )

        linear.train()
        for _ in range(self._n_epochs):
            optimizer.zero_grad()
            logits = linear(X_t)

            # Classification loss
            if n_out == 1:
                cls_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), y_t.float(),
                )
            else:
                cls_loss = nn.functional.cross_entropy(logits, y_t)

            # Proximal term: mu/2 * ||w - w_global||^2
            prox_loss = (
                torch.sum((linear.weight - global_weight) ** 2)
                + torch.sum((linear.bias - global_bias) ** 2)
            )

            loss = cls_loss + (self.mu / 2.0) * prox_loss
            loss.backward()
            optimizer.step()

        self._model._fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on a local batch.

        If global params are available and mu > 0, adds proximal
        regularisation.  Otherwise falls back to standard training.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)
        """
        self._n_samples += len(X)

        if self.mu > 0.0 and self._global_params:
            self._fit_with_prox(X, y)
        else:
            self._model.fit(X, y)

        self._model_params = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n,)
        """
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Federation interface
    # ------------------------------------------------------------------

    def set_global_params(self, params: dict[str, np.ndarray]) -> None:
        """Receive global model from the server.

        Parameters
        ----------
        params : dict[str, np.ndarray]
        """
        self._global_params = _copy_params(params)
        self._model.set_params(params)
        self._model_params = self._model.get_params()

    def get_upload(self) -> FedProxUpload:
        """Package local state for upload.

        Returns
        -------
        FedProxUpload
        """
        return FedProxUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Communication cost for one upload.

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        return model_bytes(self._model_params, precision_bits)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FedProxServer:
    """Server for FedProx: weighted FedAvg aggregation.

    Parameters
    ----------
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    seed : int
        Random seed.  Default 42.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed

        rng = np.random.RandomState(seed)
        n_out = 1 if n_classes == 2 else n_classes
        self.global_params: dict[str, np.ndarray] = {
            "coef": (rng.randn(n_out * n_features) * 0.01).astype(np.float64),
            "intercept": np.zeros(n_out, dtype=np.float64),
        }

    def aggregate(self, uploads: list[FedProxUpload]) -> dict[str, np.ndarray]:
        """Weighted average of client models (same as FedAvg).

        Parameters
        ----------
        uploads : list[FedProxUpload]

        Returns
        -------
        dict[str, np.ndarray]
            Updated global model.
        """
        if not uploads:
            return _copy_params(self.global_params)

        valid = [u for u in uploads if u.model_params]
        if not valid:
            return _copy_params(self.global_params)

        weights = [float(max(u.n_samples, 1)) for u in valid]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0] * len(valid)
            total_w = float(len(valid))

        agg: dict[str, np.ndarray] = {}
        for key in valid[0].model_params:
            stacked = np.stack([u.model_params[key] for u in valid])
            w_arr = np.array(weights, dtype=np.float64) / total_w
            agg[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))

        self.global_params = _copy_params(agg)
        return _copy_params(self.global_params)

    def download_bytes(self, n_clients: int, precision_bits: int = 32) -> float:
        """Communication cost for broadcasting global model.

        Parameters
        ----------
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        return float(n_clients) * model_bytes(self.global_params, precision_bits)


# ---------------------------------------------------------------------------
# Full runner
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """Unified result container (matches runners.MethodResult).

    Parameters
    ----------
    method_name : str
    accuracy_matrix : np.ndarray
    predicted_concept_matrix : np.ndarray
    total_bytes : float
    """

    method_name: str
    accuracy_matrix: np.ndarray
    predicted_concept_matrix: np.ndarray
    total_bytes: float

    def to_experiment_log(self, ground_truth: np.ndarray) -> ExperimentLog:
        """Convert to ExperimentLog for unified metric computation.

        Parameters
        ----------
        ground_truth : np.ndarray

        Returns
        -------
        ExperimentLog
        """
        return ExperimentLog(
            ground_truth=ground_truth,
            predicted=self.predicted_concept_matrix,
            accuracy_curve=self.accuracy_matrix,
            total_bytes=self.total_bytes,
            method_name=self.method_name,
        )


def run_fedprox_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    mu: float = 0.01,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run FedProx end-to-end on a DriftDataset.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
        Federation period.
    mu : float
        Proximal regularisation strength.
    lr : float
        Local learning rate.
    n_epochs : int
        Local epochs.

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FedProxClient(
            k, n_features, n_classes,
            mu=mu, lr=lr, n_epochs=n_epochs,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = FedProxServer(n_features=n_features, n_classes=n_classes, seed=42)

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_concept_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            accuracy_matrix[k, t] = _accuracy(y, clients[k].predict(X))

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            global_params = server.aggregate(uploads)
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b
            for c in clients:
                c.set_global_params(global_params)

    return MethodResult(
        method_name="FedProx",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
