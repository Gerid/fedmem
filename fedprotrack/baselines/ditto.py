from __future__ import annotations

"""Ditto baseline: personalised FL via global+local models with proximal regularisation.

Ditto (Li et al., ICML 2021) maintains two models per client:
  - Global model w: participates in standard FedAvg aggregation.
  - Personalized model v: trained locally with proximal term lambda/2 ||v - w||^2.
Inference uses the personalized model.

Communication budget: same as FedAvg (only global model uploaded/downloaded).
"""

from dataclasses import dataclass

import numpy as np

from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


@dataclass
class MethodResult:
    method_name: str
    accuracy_matrix: np.ndarray
    predicted_concept_matrix: np.ndarray
    total_bytes: float

    def to_experiment_log(self, ground_truth: np.ndarray) -> ExperimentLog:
        return ExperimentLog(
            ground_truth=ground_truth,
            predicted=self.predicted_concept_matrix,
            accuracy_curve=self.accuracy_matrix,
            total_bytes=self.total_bytes,
            method_name=self.method_name,
        )


@dataclass
class DittoUpload:
    client_id: int
    model_params: dict[str, np.ndarray]
    n_samples: int


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _extract_dims(dataset: DriftDataset) -> tuple[int, int, int, int]:
    K = dataset.config.K
    T = dataset.config.T
    X0, _ = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        labels.update(int(v) for v in np.unique(y))
    n_classes = max(labels) + 1
    return K, T, n_features, n_classes


class DittoClient:
    """Per-client Ditto logic with global + personalized models.

    Parameters
    ----------
    client_id : int
        Unique client identifier.
    n_features : int
        Input feature dimensionality.
    n_classes : int
        Number of output classes.
    lamda : float
        Proximal regularisation weight tying the personal model to the
        global model.  Default 0.1.
    tau : int
        Number of personalisation epochs for the personal model per round.
        Default 5.
    lr : float
        Learning rate for both global and personal models.  Default 0.01.
    n_epochs : int
        Number of local epochs for the global model per round.  Default 5.
    seed : int
        Random seed for reproducibility.  Default 0.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        lamda: float = 0.1,
        tau: int = 5,
        lr: float = 0.01,
        n_epochs: int = 5,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.lamda = lamda
        self.tau = max(1, tau)
        self.lr = lr
        self.n_epochs = n_epochs
        self._seed = seed
        self._n_samples = 0

        self._global_model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._personal_model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=1,
            seed=seed + 17,
        )

    def set_global_params(self, params: dict[str, np.ndarray]) -> None:
        """Set global model parameters (received from server).

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Aggregated global model parameters.
        """
        self._global_model.set_params(_copy_params(params))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Local training round: update global model, then train personal model.

        Steps:
        1. Train the global model on (X, y) for ``n_epochs`` epochs.
        2. Copy global params to the personal model as the starting point.
        3. Train the personal model for ``tau`` epochs with proximal
           regularisation: after each epoch, apply
           ``v[k] -= lr * lamda * (v[k] - w[k])`` for each parameter key.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        self._n_samples += len(X)

        # Step 1: train global model
        self._global_model.fit(X, y)

        # Step 2: copy global params to personal model as starting point
        global_params = self._global_model.get_params()
        self._personal_model.set_params(_copy_params(global_params))

        # Step 3: train personal model with proximal regularisation
        for _ in range(self.tau):
            self._personal_model.fit(X, y)
            # Proximal step: blend personal model back toward global
            personal_params = self._personal_model.get_params()
            for key in personal_params:
                personal_params[key] = (
                    personal_params[key]
                    - self.lr * self.lamda * (personal_params[key] - global_params[key])
                )
            self._personal_model.set_params(personal_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the personalized model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self._personal_model.predict(X)

    def get_upload(self) -> DittoUpload:
        """Prepare upload for federation (global model only).

        Returns
        -------
        DittoUpload
        """
        return DittoUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._global_model.get_params()),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute upload size in bytes (global model only).

        Parameters
        ----------
        precision_bits : int
            Bit-width per scalar element.

        Returns
        -------
        float
        """
        return model_bytes(self._global_model.get_params(), precision_bits=precision_bits)


class DittoServer:
    """Server-side aggregation for Ditto — standard FedAvg weighted average.

    Parameters
    ----------
    n_features : int
        Input feature dimensionality.
    n_classes : int
        Number of output classes.
    seed : int
        Random seed for parameter initialisation.
    """

    def __init__(self, n_features: int, n_classes: int, seed: int = 0) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed
        self.global_params = self._init_params(seed)

    def _init_params(self, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(seed)
        n_out = 1 if self.n_classes == 2 else self.n_classes
        return {
            "coef": (rng.randn(n_out * self.n_features) * 0.01).astype(np.float64),
            "intercept": np.zeros(n_out, dtype=np.float64),
        }

    def aggregate(self, uploads: list[DittoUpload]) -> dict[str, np.ndarray]:
        """Weighted-average aggregation of client global model parameters.

        Parameters
        ----------
        uploads : list[DittoUpload]
            Client uploads containing global model params and sample counts.

        Returns
        -------
        dict[str, np.ndarray]
            Aggregated global model parameters.
        """
        if not uploads:
            return _copy_params(self.global_params)

        total = sum(max(1, u.n_samples) for u in uploads if u.model_params)
        if total <= 0:
            return _copy_params(self.global_params)

        keys = list(uploads[0].model_params.keys())
        aggregated: dict[str, np.ndarray] = {}
        for key in keys:
            acc = None
            for upload in uploads:
                if key not in upload.model_params:
                    continue
                weight = max(1, upload.n_samples)
                contrib = upload.model_params[key] * weight
                acc = contrib if acc is None else acc + contrib
            if acc is not None:
                aggregated[key] = acc / total

        if aggregated:
            self.global_params = _copy_params(aggregated)
        return _copy_params(self.global_params)

    def download_bytes(self, n_clients: int, precision_bits: int = 32) -> float:
        """Compute total download bytes for broadcasting to all clients.

        Parameters
        ----------
        n_clients : int
            Number of clients receiving the global model.
        precision_bits : int
            Bit-width per scalar element.

        Returns
        -------
        float
        """
        return float(n_clients) * model_bytes(self.global_params, precision_bits=precision_bits)


def run_ditto_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    lamda: float = 0.1,
    tau: int = 5,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run Ditto on a full DriftDataset.

    Parameters
    ----------
    dataset : DriftDataset
        Synthetic or real federated dataset.
    federation_every : int
        Federation frequency (every N rounds).
    lamda : float
        Proximal regularisation weight for personalization.
    tau : int
        Number of personalisation epochs per round.
    lr : float
        Learning rate for local models.
    n_epochs : int
        Number of local epochs for the global model per round.

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        DittoClient(
            k,
            n_features,
            n_classes,
            lamda=lamda,
            tau=tau,
            lr=lr,
            n_epochs=n_epochs,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = DittoServer(n_features=n_features, n_classes=n_classes, seed=42)

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_concept_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        # Evaluate
        for k in range(K):
            X, y = dataset.eval_batch(k, t)
            accuracy_matrix[k, t] = _accuracy(y, clients[k].predict(X))

        # Train
        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        # Federate
        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [client.get_upload() for client in clients]
            upload_bytes = sum(client.upload_bytes() for client in clients)
            global_params = server.aggregate(uploads)
            download_bytes = server.download_bytes(K)
            total_bytes += upload_bytes + download_bytes
            for client in clients:
                client.set_global_params(global_params)

    return MethodResult(
        method_name="Ditto",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
