from __future__ import annotations

"""Lightweight pFedMe baseline adapted to TorchLinearClassifier + DriftDataset."""

from dataclasses import dataclass

import numpy as np

from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from ..models import TorchLinearClassifier
from ..models.factory import create_model
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
class PFedMeUpload:
    client_id: int
    model_params: dict[str, np.ndarray]
    personalized_params: dict[str, np.ndarray]
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


class PFedMeClient:
    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        local_epochs: int = 3,
        K: int = 5,
        lamda: float = 0.1,
        personal_learning_rate: float = 0.05,
        seed: int = 0,
        model_type: str = "linear",
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.local_epochs = local_epochs
        self.K = max(1, K)
        self.lamda = lamda
        self.personal_learning_rate = personal_learning_rate
        self._seed = seed
        self._model_type = model_type

        self._model = create_model(
            model_type,
            n_features,
            n_classes,
            lr=personal_learning_rate,
            n_epochs=local_epochs,
            seed=seed,
        )
        self._global_params: dict[str, np.ndarray] = {}
        self._model_params: dict[str, np.ndarray] = {}
        self._personalized_params: dict[str, np.ndarray] = {}
        self._n_samples = 0

    def set_global_params(self, params: dict[str, np.ndarray]) -> None:
        self._global_params = _copy_params(params)
        if params and not self._model_params:
            self._model.set_params(params)
            self._model_params = _copy_params(params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_samples += len(X)

        base_params = (
            _copy_params(self._model_params)
            if self._model_params
            else (_copy_params(self._global_params) if self._global_params else {})
        )

        personalized = create_model(
            self._model_type,
            self.n_features,
            self.n_classes,
            lr=self.personal_learning_rate,
            n_epochs=self.local_epochs,
            seed=self._seed + self.client_id + self._n_samples,
        )
        if base_params:
            personalized.set_params(base_params)

        for _ in range(self.K):
            personalized.fit(X, y)

        self._personalized_params = personalized.get_params()
        if not base_params:
            self._model.set_params(self._personalized_params)
            self._model_params = self._model.get_params()
            return

        updated = {
            key: base_params[key]
            - self.lamda * self.personal_learning_rate
            * (base_params[key] - self._personalized_params[key])
            for key in base_params
        }
        self._model.set_params(updated)
        self._model_params = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_upload(self) -> PFedMeUpload:
        return PFedMeUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            personalized_params=_copy_params(self._personalized_params),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        return model_bytes(self._model_params, precision_bits=precision_bits)


class PFedMeServer:
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

    def aggregate(self, uploads: list[PFedMeUpload]) -> dict[str, np.ndarray]:
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
        return float(n_clients) * model_bytes(self.global_params, precision_bits=precision_bits)


def run_pfedme_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    local_epochs: int = 3,
    K_steps: int = 5,
    lamda: float = 0.1,
    personal_learning_rate: float = 0.05,
    model_type: str = "linear",
) -> MethodResult:
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        PFedMeClient(
            k,
            n_features,
            n_classes,
            local_epochs=local_epochs,
            K=K_steps,
            lamda=lamda,
            personal_learning_rate=personal_learning_rate,
            seed=42 + k,
            model_type=model_type,
        )
        for k in range(K)
    ]
    server = PFedMeServer(n_features=n_features, n_classes=n_classes, seed=42)

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_concept_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.eval_batch(k, t)
            accuracy_matrix[k, t] = _accuracy(y, clients[k].predict(X))

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [client.get_upload() for client in clients]
            upload_bytes = sum(client.upload_bytes() for client in clients)
            global_params = server.aggregate(uploads)
            download_bytes = server.download_bytes(K)
            total_bytes += upload_bytes + download_bytes
            for client in clients:
                client.set_global_params(global_params)

    return MethodResult(
        method_name="pFedMe",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
