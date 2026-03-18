from __future__ import annotations

"""Lightweight APFL baseline adapted to the repo's linear classifier stack."""

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
class APFLUpload:
    client_id: int
    global_params: dict[str, np.ndarray]
    alpha: float
    n_samples: int


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _blend_params(
    global_params: dict[str, np.ndarray],
    local_params: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    return {
        key: alpha * local_params[key] + (1.0 - alpha) * global_params[key]
        for key in global_params
    }


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


class APFLClient:
    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        alpha: float = 0.5,
        alpha_lr: float = 0.05,
        local_steps: int = 2,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.alpha_lr = alpha_lr
        self.local_steps = max(1, local_steps)
        self._seed = seed

        self._global_model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=0.05,
            n_epochs=1,
            seed=seed,
        )
        self._local_model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=0.05,
            n_epochs=1,
            seed=seed + 17,
        )
        self._personal_model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=0.05,
            n_epochs=1,
            seed=seed + 29,
        )
        init_params = self._global_model.get_params()
        self._local_model.set_params(init_params)
        self._personal_model.set_params(init_params)
        self._global_params: dict[str, np.ndarray] = _copy_params(init_params)
        self._local_params: dict[str, np.ndarray] = _copy_params(init_params)
        self._personal_params: dict[str, np.ndarray] = _copy_params(init_params)
        self._alpha = float(np.clip(alpha, 0.0, 1.0))
        self._n_samples = 0

    @property
    def alpha(self) -> float:
        return float(self._alpha)

    def _refresh_personal_model(self) -> None:
        if self._global_params and self._local_params:
            params = _blend_params(self._global_params, self._local_params, self._alpha)
        elif self._global_params:
            params = _copy_params(self._global_params)
        elif self._local_params:
            params = _copy_params(self._local_params)
        else:
            return
        self._personal_params = _copy_params(params)
        self._personal_model.set_params(params)

    def set_global_params(self, params: dict[str, np.ndarray]) -> None:
        self._global_params = _copy_params(params)
        if params:
            self._global_model.set_params(params)
            self._refresh_personal_model()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_samples += len(X)

        if self._global_params:
            self._global_model.set_params(self._global_params)

        for _ in range(self.local_steps):
            self._global_model.partial_fit(X, y)
            self._local_model.partial_fit(X, y)

        self._global_params = self._global_model.get_params()
        self._local_params = self._local_model.get_params()

        global_loss = self._global_model.predict_loss(X, y)
        local_loss = self._local_model.predict_loss(X, y)
        if np.isfinite(global_loss) and np.isfinite(local_loss):
            self._alpha = float(
                np.clip(self._alpha + self.alpha_lr * (global_loss - local_loss), 0.0, 1.0)
            )

        self._refresh_personal_model()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._personal_model.predict(X)

    def get_upload(self) -> APFLUpload:
        return APFLUpload(
            client_id=self.client_id,
            global_params=_copy_params(self._global_params),
            alpha=self._alpha,
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        return model_bytes(self._global_params, precision_bits=precision_bits) + precision_bits / 8


class APFLServer:
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

    def aggregate(self, uploads: list[APFLUpload]) -> dict[str, np.ndarray]:
        if not uploads:
            return _copy_params(self.global_params)

        total = sum(max(1, u.n_samples) for u in uploads if u.global_params)
        if total <= 0:
            return _copy_params(self.global_params)

        keys = list(uploads[0].global_params.keys())
        aggregated: dict[str, np.ndarray] = {}
        for key in keys:
            acc = None
            for upload in uploads:
                if key not in upload.global_params:
                    continue
                weight = max(1, upload.n_samples)
                contrib = upload.global_params[key] * weight
                acc = contrib if acc is None else acc + contrib
            if acc is not None:
                aggregated[key] = acc / total

        if aggregated:
            self.global_params = _copy_params(aggregated)
        return _copy_params(self.global_params)

    def download_bytes(self, n_clients: int, precision_bits: int = 32) -> float:
        return float(n_clients) * model_bytes(self.global_params, precision_bits=precision_bits)


def run_apfl_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    alpha: float = 0.5,
    alpha_lr: float = 0.05,
    local_steps: int = 2,
) -> MethodResult:
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        APFLClient(
            k,
            n_features,
            n_classes,
            alpha=alpha,
            alpha_lr=alpha_lr,
            local_steps=local_steps,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = APFLServer(n_features=n_features, n_classes=n_classes, seed=42)

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
            uploads = [client.get_upload() for client in clients]
            upload_bytes = sum(client.upload_bytes() for client in clients)
            global_params = server.aggregate(uploads)
            download_bytes = server.download_bytes(K)
            total_bytes += upload_bytes + download_bytes
            for client in clients:
                client.set_global_params(global_params)

    return MethodResult(
        method_name="APFL",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
