from __future__ import annotations

"""Lightweight FedEM baseline adapted to TorchLinearClassifier."""

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
class FedEMUpload:
    client_id: int
    expert_params: list[dict[str, np.ndarray]]
    responsibilities: np.ndarray
    n_samples: int


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).any():
        return np.ones_like(x, dtype=np.float64) / max(1, len(x))
    x = x - np.max(x)
    exp = np.exp(np.clip(x, -50.0, 50.0))
    denom = exp.sum()
    if denom <= 0:
        return np.ones_like(x, dtype=np.float64) / max(1, len(x))
    return exp / denom


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


class FedEMClient:
    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        n_components: int = 3,
        lr: float = 0.05,
        local_epochs: int = 2,
        seed: int = 0,
        model_type: str = "linear",
    ) -> None:
        if n_components <= 0:
            raise ValueError(f"n_components must be > 0, got {n_components}")
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_components = n_components
        self.lr = lr
        self.local_epochs = local_epochs
        self._seed = seed
        self._model_type = model_type

        self._experts = [
            create_model(
                model_type,
                n_features,
                n_classes,
                lr=lr,
                n_epochs=1,
                seed=seed + 31 * i,
            )
            for i in range(n_components)
        ]
        self._global_expert_params: list[dict[str, np.ndarray]] = []
        self._expert_params: list[dict[str, np.ndarray]] = []
        self._responsibilities = np.ones(n_components, dtype=np.float64) / n_components
        self._selected_expert = 0
        self._n_samples = 0

    @property
    def selected_expert_id(self) -> int:
        return int(self._selected_expert)

    def set_global_experts(self, expert_params: list[dict[str, np.ndarray]]) -> None:
        self._global_expert_params = [_copy_params(p) for p in expert_params]
        for expert, params in zip(self._experts, self._global_expert_params):
            if params:
                expert.set_params(params)

    def _current_params(self) -> list[dict[str, np.ndarray]]:
        return [expert.get_params() for expert in self._experts]

    def _expert_loss(self, expert, X: np.ndarray, y: np.ndarray) -> float:
        if expert._fitted:
            return expert.predict_loss(X, y)

        temp = create_model(
            self._model_type,
            self.n_features,
            self.n_classes,
            lr=self.lr,
            n_epochs=1,
            seed=self._seed,
        )
        temp.set_params(expert.get_params())
        return temp.predict_loss(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._experts:
            return np.zeros(len(X), dtype=np.int64)
        votes = np.zeros((len(X), self.n_classes), dtype=np.float64)
        for weight, expert in zip(self._responsibilities, self._experts):
            preds = expert.predict(X)
            votes[np.arange(len(X)), preds] += float(weight)
        return votes.argmax(axis=1).astype(np.int64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_samples += len(X)

        if self._global_expert_params:
            for expert, params in zip(self._experts, self._global_expert_params):
                if params:
                    expert.set_params(params)

        losses = np.array(
            [self._expert_loss(expert, X, y) for expert in self._experts],
            dtype=np.float64,
        )
        if not np.isfinite(losses).any():
            responsibilities = np.ones(self.n_components, dtype=np.float64) / self.n_components
        else:
            responsibilities = _softmax(-losses)

        n_local_steps = max(1, int(self.local_epochs))
        for expert in self._experts:
            for _ in range(n_local_steps):
                expert.partial_fit(X, y)

        self._expert_params = self._current_params()
        self._responsibilities = responsibilities / responsibilities.sum()
        self._selected_expert = int(np.argmax(self._responsibilities))

    def get_upload(self) -> FedEMUpload:
        return FedEMUpload(
            client_id=self.client_id,
            expert_params=[_copy_params(p) for p in self._expert_params],
            responsibilities=self._responsibilities.copy(),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        payload = sum(model_bytes(params, precision_bits=precision_bits) for params in self._expert_params)
        payload += self._responsibilities.size * precision_bits / 8
        return float(payload)


class FedEMServer:
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        n_components: int = 3,
        seed: int = 0,
    ) -> None:
        if n_components <= 0:
            raise ValueError(f"n_components must be > 0, got {n_components}")
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_components = n_components
        self._seed = seed
        self.global_experts = self._init_params(seed)

    def _init_params(self, seed: int) -> list[dict[str, np.ndarray]]:
        rng = np.random.RandomState(seed)
        n_out = 1 if self.n_classes == 2 else self.n_classes
        experts = []
        for _ in range(self.n_components):
            experts.append(
                {
                    "coef": (rng.randn(n_out * self.n_features) * 0.01).astype(np.float64),
                    "intercept": np.zeros(n_out, dtype=np.float64),
                }
            )
        return experts

    def aggregate(self, uploads: list[FedEMUpload]) -> list[dict[str, np.ndarray]]:
        if not uploads:
            return [_copy_params(p) for p in self.global_experts]

        aggregated: list[dict[str, np.ndarray]] = []
        for expert_idx in range(self.n_components):
            weighted_sum: dict[str, np.ndarray] | None = None
            total_weight = 0.0
            for upload in uploads:
                if expert_idx >= len(upload.expert_params):
                    continue
                params = upload.expert_params[expert_idx]
                if not params:
                    continue
                weight = float(max(1, upload.n_samples)) * float(upload.responsibilities[expert_idx])
                if weight <= 0:
                    continue
                if weighted_sum is None:
                    weighted_sum = {k: v * weight for k, v in params.items()}
                else:
                    for key in params:
                        weighted_sum[key] = weighted_sum[key] + params[key] * weight
                total_weight += weight
            if weighted_sum is None or total_weight <= 0:
                aggregated.append(_copy_params(self.global_experts[expert_idx]))
            else:
                aggregated.append({k: v / total_weight for k, v in weighted_sum.items()})

        self.global_experts = [_copy_params(p) for p in aggregated]
        return [_copy_params(p) for p in self.global_experts]

    def download_bytes(self, n_clients: int, precision_bits: int = 32) -> float:
        return float(n_clients) * sum(
            model_bytes(params, precision_bits=precision_bits) for params in self.global_experts
        )


def run_fedem_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    n_components: int = 3,
    local_epochs: int = 2,
    lr: float = 0.05,
    model_type: str = "linear",
) -> MethodResult:
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FedEMClient(
            k,
            n_features,
            n_classes,
            n_components=n_components,
            lr=lr,
            local_epochs=local_epochs,
            seed=42 + k,
            model_type=model_type,
        )
        for k in range(K)
    ]
    server = FedEMServer(
        n_features=n_features,
        n_classes=n_classes,
        n_components=n_components,
        seed=42,
    )

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
            global_experts = server.aggregate(uploads)
            download_bytes = server.download_bytes(K)
            total_bytes += upload_bytes + download_bytes
            for client in clients:
                client.set_global_experts(global_experts)

        for k in range(K):
            predicted_concept_matrix[k, t] = clients[k].selected_expert_id

    return MethodResult(
        method_name="FedEM",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
