from __future__ import annotations

"""CFL baseline: clustered federated learning with update-similarity splits.

This is a lightweight port of the original CFL notebook-style workflow:
clients train locally, the server measures the similarity of client
updates, and an agglomerative split is triggered when a cluster looks
heterogeneous. The implementation is adapted to the repo's
``TorchLinearClassifier`` and numpy-based federation boundary.

The upstream source used for the port is Felix Sattler's CFL reference
implementation (``clustered-federated-learning``), especially the
cluster-splitting logic driven by pairwise cosine similarities and update
norm thresholds.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from ..federation.aggregator import FedAvgAggregator
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _params_to_vector(params: dict[str, np.ndarray]) -> np.ndarray:
    if not params:
        return np.zeros(1, dtype=np.float64)
    return np.concatenate([v.reshape(-1) for v in params.values()]).astype(np.float64)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pairwise_cosine_similarity(vectors: Sequence[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    sims = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_sim(vectors[i], vectors[j])
            sims[i, j] = sims[j, i] = sim
    return sims


def _weighted_average_params(
    params_list: Sequence[dict[str, np.ndarray]],
    weights: Sequence[float] | None = None,
) -> dict[str, np.ndarray]:
    if not params_list:
        return {}
    return FedAvgAggregator().aggregate(list(params_list), list(weights) if weights is not None else None)


def _fresh_params(n_features: int, n_classes: int, seed: int) -> dict[str, np.ndarray]:
    model = TorchLinearClassifier(
        n_features=n_features,
        n_classes=n_classes,
        seed=seed,
    )
    return model.get_params()


def _two_way_split(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
    """Split a cluster into two groups by cosine-distance agglomeration."""

    if len(vectors) < 2:
        return None

    sim = _pairwise_cosine_similarity(vectors)
    dist = np.clip(1.0 - sim, 0.0, None)
    if np.allclose(dist, 0.0):
        return None

    try:
        clustering = AgglomerativeClustering(
            n_clusters=2,
            metric="precomputed",
            linkage="complete",
        ).fit(dist)
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=2,
            affinity="precomputed",
            linkage="complete",
        ).fit(dist)

    labels = np.asarray(clustering.labels_, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return None
    return labels


@dataclass
class CFLUpload:
    """Local update uploaded by one CFL client."""

    client_id: int
    model_params: dict[str, np.ndarray]
    update_vector: np.ndarray
    n_samples: int
    cluster_id: int


class CFLClient:
    """CFL client with a single linear classifier."""

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        lr: float = 0.1,
        n_epochs: int = 10,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed
        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._n_samples = 0
        self._cluster_id = 0
        self._model_params: dict[str, np.ndarray] = {}
        self._update_vector = np.zeros(1, dtype=np.float64)

    def set_model_params(self, params: dict[str, np.ndarray]) -> None:
        if params:
            self._model.set_params(_copy_params(params))
            self._model_params = _copy_params(params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        before = _copy_params(self._model.get_params())
        self._model.fit(X, y)
        after = self._model.get_params()
        self._model_params = _copy_params(after)
        self._update_vector = _params_to_vector(after) - _params_to_vector(before)
        self._n_samples += int(len(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_upload(self) -> CFLUpload:
        return CFLUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            update_vector=self._update_vector.copy(),
            n_samples=self._n_samples,
            cluster_id=self._cluster_id,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        return model_bytes(self._model_params, precision_bits)


class CFLServer:
    """Server-side CFL controller.

    The server starts from a single shared cluster and grows the number of
    clusters when a cluster's mean update norm is small but the maximum
    client update inside that cluster remains large. This mirrors the
    splitting rule used in the original CFL notebook.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        seed: int = 0,
        eps_1: float = 0.4,
        eps_2: float = 1.6,
        warmup_rounds: int = 20,
        max_clusters: int = 8,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.seed = seed
        self.eps_1 = eps_1
        self.eps_2 = eps_2
        self.warmup_rounds = warmup_rounds
        self.max_clusters = max_clusters
        self._fedavg = FedAvgAggregator()
        self.cluster_models: list[dict[str, np.ndarray]] = [
            _fresh_params(n_features, n_classes, seed),
        ]
        self.client_cluster_map: dict[int, int] = {}

    @property
    def n_clusters(self) -> int:
        return len(self.cluster_models)

    def broadcast(self, client_id: int) -> dict[str, np.ndarray]:
        cluster_id = self.client_cluster_map.get(client_id, 0)
        cluster_id = max(0, min(cluster_id, len(self.cluster_models) - 1))
        return _copy_params(self.cluster_models[cluster_id])

    def _split_cluster(self, members: list[CFLUpload]) -> list[list[CFLUpload]]:
        vectors = [u.update_vector for u in members]
        labels = _two_way_split(vectors)
        if labels is None:
            return [members]

        left = [u for u, lbl in zip(members, labels) if int(lbl) == 0]
        right = [u for u, lbl in zip(members, labels) if int(lbl) == 1]
        if not left or not right:
            return [members]
        return [left, right]

    def _should_split(self, members: list[CFLUpload], round_idx: int) -> bool:
        if len(members) <= 2:
            return False
        if len(self.cluster_models) >= self.max_clusters:
            return False
        if round_idx < self.warmup_rounds:
            return False
        norms = np.asarray([np.linalg.norm(u.update_vector) for u in members], dtype=np.float64)
        mean_norm = float(np.linalg.norm(np.mean([u.update_vector for u in members], axis=0)))
        max_norm = float(norms.max(initial=0.0))
        return mean_norm < self.eps_1 and max_norm > self.eps_2

    def aggregate(
        self,
        uploads: list[CFLUpload],
        round_idx: int = 0,
    ) -> list[dict[str, np.ndarray]]:
        if not uploads:
            return [_copy_params(p) for p in self.cluster_models]

        grouped: dict[int, list[CFLUpload]] = {}
        for upload in uploads:
            cluster_id = self.client_cluster_map.get(upload.client_id, upload.cluster_id)
            grouped.setdefault(cluster_id, []).append(upload)

        next_groups: list[list[CFLUpload]] = []
        next_assignments: dict[int, int] = {}
        next_models: list[dict[str, np.ndarray]] = []

        for cluster_id in sorted(grouped):
            members = grouped[cluster_id]
            parts = self._split_cluster(members) if self._should_split(members, round_idx) else [members]
            for part in parts:
                next_groups.append(part)

        for new_cluster_id, members in enumerate(next_groups):
            params_list = [u.model_params for u in members if u.model_params]
            if params_list:
                weights = [float(max(u.n_samples, 1)) for u in members if u.model_params]
                aggregated = _weighted_average_params(params_list, weights)
            else:
                aggregated = _copy_params(self.cluster_models[min(new_cluster_id, len(self.cluster_models) - 1)])
            next_models.append(aggregated)
            for upload in members:
                next_assignments[upload.client_id] = new_cluster_id

        self.cluster_models = next_models or self.cluster_models
        self.client_cluster_map = next_assignments
        return [_copy_params(p) for p in self.cluster_models]

    def download_bytes(self, n_clients: int, precision_bits: int = 32) -> float:
        per_client = sum(model_bytes(params, precision_bits) for params in self.cluster_models)
        return float(n_clients) * float(per_client)
