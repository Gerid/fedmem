from __future__ import annotations

"""FLUX baseline: descriptor-driven clustered federated learning.

This is a lightweight migration of the official FLUX repository
(`dariofenoglio98/FLUX`) to the repository's linear-classifier / drift
dataset setup.

The original FLUX code is written for image models with rich latent-space
descriptors. Here we keep the same high-level recipe:
- extract compact per-client descriptors from local batches;
- cluster clients by descriptor similarity;
- aggregate client models within each cluster;
- support a "prior" variant that uses a fixed cluster count.

The descriptors are simplified to statistics over the current batch:
feature mean/std, label histogram, and class-conditional means. This is
enough to separate the synthetic drift streams used in this repo while
remaining cheap and deterministic.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


@dataclass
class FLUXUpload:
    """Payload uploaded by one FLUX client."""

    client_id: int
    descriptor: np.ndarray
    model_params: dict[str, np.ndarray]
    n_samples: int
    cluster_id: int | None


@dataclass
class FLUXUpdate:
    """Cluster-specific state broadcast by the server."""

    cluster_id: int
    model_params: dict[str, np.ndarray]
    centroid: np.ndarray


@dataclass
class FLUXResult:
    """Compact result returned by ``run_flux_full``."""

    method_name: str
    accuracy_matrix: np.ndarray
    predicted_cluster_matrix: np.ndarray
    total_bytes: float


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _stack_params(
    params_list: list[dict[str, np.ndarray]],
    weights: list[int],
) -> dict[str, np.ndarray]:
    total = float(sum(max(1, w) for w in weights))
    agg: dict[str, np.ndarray] = {}
    for key in params_list[0]:
        accum = None
        for params, weight in zip(params_list, weights, strict=True):
            arr = params[key]
            w = max(1, int(weight))
            accum = arr * w if accum is None else accum + arr * w
        agg[key] = accum / total
    return agg


def _descriptor_bytes(descriptor: np.ndarray, precision_bits: int = 32) -> float:
    return float(descriptor.size * precision_bits / 8)


def _standardize(X: np.ndarray) -> np.ndarray:
    if len(X) <= 1:
        return X.copy()
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std


def _estimate_dbscan_eps(X: np.ndarray) -> float:
    if len(X) <= 2:
        return 0.5
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    nn = np.min(dists, axis=1)
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        return 0.5
    eps = float(np.median(nn) * 1.25)
    return max(eps, 0.25)


class FLUXClient:
    """Client-side FLUX model with descriptor extraction."""

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed
        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=0.1,
            n_epochs=3,
            seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._n_samples = 0
        self._cluster_id: int | None = None
        self._cluster_centroid: np.ndarray | None = None
        self._descriptor_cache = np.zeros(
            self.n_features * 2 + self.n_classes + self.n_features * self.n_classes,
            dtype=np.float64,
        )

    def _descriptor(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        feat_mean = X.mean(axis=0)
        feat_std = X.std(axis=0)
        hist = np.bincount(y.astype(np.int64), minlength=self.n_classes).astype(np.float64)
        hist /= max(1.0, hist.sum())

        cond_means = []
        for cls in range(self.n_classes):
            mask = y == cls
            if np.any(mask):
                cond_means.append(X[mask].mean(axis=0))
            else:
                cond_means.append(np.zeros(self.n_features, dtype=np.float64))
        descriptor = np.concatenate([feat_mean, feat_std, hist, np.concatenate(cond_means)])
        return descriptor.astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_samples += len(X)
        self._model.fit(X, y)
        self._model_params = self._model.get_params()
        self._descriptor_cache = self._descriptor(X, y)
        self._cluster_id = self._cluster_id if self._cluster_id is not None else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def descriptor(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self._descriptor(X, y)

    @property
    def cluster_id(self) -> int | None:
        return self._cluster_id

    def set_cluster_state(self, update: FLUXUpdate | None) -> None:
        if update is None:
            return
        self._cluster_id = update.cluster_id
        self._cluster_centroid = update.centroid.copy()
        self._model.set_params(update.model_params)
        self._model_params = _copy_params(update.model_params)

    def get_upload(self) -> FLUXUpload:
        return FLUXUpload(
            client_id=self.client_id,
            descriptor=self._descriptor_cache.copy(),
            model_params=_copy_params(self._model_params),
            n_samples=self._n_samples,
            cluster_id=self._cluster_id,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        return model_bytes(self._model_params, precision_bits=precision_bits) + _descriptor_bytes(
            self._descriptor_cache, precision_bits=precision_bits
        )


class FLUXServer:
    """Server-side clustering and per-cluster FedAvg."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        prior_n_clusters: int | None = None,
        dbscan_min_samples: int = 2,
        dbscan_eps: float | None = None,
        seed: int = 0,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.prior_n_clusters = prior_n_clusters
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_eps = dbscan_eps
        self._seed = seed
        self._cluster_models: dict[int, dict[str, np.ndarray]] = {}
        self._cluster_centroids: dict[int, np.ndarray] = {}
        self._client_assignments: dict[int, int] = {}

    def _cluster_labels(self, descriptors: np.ndarray) -> np.ndarray:
        if len(descriptors) == 0:
            return np.zeros(0, dtype=np.int32)
        if len(descriptors) == 1:
            return np.zeros(1, dtype=np.int32)

        X = _standardize(descriptors)
        if self.prior_n_clusters is not None:
            n_clusters = max(1, min(self.prior_n_clusters, len(descriptors)))
            labels = KMeans(n_clusters=n_clusters, random_state=self._seed, n_init=10).fit_predict(X)
            return labels.astype(np.int32)

        eps = self.dbscan_eps if self.dbscan_eps is not None else _estimate_dbscan_eps(X)
        labels = DBSCAN(eps=eps, min_samples=self.dbscan_min_samples).fit_predict(X)
        valid = set(int(v) for v in labels if v >= 0)
        if len(valid) >= 2:
            remap = labels.copy()
            next_label = len(valid)
            for idx, label in enumerate(labels):
                if label == -1:
                    remap[idx] = next_label
                    next_label += 1
            return remap.astype(np.int32)

        best_labels = np.zeros(len(descriptors), dtype=np.int32)
        best_score = float("-inf")
        upper = min(6, len(descriptors) - 1)
        for n_clusters in range(2, upper + 1):
            labels_k = KMeans(n_clusters=n_clusters, random_state=self._seed, n_init=10).fit_predict(X)
            if len(set(labels_k)) < 2:
                continue
            score = silhouette_score(X, labels_k)
            if score > best_score:
                best_score = score
                best_labels = labels_k.astype(np.int32)
        return best_labels

    def aggregate(self, uploads: list[FLUXUpload]) -> dict[int, FLUXUpdate]:
        if not uploads:
            return {}

        descriptors = np.stack([u.descriptor for u in uploads], axis=0)
        labels = self._cluster_labels(descriptors)

        cluster_members: dict[int, list[FLUXUpload]] = {}
        for upload, label in zip(uploads, labels, strict=True):
            cluster_id = int(label)
            self._client_assignments[upload.client_id] = cluster_id
            cluster_members.setdefault(cluster_id, []).append(upload)

        self._cluster_models = {}
        self._cluster_centroids = {}
        for cluster_id, members in cluster_members.items():
            params = _stack_params(
                [u.model_params for u in members],
                [u.n_samples for u in members],
            )
            centroid = np.stack([u.descriptor for u in members], axis=0).mean(axis=0)
            self._cluster_models[cluster_id] = params
            self._cluster_centroids[cluster_id] = centroid

        return {
            upload.client_id: FLUXUpdate(
                cluster_id=self._client_assignments[upload.client_id],
                model_params=_copy_params(self._cluster_models[self._client_assignments[upload.client_id]]),
                centroid=self._cluster_centroids[self._client_assignments[upload.client_id]].copy(),
            )
            for upload in uploads
        }

    def download_bytes(self, updates: dict[int, FLUXUpdate], precision_bits: int = 32) -> float:
        if not updates:
            return 0.0
        total = 0.0
        for update in updates.values():
            total += model_bytes(update.model_params, precision_bits=precision_bits)
            total += _descriptor_bytes(update.centroid, precision_bits=precision_bits)
        return total

    def assignment_for(self, client_id: int) -> int | None:
        return self._client_assignments.get(client_id)

    @property
    def cluster_models(self) -> dict[int, dict[str, np.ndarray]]:
        return {k: _copy_params(v) for k, v in self._cluster_models.items()}

    @property
    def cluster_centroids(self) -> dict[int, np.ndarray]:
        return {k: v.copy() for k, v in self._cluster_centroids.items()}


class FLUXPriorServer(FLUXServer):
    """FLUX variant with a known prior cluster count."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_clusters: int,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_classes=n_classes,
            prior_n_clusters=n_clusters,
            seed=seed,
        )


def run_flux_full(
    dataset,
    federation_every: int = 1,
    *,
    prior_n_clusters: int | None = None,
) -> FLUXResult:
    """Run a small FLUX simulation on a ``DriftDataset``."""

    K = dataset.config.K
    T = dataset.config.T
    X0, _ = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    all_labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    clients = [
        FLUXClient(k, n_features, n_classes, seed=42 + k)
        for k in range(K)
    ]
    if prior_n_clusters is None:
        server: FLUXServer = FLUXServer(n_features, n_classes, seed=42)
    else:
        server = FLUXPriorServer(n_features, n_classes, prior_n_clusters, seed=42)

    accuracy = np.zeros((K, T), dtype=np.float64)
    predicted_clusters = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy[k, t] = _accuracy(y, preds)
            predicted_clusters[k, t] = clients[k].cluster_id if clients[k].cluster_id is not None else 0

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [client.get_upload() for client in clients]
            upload_bytes = sum(client.upload_bytes() for client in clients)
            updates = server.aggregate(uploads)
            total_bytes += upload_bytes + server.download_bytes(updates)
            for client in clients:
                update = updates.get(client.client_id)
                client.set_cluster_state(update)

    return FLUXResult(
        method_name="FLUX-prior" if prior_n_clusters is not None else "FLUX",
        accuracy_matrix=accuracy,
        predicted_cluster_matrix=predicted_clusters,
        total_bytes=total_bytes,
    )
