from __future__ import annotations

"""FedGWC baseline: Gaussian Weighting with Wasserstein Clustering.

FedGWC (ICML 2025) computes pairwise Wasserstein distances between
client classifier weight rows, applies a Gaussian kernel to get
similarities, then clusters clients via DBSCAN on the distance matrix.
Per-cluster aggregation uses Gaussian-weighted averaging.

Communication budget: same as FedAvg.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import DBSCAN

from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


@dataclass
class MethodResult:
    """Unified result container for FedGWC.

    Parameters
    ----------
    method_name : str
        Human-readable name of the method.
    accuracy_matrix : np.ndarray
        Shape (K, T). Per-client per-step classification accuracy.
    predicted_concept_matrix : np.ndarray
        Shape (K, T). Predicted concept IDs (cluster assignments).
    total_bytes : float
        Total communication cost in bytes.
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
            Shape (K, T) ground-truth concept IDs.

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


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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


@dataclass
class FedGWCUpload:
    """Upload payload from a FedGWC client.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    model_params : dict[str, np.ndarray]
        Current local model parameters.
    n_samples : int
        Cumulative number of training samples seen by this client.
    """

    client_id: int
    model_params: dict[str, np.ndarray]
    n_samples: int


class FedGWCClient:
    """Local client for FedGWC.

    Parameters
    ----------
    client_id : int
        Unique client identifier.
    n_features : int
        Dimensionality of input features.
    n_classes : int
        Number of output classes.
    lr : float
        Local SGD learning rate.
    n_epochs : int
        Number of local training epochs per round.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        lr: float = 0.01,
        n_epochs: int = 5,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._cluster_id: int = 0
        self._n_samples: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the local model on one batch.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n, n_features).
        y : np.ndarray
            Label vector of shape (n,).
        """
        self._n_samples += len(X)
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for ``X``.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n,).
        """
        return self._model.predict(X)

    def get_upload(self) -> FedGWCUpload:
        """Prepare an upload payload for the server.

        Returns
        -------
        FedGWCUpload
        """
        return FedGWCUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model.get_params()),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute upload communication cost in bytes.

        Parameters
        ----------
        precision_bits : int
            Bit-width per scalar element.

        Returns
        -------
        float
        """
        return model_bytes(self._model.get_params(), precision_bits=precision_bits)

    def set_global_params(self, params: dict[str, np.ndarray]) -> None:
        """Replace local model parameters with server-provided ones.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Model parameters to load.
        """
        self._model.set_params(_copy_params(params))

    def set_cluster_id(self, cid: int) -> None:
        """Set the cluster assignment for this client.

        Parameters
        ----------
        cid : int
            Cluster identifier.
        """
        self._cluster_id = cid

    @property
    def cluster_id(self) -> int:
        """Return current cluster assignment."""
        return self._cluster_id


class FedGWCServer:
    """Server for FedGWC: Wasserstein-distance clustering + Gaussian weighting.

    Parameters
    ----------
    n_features : int
        Dimensionality of input features.
    n_classes : int
        Number of output classes.
    sigma : float
        Bandwidth of the Gaussian kernel applied to Wasserstein distances.
    dbscan_eps : float
        DBSCAN epsilon on the ``1 - similarity`` distance matrix.
    dbscan_min_samples : int
        DBSCAN min_samples parameter.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        sigma: float = 1.0,
        dbscan_eps: float = 0.5,
        dbscan_min_samples: int = 1,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.sigma = sigma
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self._seed = seed

    def aggregate(
        self,
        uploads: list[FedGWCUpload],
    ) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, int]]:
        """Cluster clients and produce per-cluster aggregated models.

        Parameters
        ----------
        uploads : list[FedGWCUpload]
            Client uploads for this round.

        Returns
        -------
        tuple[dict[int, dict[str, np.ndarray]], dict[int, int]]
            ``(cluster_models, client_clusters)`` where
            ``cluster_models`` maps cluster_id -> aggregated params and
            ``client_clusters`` maps client_id -> cluster_id.
        """
        if len(uploads) <= 1:
            # Single client fallback: everyone in cluster 0
            cluster_models: dict[int, dict[str, np.ndarray]] = {}
            client_clusters: dict[int, int] = {}
            if uploads:
                cluster_models[0] = _copy_params(uploads[0].model_params)
                client_clusters[uploads[0].client_id] = 0
            return cluster_models, client_clusters

        n = len(uploads)
        n_out = 1 if self.n_classes == 2 else self.n_classes

        # (1) Extract coef rows for each client
        coefs: list[np.ndarray] = []
        for u in uploads:
            coef = u.model_params["coef"].reshape(n_out, self.n_features)
            coefs.append(coef)

        # (2) Pairwise Wasserstein distance averaged over class rows
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                row_dists = [
                    wasserstein_distance(coefs[i][l], coefs[j][l])
                    for l in range(n_out)
                ]
                d = float(np.mean(row_dists))
                dist[i, j] = d
                dist[j, i] = d

        # (3) Gaussian kernel similarity
        sim = np.exp(-dist ** 2 / (2 * self.sigma ** 2))

        # (4) DBSCAN on 1 - similarity (clipped)
        dbscan_dist = np.clip(1.0 - sim, 0.0, np.inf)
        clustering = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
            metric="precomputed",
        ).fit(dbscan_dist)
        labels = clustering.labels_.copy()

        # (5) Handle noise points (label == -1)
        non_noise_mask = labels >= 0
        if not np.any(non_noise_mask):
            # All noise: put everything in cluster 0
            labels[:] = 0
        else:
            for idx in range(n):
                if labels[idx] == -1:
                    # Assign to nearest non-noise cluster by minimum distance
                    non_noise_indices = np.where(non_noise_mask)[0]
                    nearest = non_noise_indices[
                        np.argmin(dist[idx, non_noise_indices])
                    ]
                    labels[idx] = labels[nearest]

        # Build client_clusters mapping
        client_clusters = {}
        for idx, u in enumerate(uploads):
            client_clusters[u.client_id] = int(labels[idx])

        # (6) Per-cluster weighted average (weight by n_samples)
        unique_clusters = set(int(l) for l in labels)
        cluster_models = {}
        for cid in unique_clusters:
            members = [
                (uploads[idx], sim)
                for idx in range(n)
                if int(labels[idx]) == cid
            ]
            total_weight = sum(max(1, u.n_samples) for u, _ in members)
            keys = list(members[0][0].model_params.keys())
            agg: dict[str, np.ndarray] = {}
            for key in keys:
                acc = None
                for u, _ in members:
                    w = max(1, u.n_samples)
                    contrib = u.model_params[key] * w
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    agg[key] = acc / total_weight
            cluster_models[cid] = agg

        return cluster_models, client_clusters

    def download_bytes(
        self,
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        """Compute download communication cost.

        Each client receives one cluster model, so total download cost
        equals one model per client.

        Parameters
        ----------
        n_clients : int
            Number of participating clients.
        precision_bits : int
            Bit-width per scalar element.

        Returns
        -------
        float
        """
        n_out = 1 if self.n_classes == 2 else self.n_classes
        dummy_params = {
            "coef": np.zeros(n_out * self.n_features, dtype=np.float64),
            "intercept": np.zeros(n_out, dtype=np.float64),
        }
        return float(n_clients) * model_bytes(dummy_params, precision_bits=precision_bits)


def run_fedgwc_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    sigma: float = 1.0,
    dbscan_eps: float = 0.5,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run FedGWC end-to-end and return full accuracy + cluster matrices.

    Parameters
    ----------
    dataset : DriftDataset
        The federated drift dataset.
    federation_every : int
        Federation frequency (aggregate every N rounds).
    sigma : float
        Gaussian kernel bandwidth for Wasserstein similarity.
    dbscan_eps : float
        DBSCAN epsilon for clustering on the distance matrix.
    lr : float
        Local SGD learning rate.
    n_epochs : int
        Number of local training epochs per round.

    Returns
    -------
    MethodResult
        Contains accuracy_matrix (K, T), predicted_concept_matrix (K, T)
        with cluster IDs, and total communication bytes.
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FedGWCClient(
            k,
            n_features,
            n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = FedGWCServer(
        n_features=n_features,
        n_classes=n_classes,
        sigma=sigma,
        dbscan_eps=dbscan_eps,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_concept_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        # Evaluate before training
        for k in range(K):
            X, y = dataset.data[(k, t)]
            accuracy_matrix[k, t] = _accuracy(y, clients[k].predict(X))

        # Local training
        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        # Federation
        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [client.get_upload() for client in clients]
            upload_bytes = sum(client.upload_bytes() for client in clients)
            cluster_models, client_clusters = server.aggregate(uploads)
            download_bytes = server.download_bytes(K)
            total_bytes += upload_bytes + download_bytes

            # Distribute cluster models and assign cluster IDs
            for client in clients:
                cid = client_clusters.get(client.client_id, 0)
                client.set_cluster_id(cid)
                if cid in cluster_models:
                    client.set_global_params(cluster_models[cid])

        # Record cluster IDs
        for k in range(K):
            predicted_concept_matrix[k, t] = clients[k].cluster_id

    return MethodResult(
        method_name="FedGWC",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
