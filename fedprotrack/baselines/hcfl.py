from __future__ import annotations

"""HCFL baseline: Hierarchical Clustered Federated Learning.

HCFL (ICLR 2025) performs agglomerative clustering on flattened model
parameter vectors using cosine distance, then aggregates models within
each discovered cluster.  This enables automatic client grouping without
a fixed number of clusters.

Communication budget: same as FedAvg (model params up/down + cluster ID).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


@dataclass
class MethodResult:
    """Unified result container for baseline methods.

    Parameters
    ----------
    method_name : str
        Human-readable name of the method.
    accuracy_matrix : np.ndarray
        Shape (K, T). Per-client per-step classification accuracy.
    predicted_concept_matrix : np.ndarray
        Shape (K, T). Predicted concept IDs.
    total_bytes : float
        Total communication cost in bytes.
    """

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
class HCFLUpload:
    """Data uploaded by one HCFL client per federation round.

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


class HCFLClient:
    """Client for the HCFL baseline.

    Trains locally, uploads model parameters, and receives a cluster-specific
    aggregated model from the server.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    lr : float
        Local SGD learning rate. Default 0.01.
    n_epochs : int
        Number of local training epochs per round. Default 5.
    seed : int
        Random seed. Default 0.
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
        self._seed = seed

        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._cluster_id: int = 0
        self._n_samples: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train locally on the given batch.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        self._n_samples += len(X)
        self._model.fit(X, y)
        self._model_params = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the current model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self._model.predict(X)

    def get_upload(self) -> HCFLUpload:
        """Package locally trained model for upload.

        Returns
        -------
        HCFLUpload
        """
        return HCFLUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute bytes required to upload the model.

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        return model_bytes(self._model_params, precision_bits=precision_bits)

    def set_global_params(self, params: dict[str, np.ndarray]) -> None:
        """Receive aggregated model parameters from the server.

        Parameters
        ----------
        params : dict[str, np.ndarray]
        """
        if params:
            self._model.set_params(params)
            self._model_params = _copy_params(params)

    def set_cluster_id(self, cid: int) -> None:
        """Store the cluster assignment from the server.

        Parameters
        ----------
        cid : int
        """
        self._cluster_id = cid

    @property
    def cluster_id(self) -> int:
        """Return the current cluster assignment."""
        return self._cluster_id


class HCFLServer:
    """Server for the HCFL baseline.

    Performs agglomerative clustering on flattened model parameter vectors
    using cosine distance, then aggregates models within each cluster via
    sample-weighted averaging.

    Parameters
    ----------
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of classes.
    distance_threshold : float
        Cosine distance threshold for agglomerative clustering.
        Default 0.5.
    linkage : str
        Linkage criterion for ``AgglomerativeClustering``.
        Default ``"average"``.
    seed : int
        Random seed for initialisation. Default 42.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        distance_threshold: float = 0.5,
        linkage: str = "average",
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self._seed = seed

        # Initialise fallback global params
        rng = np.random.RandomState(seed)
        n_out = 1 if n_classes == 2 else n_classes
        self.global_params: dict[str, np.ndarray] = {
            "coef": (rng.randn(n_out * n_features) * 0.01).astype(np.float64),
            "intercept": np.zeros(n_out, dtype=np.float64),
        }

    def aggregate(
        self, uploads: list[HCFLUpload],
    ) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, int]]:
        """Cluster clients and aggregate models per cluster.

        Parameters
        ----------
        uploads : list[HCFLUpload]
            One upload per participating client.

        Returns
        -------
        cluster_models : dict[int, dict[str, np.ndarray]]
            Mapping from cluster ID to aggregated model parameters.
        client_clusters : dict[int, int]
            Mapping from client ID to assigned cluster ID.
        """
        if not uploads:
            return {0: _copy_params(self.global_params)}, {}

        if len(uploads) == 1:
            u = uploads[0]
            params = _copy_params(u.model_params) if u.model_params else _copy_params(self.global_params)
            self.global_params = _copy_params(params)
            return {0: params}, {u.client_id: 0}

        # (1) Flatten each upload's model_params into a 1-D vector
        sorted_keys = sorted(uploads[0].model_params.keys())
        vectors = np.array([
            np.concatenate([u.model_params[k].ravel() for k in sorted_keys])
            for u in uploads
        ])

        # (2) Compute pairwise cosine distance matrix
        dist_matrix = cosine_distances(vectors)

        # (3) Agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            linkage=self.linkage,
            metric="precomputed",
        )
        labels = clustering.fit_predict(dist_matrix)

        # (4) Group uploads by cluster label
        cluster_groups: dict[int, list[HCFLUpload]] = {}
        client_clusters: dict[int, int] = {}
        for upload, label in zip(uploads, labels):
            cid = int(label)
            client_clusters[upload.client_id] = cid
            if cid not in cluster_groups:
                cluster_groups[cid] = []
            cluster_groups[cid].append(upload)

        # (5) Per-cluster weighted average (weight = n_samples)
        cluster_models: dict[int, dict[str, np.ndarray]] = {}
        for cid, members in cluster_groups.items():
            weights = [float(max(m.n_samples, 1)) for m in members]
            total_w = sum(weights)
            if total_w == 0:
                weights = [1.0] * len(members)
                total_w = float(len(members))

            new_params: dict[str, np.ndarray] = {}
            for key in sorted_keys:
                stacked = np.stack([m.model_params[key] for m in members])
                w_arr = np.array(weights, dtype=np.float64) / total_w
                new_params[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))
            cluster_models[cid] = new_params

        # Update global_params to the overall weighted average (fallback)
        all_weights = [float(max(u.n_samples, 1)) for u in uploads]
        total_all = sum(all_weights)
        if total_all == 0:
            all_weights = [1.0] * len(uploads)
            total_all = float(len(uploads))
        global_new: dict[str, np.ndarray] = {}
        for key in sorted_keys:
            stacked = np.stack([u.model_params[key] for u in uploads])
            w_arr = np.array(all_weights, dtype=np.float64) / total_all
            global_new[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))
        self.global_params = global_new

        return cluster_models, client_clusters

    def download_bytes(
        self,
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        """Compute bytes to send each client its cluster model + cluster ID.

        Parameters
        ----------
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        # Each client receives one model + 4 bytes for cluster ID (int32)
        return float(n_clients) * (
            model_bytes(self.global_params, precision_bits=precision_bits) + 4
        )


def run_hcfl_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    distance_threshold: float = 0.5,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run HCFL and return full results.

    Parameters
    ----------
    dataset : DriftDataset
        Synthetic or real drift dataset.
    federation_every : int
        Federation frequency (every N rounds). Default 1.
    distance_threshold : float
        Cosine distance threshold for agglomerative clustering.
        Default 0.5.
    lr : float
        Local SGD learning rate. Default 0.01.
    n_epochs : int
        Number of local training epochs per round. Default 5.

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        HCFLClient(
            k,
            n_features,
            n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = HCFLServer(
        n_features=n_features,
        n_classes=n_classes,
        distance_threshold=distance_threshold,
        seed=42,
    )

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
            upload_b = sum(client.upload_bytes() for client in clients)
            cluster_models, client_clusters = server.aggregate(uploads)
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b

            for client in clients:
                cid = client_clusters.get(client.client_id, 0)
                client.set_cluster_id(cid)
                params = cluster_models.get(cid, server.global_params)
                client.set_global_params(params)

        # Record predicted concept identity (cluster ID)
        for k in range(K):
            predicted_concept_matrix[k, t] = clients[k].cluster_id

    return MethodResult(
        method_name="HCFL",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
