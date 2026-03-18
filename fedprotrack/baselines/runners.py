"""Full baseline runners returning (K, T) accuracy matrices.

The existing budget_sweep runners only return BudgetPoint (scalar AUC +
bytes). These wrappers produce complete per-client per-step accuracy
and predicted concept matrices, suitable for unified metric computation
via ``compute_all_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from .comm_tracker import model_bytes, prototype_bytes, fingerprint_bytes
from .compressed_fedavg import CompressedFedAvgClient, CompressedFedAvgServer
from .feddrift import FedDriftClient, FedDriftServer
from .fedproto import FedProtoClient, FedProtoAggregator
from .flash import FlashClient, FlashAggregator
from .ifca import IFCAClient, IFCAServer
from .tracked_summary import TrackedSummaryClient, TrackedSummaryServer


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


def _extract_dims(dataset: DriftDataset) -> tuple[int, int, int, int]:
    """Return (K, T, n_features, n_classes) from a DriftDataset."""
    K = dataset.config.K
    T = dataset.config.T
    X0, y0 = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    all_labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1
    return K, T, n_features, n_classes


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def run_fedproto_full(
    dataset: DriftDataset,
    federation_every: int = 1,
) -> MethodResult:
    """Run FedProto and return full accuracy matrix.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [FedProtoClient(k, n_features, n_classes) for k in range(K)]
    aggregator = FedProtoAggregator()

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            global_protos = aggregator.aggregate(uploads)
            download_b = aggregator.download_bytes(global_protos, K)
            total_bytes += upload_b + download_b
            for c in clients:
                c.set_global_prototypes(global_protos)

    # FedProto has no concept tracking
    predicted = np.zeros((K, T), dtype=np.int32)

    return MethodResult(
        method_name="FedProto",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted,
        total_bytes=total_bytes,
    )


def run_tracked_summary_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    similarity_threshold: float = 0.5,
) -> MethodResult:
    """Run TrackedSummary and return full accuracy matrix with cluster IDs.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    similarity_threshold : float

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        TrackedSummaryClient(k, n_features, n_classes, seed=42 + k)
        for k in range(K)
    ]
    server = TrackedSummaryServer(similarity_threshold=similarity_threshold)

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0
    current_cluster_ids: dict[int, int] = {k: 0 for k in range(K)}

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            aggregated = server.aggregate(uploads)
            download_b = server.download_bytes(uploads)
            total_bytes += upload_b + download_b

            # Infer cluster assignments from aggregated params
            # Clients with same aggregated params are in the same cluster
            param_to_cluster: dict[int, int] = {}
            next_cluster = 0
            for c in clients:
                params = aggregated.get(c.client_id, {})
                if params:
                    c.set_model_params(params)
                # Use id of the aggregated params dict as cluster key
                pid = id(params)
                if pid not in param_to_cluster:
                    param_to_cluster[pid] = next_cluster
                    next_cluster += 1
                current_cluster_ids[c.client_id] = param_to_cluster[pid]

        for k in range(K):
            predicted_matrix[k, t] = current_cluster_ids[k]

    return MethodResult(
        method_name="TrackedSummary",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )


def run_flash_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    distill_alpha: float = 0.3,
) -> MethodResult:
    """Run Flash (single-model drift adaptation) and return full results.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    distill_alpha : float

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FlashClient(k, n_features, n_classes, distill_alpha=distill_alpha, seed=42 + k)
        for k in range(K)
    ]
    aggregator = FlashAggregator()

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            global_params = aggregator.aggregate(uploads)
            download_b = aggregator.download_bytes(global_params, K)
            total_bytes += upload_b + download_b

            for c in clients:
                if global_params:
                    c.set_model_params(global_params)

    # Flash has no concept tracking (single global model)
    predicted = np.zeros((K, T), dtype=np.int32)

    return MethodResult(
        method_name="Flash",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted,
        total_bytes=total_bytes,
    )


def run_feddrift_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    similarity_threshold: float = 0.5,
) -> MethodResult:
    """Run FedDrift (multi-model with drift branching) and return full results.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    similarity_threshold : float

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FedDriftClient(k, n_features, n_classes, similarity_threshold=similarity_threshold, seed=42 + k)
        for k in range(K)
    ]
    server = FedDriftServer(similarity_threshold=similarity_threshold)

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            aggregated = server.aggregate(uploads)
            download_b = server.download_bytes(uploads)
            total_bytes += upload_b + download_b

            for c in clients:
                params = aggregated.get(c.client_id, {})
                if params:
                    c.set_model_params(params)

        for k in range(K):
            predicted_matrix[k, t] = clients[k].active_concept_id

    return MethodResult(
        method_name="FedDrift",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )


def run_ifca_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    n_clusters: int = 3,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run IFCA (iterative federated clustering) and return full results.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    n_clusters : int
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
        IFCAClient(k, n_features, n_classes, seed=42 + k, lr=lr, n_epochs=n_epochs)
        for k in range(K)
    ]
    server = IFCAServer(
        n_clusters=n_clusters, n_features=n_features,
        n_classes=n_classes, seed=42,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    # Distribute initial cluster models
    for c in clients:
        c.set_cluster_models(server.cluster_models)

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            updated_clusters = server.aggregate(uploads)
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b

            for c in clients:
                c.set_cluster_models(updated_clusters)

        for k in range(K):
            predicted_matrix[k, t] = clients[k]._selected_cluster

    return MethodResult(
        method_name="IFCA",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )


def run_compressed_fedavg_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    topk_fraction: float = 0.3,
) -> MethodResult:
    """Run CompressedFedAvg (sparsified model exchange) and return full results.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    topk_fraction : float

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        CompressedFedAvgClient(k, n_features, n_classes, topk_fraction=topk_fraction, seed=42 + k)
        for k in range(K)
    ]
    server = CompressedFedAvgServer(n_features, n_classes)

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(
                c.upload_bytes_from_upload(u)
                for c, u in zip(clients, uploads)
            )
            global_params = server.aggregate(uploads)
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b

            for c in clients:
                c.set_model_params(global_params)

    # No concept tracking
    predicted = np.zeros((K, T), dtype=np.int32)

    return MethodResult(
        method_name="CompressedFedAvg",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted,
        total_bytes=total_bytes,
    )
