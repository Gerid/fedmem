"""Full baseline runners returning (K, T) accuracy matrices.

The existing budget_sweep runners only return BudgetPoint (scalar AUC +
bytes). These wrappers produce complete per-client per-step accuracy
and predicted concept matrices, suitable for unified metric computation
via ``compute_all_metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .apfl import run_apfl_full as _run_apfl_impl
from .atp import run_atp_full as _run_atp_impl
from .cfl import CFLClient, CFLServer
from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from .comm_tracker import model_bytes, prototype_bytes, fingerprint_bytes
from .compressed_fedavg import CompressedFedAvgClient, CompressedFedAvgServer
from .fedccfa import FedCCFAClient, FedCCFAServer
from .feddrift import FedDriftClient, FedDriftServer
from .fedem import run_fedem_full as _run_fedem_impl
from .fedproto import FedProtoClient, FedProtoAggregator
from .fedrc import FedRCClient, FedRCServer
from .fesem import FeSEMClient, FeSEMServer
from .flash import FlashClient, FlashAggregator
from .flux import run_flux_full as _run_flux_impl
from .ifca import IFCAClient, IFCAServer
from .pfedme import run_pfedme_full as _run_pfedme_impl
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


def _from_external_result(
    result: object,
    *,
    method_name: str,
    predicted_attr: str = "predicted_concept_matrix",
) -> MethodResult:
    accuracy_matrix = np.asarray(getattr(result, "accuracy_matrix"), dtype=np.float64)
    predicted = getattr(result, predicted_attr, None)
    if predicted is None:
        predicted = np.zeros_like(accuracy_matrix, dtype=np.int32)
    else:
        predicted = np.asarray(predicted, dtype=np.int32)
    return MethodResult(
        method_name=method_name,
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted,
        total_bytes=float(getattr(result, "total_bytes")),
    )


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


def run_pfedme_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    local_epochs: int = 3,
    K_steps: int = 5,
    lamda: float = 0.1,
    personal_learning_rate: float = 0.05,
) -> MethodResult:
    """Run pFedMe and return full results."""
    result = _run_pfedme_impl(
        dataset,
        federation_every=federation_every,
        local_epochs=local_epochs,
        K_steps=K_steps,
        lamda=lamda,
        personal_learning_rate=personal_learning_rate,
    )
    return _from_external_result(result, method_name="pFedMe")


def run_apfl_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    alpha: float = 0.5,
    alpha_lr: float = 0.05,
    local_steps: int = 2,
) -> MethodResult:
    """Run APFL and return full results."""
    result = _run_apfl_impl(
        dataset,
        federation_every=federation_every,
        alpha=alpha,
        alpha_lr=alpha_lr,
        local_steps=local_steps,
    )
    return _from_external_result(result, method_name="APFL")


def run_fedem_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    n_components: int = 3,
    local_epochs: int = 2,
) -> MethodResult:
    """Run FedEM and return full results."""
    result = _run_fedem_impl(
        dataset,
        federation_every=federation_every,
        n_components=n_components,
        local_epochs=local_epochs,
    )
    return _from_external_result(result, method_name="FedEM")


def run_fedccfa_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    cluster_eps: float = 0.35,
    reid_similarity_threshold: float = 0.85,
    prototype_mix: float = 0.20,
) -> MethodResult:
    """Run FedCCFA and return full results.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    cluster_eps : float
        DBSCAN epsilon for label-wise classifier clustering.
    reid_similarity_threshold : float
        Similarity threshold for persistent label-cluster IDs.
    prototype_mix : float
        Maximum prototype-alignment strength applied on clients.

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FedCCFAClient(
            k,
            n_features,
            n_classes,
            prototype_mix=prototype_mix,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = FedCCFAServer(
        n_features=n_features,
        n_classes=n_classes,
        eps=cluster_eps,
        reid_similarity_threshold=reid_similarity_threshold,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0
    signature_to_concept: dict[tuple[int, ...], int] = {}

    def _concept_id(signature: tuple[int, ...]) -> int:
        if signature not in signature_to_concept:
            signature_to_concept[signature] = len(signature_to_concept)
        return signature_to_concept[signature]

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
            updates = server.aggregate(uploads)
            download_b = server.download_bytes(updates)
            total_bytes += upload_b + download_b

            for c in clients:
                update = updates.get(c.client_id)
                if update is not None:
                    c.set_personalized_state(update)

        for k in range(K):
            predicted_matrix[k, t] = _concept_id(clients[k].cluster_signature)

    return MethodResult(
        method_name="FedCCFA",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )


def run_cfl_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    eps_1: float = 0.4,
    eps_2: float = 1.6,
    warmup_rounds: int = 20,
    max_clusters: int = 8,
) -> MethodResult:
    """Run CFL and return full results."""
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        CFLClient(k, n_features, n_classes, seed=42 + k)
        for k in range(K)
    ]
    server = CFLServer(
        n_features=n_features,
        n_classes=n_classes,
        seed=42,
        eps_1=eps_1,
        eps_2=eps_2,
        warmup_rounds=warmup_rounds,
        max_clusters=max_clusters,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for client in clients:
        params = server.broadcast(client.client_id)
        if params:
            client.set_model_params(params)
        client._cluster_id = server.client_cluster_map.get(client.client_id, 0)

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
            server.aggregate(uploads, round_idx=t)
            download_b = 0.0
            for client in clients:
                client._cluster_id = server.client_cluster_map.get(client.client_id, 0)
                params = server.broadcast(client.client_id)
                if params:
                    download_b += model_bytes(params)
                    client.set_model_params(params)
            total_bytes += upload_b + download_b

        for k in range(K):
            predicted_matrix[k, t] = server.client_cluster_map.get(k, clients[k]._cluster_id)

    return MethodResult(
        method_name="CFL",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )


def run_fesem_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    n_clusters: int = 3,
) -> MethodResult:
    """Run FeSEM and return full results."""
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FeSEMClient(k, n_features, n_classes, seed=42 + k)
        for k in range(K)
    ]
    server = FeSEMServer(
        n_clusters=n_clusters,
        n_features=n_features,
        n_classes=n_classes,
        seed=42,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for client in clients:
        client.set_cluster_models(server.cluster_models)

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
            for client in clients:
                client.set_cluster_models(updated_clusters)

        for k in range(K):
            predicted_matrix[k, t] = clients[k]._selected_cluster

    return MethodResult(
        method_name="FeSEM",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )


def run_fedrc_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    n_clusters: int = 3,
) -> MethodResult:
    """Run FedRC and return full results."""
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FedRCClient(k, n_features, n_classes, n_clusters=n_clusters, seed=42 + k)
        for k in range(K)
    ]
    server = FedRCServer(
        n_features=n_features,
        n_classes=n_classes,
        n_clusters=n_clusters,
        seed=42,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    cluster_models, cluster_hists = server.broadcast()
    for client in clients:
        client.set_cluster_state(cluster_models, cluster_hists)

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
            server.aggregate(uploads)
            cluster_models, cluster_hists = server.broadcast()
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b
            for client in clients:
                client.set_cluster_state(cluster_models, cluster_hists)

        for k in range(K):
            predicted_matrix[k, t] = clients[k]._selected_cluster

    return MethodResult(
        method_name="FedRC",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
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


def run_atp_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    base_lr: float = 0.05,
    meta_lr: float = 0.15,
) -> MethodResult:
    """Run ATP and return full results."""
    result = _run_atp_impl(
        dataset,
        federation_every=federation_every,
        base_lr=base_lr,
        meta_lr=meta_lr,
    )
    return _from_external_result(result, method_name="ATP")


def run_flux_full(
    dataset: DriftDataset,
    federation_every: int = 1,
) -> MethodResult:
    """Run FLUX and return full results."""
    result = _run_flux_impl(
        dataset,
        federation_every=federation_every,
    )
    return _from_external_result(
        result,
        method_name="FLUX",
        predicted_attr="predicted_cluster_matrix",
    )


def run_flux_prior_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    n_clusters: int = 3,
) -> MethodResult:
    """Run FLUX-prior and return full results."""
    result = _run_flux_impl(
        dataset,
        federation_every=federation_every,
        prior_n_clusters=n_clusters,
    )
    return _from_external_result(
        result,
        method_name="FLUX-prior",
        predicted_attr="predicted_cluster_matrix",
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


# ---------------------------------------------------------------------------
# Shrinkage baselines (isotropic / anisotropic)
# ---------------------------------------------------------------------------


def _dict_to_flat(params: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate param dict values into a single flat vector."""
    return np.concatenate([v.ravel() for v in params.values()])


def _flat_to_dict(
    flat: np.ndarray,
    template: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reshape a flat vector back into the param dict structure."""
    result: dict[str, np.ndarray] = {}
    offset = 0
    for key, v in template.items():
        size = v.size
        result[key] = flat[offset : offset + size].reshape(v.shape)
        offset += size
    return result


def run_shrinkage_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    use_anisotropic: bool = True,
    lr: float = 0.05,
    n_epochs: int = 5,
) -> MethodResult:
    """Run Oracle concept-level with empirical-Bayes shrinkage.

    Each federation round:
    1. Train per-concept linear heads using oracle concept labels.
    2. Train a global linear head.
    3. Apply shrinkage: w_shrunk = (1-λ)*w_concept + λ*w_global.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    use_anisotropic : bool
        If True, compute ``r_eff`` from features and use it in λ.
    lr : float
        Learning rate for SGD training.
    n_epochs : int
        Local training epochs.

    Returns
    -------
    MethodResult
    """
    from ..estimators.shrinkage import (
        ShrinkageEstimator,
        compute_shrinkage_lambda,
        compute_effective_rank,
        estimate_sigma_B2,
    )
    from ..models import TorchLinearClassifier

    K, T, n_features, n_classes = _extract_dims(dataset)
    concept_matrix = dataset.concept_matrix
    C = int(concept_matrix.max()) + 1

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    # One model per concept + one global model.
    concept_models = {
        j: TorchLinearClassifier(
            n_features, n_classes, lr=lr, n_epochs=n_epochs,
        )
        for j in range(C)
    }
    global_model = TorchLinearClassifier(
        n_features, n_classes, lr=lr, n_epochs=n_epochs,
    )

    # Track shrinkage diagnostics for the last round.
    last_lambda_iso = 0.0
    last_lambda_aniso = 0.0
    last_r_eff = float(n_features)

    for t in range(T):
        # --- Evaluate (before training this round) ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            cid = int(concept_matrix[k, t])
            predicted_matrix[k, t] = cid
            preds = concept_models[cid].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        # --- Train (per-concept + global) ---
        concept_data: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            j: [] for j in range(C)
        }
        all_X_list: list[np.ndarray] = []
        all_y_list: list[np.ndarray] = []
        for k in range(K):
            X, y = dataset.data[(k, t)]
            cid = int(concept_matrix[k, t])
            concept_data[cid].append((X, y))
            all_X_list.append(X)
            all_y_list.append(y)

        for j in range(C):
            if concept_data[j]:
                Xj = np.concatenate([xy[0] for xy in concept_data[j]])
                yj = np.concatenate([xy[1] for xy in concept_data[j]])
                concept_models[j].fit(Xj, yj)

        X_all = np.concatenate(all_X_list)
        y_all = np.concatenate(all_y_list)
        global_model.fit(X_all, y_all)

        # --- Federate with shrinkage ---
        if (t + 1) % federation_every == 0 and t < T - 1:
            concept_params = [concept_models[j].get_params() for j in range(C)]
            global_params = global_model.get_params()

            # Flatten to vectors for shrinkage math.
            concept_vecs = [_dict_to_flat(p) for p in concept_params]
            global_vec = _dict_to_flat(global_params)

            # Estimate noise variance from prediction residuals.
            logits = global_model.predict(X_all)
            sigma2 = float(np.mean((logits != y_all).astype(np.float64)))
            sigma2 = max(sigma2, 0.01)  # floor
            n_per_client = len(dataset.data[(0, t)][0])

            # Compute r_eff from features.
            r_eff = compute_effective_rank(X_all) if use_anisotropic else float(n_features)

            # Estimate between-concept variance.
            sigma_B2 = estimate_sigma_B2(concept_vecs, sigma2, K, C, n_per_client)

            d_param = concept_vecs[0].shape[0]
            lambda_iso = compute_shrinkage_lambda(sigma2, sigma_B2, K, C, n_per_client, d_param)
            lambda_aniso = compute_shrinkage_lambda(sigma2, sigma_B2, K, C, n_per_client, r_eff)
            lam = lambda_aniso if use_anisotropic else lambda_iso

            last_lambda_iso = lambda_iso
            last_lambda_aniso = lambda_aniso
            last_r_eff = r_eff

            # Apply shrinkage.
            for j in range(C):
                w_shrunk = (1 - lam) * concept_vecs[j] + lam * global_vec
                concept_models[j].set_params(_flat_to_dict(w_shrunk, concept_params[j]))

            param_bytes = model_bytes(concept_params[0])
            total_bytes += K * param_bytes * 2

    name = "Shrinkage-aniso" if use_anisotropic else "Shrinkage-iso"
    result = MethodResult(
        method_name=name,
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )
    # Attach diagnostics for experiment scripts.
    result.lambda_iso = last_lambda_iso  # type: ignore[attr-defined]
    result.lambda_aniso = last_lambda_aniso  # type: ignore[attr-defined]
    result.r_eff = last_r_eff  # type: ignore[attr-defined]
    return result
