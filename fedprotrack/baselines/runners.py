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
from .fedccfa_impl import FedCCFAImplClient, FedCCFAImplServer
from .feddrift import FedDriftClient, FedDriftServer
from .fedem import run_fedem_full as _run_fedem_impl
from .fedproto import FedProtoClient, FedProtoAggregator
from .fedrc import FedRCClient, FedRCServer
from .fesem import FeSEMClient, FeSEMServer
from .fedprox import run_fedprox_full as _run_fedprox_impl
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


def run_fedccfa_impl_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    n_clusters_max: int = 10,
    dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 1,
    gamma: float = 20.0,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run FedCCFA-Impl (DBSCAN row clustering + feature alignment).

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    n_clusters_max : int
        Informational upper bound on clusters.
    dbscan_eps : float
        DBSCAN eps.
    dbscan_min_samples : int
        DBSCAN min_samples.
    gamma : float
        Feature-alignment loss weight.
    lr : float
        Local learning rate.
    n_epochs : int
        Local epochs.

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        FedCCFAImplClient(
            k, n_features, n_classes,
            gamma=gamma, lr=lr, n_epochs=n_epochs,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = FedCCFAImplServer(
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_max=n_clusters_max,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        seed=42,
    )

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
            updates = server.aggregate(uploads)
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b

            for c in clients:
                update = updates.get(c.client_id)
                if update is not None:
                    c.set_update(update)

        for k in range(K):
            predicted_matrix[k, t] = clients[k].cluster_id

    return MethodResult(
        method_name="FedCCFA-Impl",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_matrix,
        total_bytes=total_bytes,
    )


def run_fedprox_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    mu: float = 0.01,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run FedProx and return full results.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    mu : float
        Proximal regularisation strength.
    lr : float
        Local learning rate.
    n_epochs : int
        Local epochs.

    Returns
    -------
    MethodResult
    """
    result = _run_fedprox_impl(
        dataset,
        federation_every=federation_every,
        mu=mu,
        lr=lr,
        n_epochs=n_epochs,
    )
    return _from_external_result(result, method_name="FedProx")
