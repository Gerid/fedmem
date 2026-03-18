from __future__ import annotations

"""Budget sweep across baseline methods for budget-vs-accuracy frontiers."""

from dataclasses import dataclass

import numpy as np

from ..drift_generator.generator import DriftDataset
from ..federation.aggregator import FedAvgAggregator
from ..models import TorchLinearClassifier
from ..metrics.budget_metrics import compute_accuracy_auc
from .comm_tracker import model_bytes, fingerprint_bytes, prototype_bytes
from .compressed_fedavg import CompressedFedAvgClient, CompressedFedAvgServer
from .feddrift import FedDriftClient, FedDriftServer
from .fedproto import FedProtoClient, FedProtoAggregator
from .flash import FlashClient, FlashAggregator
from .ifca import IFCAClient, IFCAServer
from .tracked_summary import TrackedSummaryClient, TrackedSummaryServer


@dataclass
class BudgetPoint:
    """Single (method, federation_every) -> (bytes, auc) measurement.

    Parameters
    ----------
    method_name : str
        Human-readable name of the federated method.
    federation_every : int
        Federation is triggered every this many time steps.
    total_bytes : float
        Total bytes transmitted (upload + download) across all rounds.
    accuracy_auc : float
        Trapezoidal AUC of the mean per-client accuracy curve.
    """

    method_name: str
    federation_every: int
    total_bytes: float
    accuracy_auc: float


# ---------------------------------------------------------------------------
# Internal helpers: data extraction
# ---------------------------------------------------------------------------

def _extract_dataset_dims(dataset: DriftDataset) -> tuple[int, int, int, int]:
    """Return (K, T, n_features, n_classes) from a DriftDataset.

    Parameters
    ----------
    dataset : DriftDataset

    Returns
    -------
    tuple[int, int, int, int]
        K, T, n_features, n_classes.
    """
    K = dataset.config.K
    T = dataset.config.T
    X0, y0 = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    # Collect all unique labels across the entire dataset to get n_classes
    all_labels: set[int] = set()
    for (k, t), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1
    return K, T, n_features, n_classes


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly predicted labels.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    float
    """
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _budget_point_from_full_result(result: object, federation_every: int) -> BudgetPoint:
    accuracy_matrix = np.asarray(getattr(result, "accuracy_matrix"), dtype=np.float64)
    return BudgetPoint(
        method_name=str(getattr(result, "method_name")),
        federation_every=federation_every,
        total_bytes=float(getattr(result, "total_bytes")),
        accuracy_auc=compute_accuracy_auc(accuracy_matrix),
    )


# ---------------------------------------------------------------------------
# Runner: FedAvg-Full
# ---------------------------------------------------------------------------

def _run_fedavg_full(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    """Run standard FedAvg with full model exchange.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
        Federation triggered every this many steps.

    Returns
    -------
    BudgetPoint
    """
    K, T, n_features, n_classes = _extract_dataset_dims(dataset)
    aggregator = FedAvgAggregator()

    # Local PyTorch models per client (on GPU)
    models = [
        TorchLinearClassifier(
            n_features=n_features, n_classes=n_classes,
            lr=0.01, n_epochs=5, seed=42 + k,
        )
        for k in range(K)
    ]
    # Current model params per client (for communication accounting)
    client_params: list[dict[str, np.ndarray]] = [{}] * K

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        # --- 1. Predict (use current model) ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = models[k].predict(X)
            accuracy_curve[k, t] = _accuracy(y, preds)

        # --- 2. Fit on current batch ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            models[k].fit(X, y)
            client_params[k] = models[k].get_params()

        # --- 3. Federation (if scheduled and not last step) ---
        if (t + 1) % federation_every == 0 and t < T - 1:
            valid = [p for p in client_params if p]
            if valid:
                # Upload cost: K * model_bytes per client
                upload_b = sum(model_bytes(p) for p in valid)
                global_params = aggregator.aggregate(valid)
                # Download cost: K * model_bytes(global)
                download_b = K * model_bytes(global_params)
                total_bytes += upload_b + download_b

                # Distribute global model back to all clients (load to GPU)
                for k in range(K):
                    models[k].set_params(global_params)
                    client_params[k] = {
                        k_: v.copy() for k_, v in global_params.items()
                    }

    auc = compute_accuracy_auc(accuracy_curve)
    return BudgetPoint(
        method_name="FedAvg-Full",
        federation_every=federation_every,
        total_bytes=total_bytes,
        accuracy_auc=auc,
    )


# ---------------------------------------------------------------------------
# Runner: FedProto
# ---------------------------------------------------------------------------

def _run_fedproto(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    """Run FedProto baseline.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
        Federation triggered every this many steps.

    Returns
    -------
    BudgetPoint
    """
    K, T, n_features, n_classes = _extract_dataset_dims(dataset)
    clients = [FedProtoClient(k, n_features, n_classes) for k in range(K)]
    aggregator = FedProtoAggregator()

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        # --- 1. Predict ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_curve[k, t] = _accuracy(y, preds)

        # --- 2. Fit ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        # --- 3. Federation ---
        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            global_protos = aggregator.aggregate(uploads)
            download_b = aggregator.download_bytes(global_protos, K)
            total_bytes += upload_b + download_b

            for c in clients:
                c.set_global_prototypes(global_protos)

    auc = compute_accuracy_auc(accuracy_curve)
    return BudgetPoint(
        method_name="FedProto",
        federation_every=federation_every,
        total_bytes=total_bytes,
        accuracy_auc=auc,
    )


def _run_fedccfa(
    dataset: DriftDataset,
    federation_every: int,
    cluster_eps: float = 0.35,
) -> BudgetPoint:
    """Run FedCCFA baseline.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    cluster_eps : float
        DBSCAN epsilon for label-wise classifier clustering.

    Returns
    -------
    BudgetPoint
    """
    from .runners import run_fedccfa_full

    result = run_fedccfa_full(
        dataset,
        federation_every=federation_every,
        cluster_eps=cluster_eps,
    )
    auc = compute_accuracy_auc(result.accuracy_matrix)
    return BudgetPoint(
        method_name="FedCCFA",
        federation_every=federation_every,
        total_bytes=result.total_bytes,
        accuracy_auc=auc,
    )


def _run_pfedme(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    from .runners import run_pfedme_full

    return _budget_point_from_full_result(
        run_pfedme_full(dataset, federation_every=federation_every),
        federation_every,
    )


def _run_apfl(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    from .runners import run_apfl_full

    return _budget_point_from_full_result(
        run_apfl_full(dataset, federation_every=federation_every),
        federation_every,
    )


def _run_fedem(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    from .runners import run_fedem_full

    return _budget_point_from_full_result(
        run_fedem_full(dataset, federation_every=federation_every),
        federation_every,
    )


def _run_cfl(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    from .runners import run_cfl_full

    return _budget_point_from_full_result(
        run_cfl_full(dataset, federation_every=federation_every),
        federation_every,
    )


def _run_fesem(
    dataset: DriftDataset,
    federation_every: int,
    n_clusters: int = 3,
) -> BudgetPoint:
    from .runners import run_fesem_full

    return _budget_point_from_full_result(
        run_fesem_full(
            dataset,
            federation_every=federation_every,
            n_clusters=n_clusters,
        ),
        federation_every,
    )


def _run_fedrc(
    dataset: DriftDataset,
    federation_every: int,
    n_clusters: int = 3,
) -> BudgetPoint:
    from .runners import run_fedrc_full

    return _budget_point_from_full_result(
        run_fedrc_full(
            dataset,
            federation_every=federation_every,
            n_clusters=n_clusters,
        ),
        federation_every,
    )


def _run_atp(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    from .runners import run_atp_full

    return _budget_point_from_full_result(
        run_atp_full(dataset, federation_every=federation_every),
        federation_every,
    )


def _run_flux(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    from .runners import run_flux_full

    return _budget_point_from_full_result(
        run_flux_full(dataset, federation_every=federation_every),
        federation_every,
    )


def _run_flux_prior(
    dataset: DriftDataset,
    federation_every: int,
    n_clusters: int = 3,
) -> BudgetPoint:
    from .runners import run_flux_prior_full

    return _budget_point_from_full_result(
        run_flux_prior_full(
            dataset,
            federation_every=federation_every,
            n_clusters=n_clusters,
        ),
        federation_every,
    )


# ---------------------------------------------------------------------------
# Runner: TrackedSummary
# ---------------------------------------------------------------------------

def _run_tracked_summary(
    dataset: DriftDataset,
    federation_every: int,
    similarity_threshold: float = 0.5,
) -> BudgetPoint:
    """Run TrackedSummary baseline.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
        Federation triggered every this many steps.
    similarity_threshold : float
        Cosine similarity threshold for fingerprint clustering. Default 0.5.

    Returns
    -------
    BudgetPoint
    """
    K, T, n_features, n_classes = _extract_dataset_dims(dataset)
    clients = [
        TrackedSummaryClient(k, n_features, n_classes, seed=42 + k)
        for k in range(K)
    ]
    server = TrackedSummaryServer(similarity_threshold=similarity_threshold)

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        # --- 1. Predict ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_curve[k, t] = _accuracy(y, preds)

        # --- 2. Fit ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        # --- 3. Federation ---
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

    auc = compute_accuracy_auc(accuracy_curve)
    return BudgetPoint(
        method_name="TrackedSummary",
        federation_every=federation_every,
        total_bytes=total_bytes,
        accuracy_auc=auc,
    )


# ---------------------------------------------------------------------------
# Runner: Flash
# ---------------------------------------------------------------------------

def _run_flash(dataset: DriftDataset, federation_every: int) -> BudgetPoint:
    """Run Flash baseline with drift-aware knowledge distillation.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int

    Returns
    -------
    BudgetPoint
    """
    K, T, n_features, n_classes = _extract_dataset_dims(dataset)
    clients = [
        FlashClient(k, n_features, n_classes, seed=42 + k)
        for k in range(K)
    ]
    aggregator = FlashAggregator()

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_curve[k, t] = _accuracy(y, preds)

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

    auc = compute_accuracy_auc(accuracy_curve)
    return BudgetPoint(
        method_name="Flash",
        federation_every=federation_every,
        total_bytes=total_bytes,
        accuracy_auc=auc,
    )


# ---------------------------------------------------------------------------
# Runner: FedDrift
# ---------------------------------------------------------------------------

def _run_feddrift(
    dataset: DriftDataset,
    federation_every: int,
    similarity_threshold: float = 0.5,
) -> BudgetPoint:
    """Run FedDrift baseline with multi-model drift branching.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    similarity_threshold : float

    Returns
    -------
    BudgetPoint
    """
    K, T, n_features, n_classes = _extract_dataset_dims(dataset)
    clients = [
        FedDriftClient(k, n_features, n_classes,
                       similarity_threshold=similarity_threshold, seed=42 + k)
        for k in range(K)
    ]
    server = FedDriftServer(similarity_threshold=similarity_threshold)

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_curve[k, t] = _accuracy(y, preds)

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

    auc = compute_accuracy_auc(accuracy_curve)
    return BudgetPoint(
        method_name="FedDrift",
        federation_every=federation_every,
        total_bytes=total_bytes,
        accuracy_auc=auc,
    )


# ---------------------------------------------------------------------------
# Runner: IFCA
# ---------------------------------------------------------------------------

def _run_ifca(
    dataset: DriftDataset,
    federation_every: int,
    n_clusters: int = 3,
) -> BudgetPoint:
    """Run IFCA baseline with iterative federated clustering.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    n_clusters : int

    Returns
    -------
    BudgetPoint
    """
    K, T, n_features, n_classes = _extract_dataset_dims(dataset)
    clients = [IFCAClient(k, n_features, n_classes, seed=42 + k) for k in range(K)]
    server = IFCAServer(
        n_clusters=n_clusters, n_features=n_features,
        n_classes=n_classes, seed=42,
    )

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for c in clients:
        c.set_cluster_models(server.cluster_models)

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_curve[k, t] = _accuracy(y, preds)

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            updated = server.aggregate(uploads)
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b
            for c in clients:
                c.set_cluster_models(updated)

    auc = compute_accuracy_auc(accuracy_curve)
    return BudgetPoint(
        method_name="IFCA",
        federation_every=federation_every,
        total_bytes=total_bytes,
        accuracy_auc=auc,
    )


# ---------------------------------------------------------------------------
# Runner: CompressedFedAvg
# ---------------------------------------------------------------------------

def _run_compressed_fedavg(
    dataset: DriftDataset,
    federation_every: int,
    topk_fraction: float = 0.3,
) -> BudgetPoint:
    """Run CompressedFedAvg baseline with top-k sparsification.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
    topk_fraction : float

    Returns
    -------
    BudgetPoint
    """
    K, T, n_features, n_classes = _extract_dataset_dims(dataset)
    clients = [
        CompressedFedAvgClient(k, n_features, n_classes,
                               topk_fraction=topk_fraction, seed=42 + k)
        for k in range(K)
    ]
    server = CompressedFedAvgServer(n_features, n_classes)

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_curve[k, t] = _accuracy(y, preds)

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

    auc = compute_accuracy_auc(accuracy_curve)
    return BudgetPoint(
        method_name="CompressedFedAvg",
        federation_every=federation_every,
        total_bytes=total_bytes,
        accuracy_auc=auc,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_budget_sweep(
    dataset: DriftDataset,
    federation_every_values: list[int] | None = None,
    similarity_threshold: float = 0.5,
) -> list[BudgetPoint]:
    """Run all baseline methods across a range of federation frequencies.

    For each combination of method and ``federation_every`` value one
    ``BudgetPoint`` is produced, yielding ``17 * len(federation_every_values)``
    points in total.

    Parameters
    ----------
    dataset : DriftDataset
        The dataset to simulate on.
    federation_every_values : list of int, optional
        Federation is triggered every this many time steps. Defaults to
        ``[1, 2, 5, 10]``.
    similarity_threshold : float
        Cosine similarity threshold used by TrackedSummary / FedDrift
        clustering. Default 0.5.

    Returns
    -------
    list[BudgetPoint]
        One BudgetPoint per (method, federation_every) pair.
    """
    if federation_every_values is None:
        federation_every_values = [1, 2, 5, 10]

    results: list[BudgetPoint] = []
    for fe in federation_every_values:
        results.append(_run_fedavg_full(dataset, fe))
        results.append(_run_ifca(dataset, fe))
        results.append(_run_fedrc(dataset, fe))
        results.append(_run_fedem(dataset, fe))
        results.append(_run_fesem(dataset, fe))
        results.append(_run_feddrift(dataset, fe, similarity_threshold))
        results.append(_run_cfl(dataset, fe))
        results.append(_run_pfedme(dataset, fe))
        results.append(_run_apfl(dataset, fe))
        results.append(_run_atp(dataset, fe))
        results.append(_run_flux(dataset, fe))
        results.append(_run_flux_prior(dataset, fe))
        results.append(_run_fedproto(dataset, fe))
        results.append(_run_fedccfa(dataset, fe))
        results.append(_run_tracked_summary(dataset, fe, similarity_threshold))
        results.append(_run_flash(dataset, fe))
        results.append(_run_compressed_fedavg(dataset, fe))

    return results


def find_crossover_points(
    points_a: list[BudgetPoint],
    points_b: list[BudgetPoint],
) -> list[tuple[float, float]]:
    """Find communication budget crossover points between two methods.

    Linearly interpolates both methods' curves (sorted by ``total_bytes``) and
    returns every (bytes, auc) coordinate where the AUC curves of method A and
    method B intersect.

    Parameters
    ----------
    points_a : list[BudgetPoint]
        Budget points for method A (need not be pre-sorted).
    points_b : list[BudgetPoint]
        Budget points for method B (need not be pre-sorted).

    Returns
    -------
    list of (bytes, auc) tuples
        Each entry is a crossover coordinate.  Returns an empty list if the
        curves do not intersect or if either input has fewer than two points.
    """
    if len(points_a) < 2 or len(points_b) < 2:
        return []

    # Sort both by total_bytes
    sa = sorted(points_a, key=lambda p: p.total_bytes)
    sb = sorted(points_b, key=lambda p: p.total_bytes)

    bytes_a = np.array([p.total_bytes for p in sa], dtype=np.float64)
    auc_a = np.array([p.accuracy_auc for p in sa], dtype=np.float64)

    bytes_b = np.array([p.total_bytes for p in sb], dtype=np.float64)
    auc_b = np.array([p.accuracy_auc for p in sb], dtype=np.float64)

    # Build a common grid spanning the overlapping byte range
    lo = max(bytes_a[0], bytes_b[0])
    hi = min(bytes_a[-1], bytes_b[-1])
    if hi <= lo:
        return []

    n_grid = max(len(bytes_a), len(bytes_b)) * 10
    grid = np.linspace(lo, hi, n_grid)

    interp_a = np.interp(grid, bytes_a, auc_a)
    interp_b = np.interp(grid, bytes_b, auc_b)

    diff = interp_a - interp_b
    crossovers: list[tuple[float, float]] = []

    for i in range(len(diff) - 1):
        if diff[i] == 0.0:
            crossovers.append((float(grid[i]), float(interp_a[i])))
        elif diff[i] * diff[i + 1] < 0:
            # Linear interpolation within [grid[i], grid[i+1]]
            t = diff[i] / (diff[i] - diff[i + 1])
            cross_bytes = grid[i] + t * (grid[i + 1] - grid[i])
            cross_auc = interp_a[i] + t * (interp_a[i + 1] - interp_a[i])
            crossovers.append((float(cross_bytes), float(cross_auc)))

    return crossovers
