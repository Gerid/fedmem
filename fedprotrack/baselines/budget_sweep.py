from __future__ import annotations

"""Budget sweep: compare FedAvg-Full, FedProto, and TrackedSummary across
federation frequencies to produce Pareto-style (bytes, AUC) trade-off curves.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..drift_generator.generator import DriftDataset
from ..federation.aggregator import FedAvgAggregator
from ..metrics.budget_metrics import compute_accuracy_auc
from .comm_tracker import model_bytes, fingerprint_bytes, prototype_bytes
from .fedproto import FedProtoClient, FedProtoAggregator
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

    # Local sklearn models per client
    models: list[LogisticRegression | None] = [None] * K
    # Current model params per client (for communication accounting)
    client_params: list[dict[str, np.ndarray]] = [{}] * K

    accuracy_curve = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    # Helper: build params dict from fitted model
    def _params_from_model(model: LogisticRegression) -> dict[str, np.ndarray]:
        return {
            "coef": model.coef_.flatten().copy(),
            "intercept": model.intercept_.flatten().copy(),
        }

    for t in range(T):
        # --- 1. Predict (use current model) ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            if models[k] is None:
                preds = np.zeros(len(X), dtype=np.int64)
            else:
                preds = models[k].predict(X).astype(np.int64)
            accuracy_curve[k, t] = _accuracy(y, preds)

        # --- 2. Fit on current batch ---
        for k in range(K):
            X, y = dataset.data[(k, t)]
            classes = np.arange(n_classes)
            # Augment to stabilize coef_ shape when classes are missing
            if len(np.unique(y)) < n_classes:
                missing = [c for c in classes if c not in np.unique(y)]
                X_aug = np.vstack([X] + [X[[0]] * 0.0 for _ in missing])
                y_aug = np.concatenate([y, np.array(missing, dtype=y.dtype)])
            else:
                X_aug, y_aug = X, y

            model = LogisticRegression(
                max_iter=200,
                random_state=42 + k,
                solver="lbfgs",
                
            )
            model.fit(X_aug, y_aug)
            models[k] = model
            client_params[k] = _params_from_model(model)

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

                # Distribute global model back to all clients
                for k in range(K):
                    coef = global_params["coef"]
                    intercept = global_params["intercept"]
                    expected_rows = 1 if n_classes == 2 else n_classes
                    coef_2d = coef.reshape(expected_rows, n_features)
                    if models[k] is not None:
                        models[k].coef_ = coef_2d.copy()
                        models[k].intercept_ = intercept.copy()
                    client_params[k] = {
                        "coef": coef.copy(),
                        "intercept": intercept.copy(),
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
# Public API
# ---------------------------------------------------------------------------

def run_budget_sweep(
    dataset: DriftDataset,
    federation_every_values: list[int] | None = None,
    similarity_threshold: float = 0.5,
) -> list[BudgetPoint]:
    """Run all three methods across a range of federation frequencies.

    For each combination of method and ``federation_every`` value one
    ``BudgetPoint`` is produced, yielding ``3 * len(federation_every_values)``
    points in total.

    Parameters
    ----------
    dataset : DriftDataset
        The dataset to simulate on.
    federation_every_values : list of int, optional
        Federation is triggered every this many time steps. Defaults to
        ``[1, 2, 5, 10]``.
    similarity_threshold : float
        Cosine similarity threshold used by TrackedSummary clustering.
        Default 0.5.

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
        results.append(_run_fedproto(dataset, fe))
        results.append(_run_tracked_summary(dataset, fe, similarity_threshold))

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
