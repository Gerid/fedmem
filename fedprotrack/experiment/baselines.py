"""Baseline experiment runners for comparison.

Provides simplified runners for:
  - LocalOnly: Each client trains independently, no federation
  - FedAvgBaseline: Standard FedAvg, no drift detection
  - OracleBaseline: Uses ground-truth concept IDs (reference baseline;
    not a strict upper bound --- see run_oracle_baseline docstring)
"""

from __future__ import annotations

import numpy as np

from ..baselines.comm_tracker import model_bytes
from ..drift_generator import DriftDataset, GeneratorConfig, generate_drift_dataset
from ..models import TorchLinearClassifier
from ..evaluation.metrics import (
    compute_prequential_accuracy,
    compute_concept_tracking_accuracy,
    compute_forgetting_measure,
    compute_backward_transfer,
    ConceptTrackingMetrics,
)
from .runner import ExperimentConfig, ExperimentResult


def _infer_n_features(generator_type: str | None, dataset: DriftDataset | None = None) -> int:
    if dataset is not None and dataset.data:
        first_key = next(iter(dataset.data))
        X, _ = dataset.data[first_key]
        return X.shape[1]
    return {"sine": 2, "sea": 3, "circle": 2}.get(generator_type or "", 2)


def _infer_n_classes(dataset: DriftDataset) -> int:
    all_labels: set[int] = set()
    for _, y in dataset.data.values():
        all_labels.update(int(v) for v in np.unique(y))
    if not all_labels:
        return 2
    return max(all_labels) + 1


def run_local_only(
    config: ExperimentConfig,
    dataset: DriftDataset | None = None,
) -> ExperimentResult:
    """Local-only baseline: each client trains independently.

    No drift detection, no federation. Each client simply does online SGD
    on its own data stream.

    Parameters
    ----------
    config : ExperimentConfig
    dataset : DriftDataset, optional

    Returns
    -------
    ExperimentResult
    """
    if dataset is None:
        dataset = generate_drift_dataset(config.generator_config)

    gc = config.generator_config
    if dataset is not None and hasattr(dataset, "concept_matrix"):
        K, T = dataset.concept_matrix.shape
    else:
        K, T = gc.K, gc.T

    n_features = _infer_n_features(getattr(gc, "generator_type", None), dataset)
    n_classes = _infer_n_classes(dataset)
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)

    for k in range(K):
        model = TorchLinearClassifier(
            n_features=n_features, n_classes=n_classes,
            lr=0.1, n_epochs=1, seed=42 + k,
        )

        for t in range(T):
            if dataset.test_data is not None:
                X_test, y_test = dataset.eval_batch(k, t)
                X_train, y_train = dataset.data[(k, t)]
            else:
                X, y = dataset.data[(k, t)]
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]

            y_pred = model.predict(X_test)

            acc_matrix[k, t] = float(np.mean(y_pred == y_test))
            predicted_matrix[k, t] = 0  # no concept tracking

            model.partial_fit(X_train, y_train)

    return ExperimentResult(
        config=config,
        method_name="LocalOnly",
        accuracy_matrix=acc_matrix,
        concept_tracking_accuracy=0.0,
        predicted_concept_matrix=predicted_matrix,
        true_concept_matrix=dataset.concept_matrix,
        mean_accuracy=float(acc_matrix.mean()),
        final_accuracy=float(acc_matrix[:, -1].mean()),
        forgetting=0.0,
        backward_transfer=0.0,
        total_bytes=0.0,
        tracking_metrics=ConceptTrackingMetrics(),
    )


def run_fedavg_baseline(
    config: ExperimentConfig,
    dataset: DriftDataset | None = None,
    *,
    lr: float = 0.1,
    n_epochs: int = 1,
    seed: int = 42,
) -> ExperimentResult:
    """Standard FedAvg baseline with configurable federation cadence.

    No drift detection, no concept awareness. Demonstrates the negative
    interference caused by concept heterogeneity.

    Parameters
    ----------
    config : ExperimentConfig
    dataset : DriftDataset, optional
    lr : float
        Local learning rate for each client.
    n_epochs : int
        Local epochs per client update.
    seed : int
        Base seed for client model initialization.

    Returns
    -------
    ExperimentResult
    """
    if dataset is None:
        dataset = generate_drift_dataset(config.generator_config)

    gc = config.generator_config
    if dataset is not None and hasattr(dataset, "concept_matrix"):
        K, T = dataset.concept_matrix.shape
    else:
        K, T = gc.K, gc.T
    n_features = _infer_n_features(getattr(gc, "generator_type", None), dataset)
    n_classes = _infer_n_classes(dataset)
    federation_every = max(1, int(config.federation_every))

    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    models = [
        TorchLinearClassifier(
            n_features=n_features, n_classes=n_classes,
            lr=lr, n_epochs=n_epochs, seed=seed + k,
        )
        for k in range(K)
    ]

    for t in range(T):
        client_params_list: list[dict[str, np.ndarray]] = []

        for k in range(K):
            if dataset.test_data is not None:
                X_test, y_test = dataset.eval_batch(k, t)
                X_train, y_train = dataset.data[(k, t)]
            else:
                X, y = dataset.data[(k, t)]
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]

            # Predict with the current local model copy.
            y_pred = models[k].predict(X_test)
            acc_matrix[k, t] = float(np.mean(y_pred == y_test))

            if n_epochs > 1:
                models[k].fit(X_train, y_train)
            else:
                models[k].partial_fit(X_train, y_train)
            client_params_list.append(models[k].get_params())

        if (t + 1) % federation_every == 0 and t < T - 1:
            global_params: dict[str, np.ndarray] = {}
            for key in client_params_list[0]:
                stacked = np.stack([p[key] for p in client_params_list])
                global_params[key] = np.mean(stacked, axis=0)

            one_model_bytes = model_bytes(global_params)
            total_bytes += K * one_model_bytes + K * one_model_bytes

            for model in models:
                model.set_params(global_params)

    return ExperimentResult(
        config=config,
        method_name="FedAvg",
        accuracy_matrix=acc_matrix,
        concept_tracking_accuracy=0.0,
        predicted_concept_matrix=predicted_matrix,
        true_concept_matrix=dataset.concept_matrix,
        mean_accuracy=float(acc_matrix.mean()),
        final_accuracy=float(acc_matrix[:, -1].mean()),
        forgetting=0.0,
        backward_transfer=0.0,
        total_bytes=total_bytes,
        tracking_metrics=ConceptTrackingMetrics(),
    )


def run_oracle_baseline(
    config: ExperimentConfig,
    dataset: DriftDataset | None = None,
    *,
    lr: float = 0.1,
    n_epochs: int = 5,
    seed: int = 42,
) -> ExperimentResult:
    """Oracle baseline: uses ground-truth concept IDs.

    Maintains one federated model per true concept and aggregates across
    all clients currently assigned to that concept. This is a **reference
    baseline** for concept-aware clustering methods: it assumes access to
    the true concept identity at every step and performs per-concept
    aggregation under the shared training budget. It is *not* a strict
    upper bound --- fewer samples per concept group (K/C vs K) makes it
    possible for methods with implicit shrinkage (Corollary "Implicit
    Shrinkage") or memory reuse (FedProTrack) to match or exceed Oracle
    when K/C is small, and Oracle can underperform FedAvg when per-concept
    sample sizes are too small for the per-concept task complexity (see
    fMoW 62-class discussion in the paper).

    Parameters
    ----------
    config : ExperimentConfig
    dataset : DriftDataset, optional
    lr : float
        Local learning rate for each client. Default 0.1.
    n_epochs : int
        Local epochs per client update. Default 5.
    seed : int
        Base seed for client model initialization. Default 42.

    Returns
    -------
    ExperimentResult
    """
    if dataset is None:
        dataset = generate_drift_dataset(config.generator_config)

    gc = config.generator_config
    if dataset is not None and hasattr(dataset, "concept_matrix"):
        K, T = dataset.concept_matrix.shape
    else:
        K, T = gc.K, gc.T

    n_features = _infer_n_features(getattr(gc, "generator_type", None), dataset)
    n_classes = _infer_n_classes(dataset)
    federation_every = max(1, int(config.federation_every))
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = dataset.concept_matrix.copy()
    total_bytes = 0.0

    # Persistent per-concept models (not recreated each step)
    concept_models: dict[int, TorchLinearClassifier] = {}

    def _get_concept_model(concept_id: int, client_k: int) -> TorchLinearClassifier:
        if concept_id not in concept_models:
            concept_models[concept_id] = TorchLinearClassifier(
                n_features=n_features,
                n_classes=n_classes,
                lr=lr,
                n_epochs=n_epochs,
                seed=seed + concept_id * 1000 + client_k,
            )
        return concept_models[concept_id]

    concept_params: dict[int, dict[str, np.ndarray]] = {}

    for t in range(T):
        uploads_by_concept: dict[int, list[dict[str, np.ndarray]]] = {}
        for k in range(K):
            true_concept = int(dataset.concept_matrix[k, t])
            if dataset.test_data is not None:
                X_test, y_test = dataset.eval_batch(k, t)
                X_train, y_train = dataset.data[(k, t)]
            else:
                X, y = dataset.data[(k, t)]
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]

            model = _get_concept_model(true_concept, k)
            if true_concept in concept_params:
                model.set_params(concept_params[true_concept])

            y_pred = model.predict(X_test)

            acc_matrix[k, t] = float(np.mean(y_pred == y_test))
            if n_epochs > 1:
                model.fit(X_train, y_train)
            else:
                model.partial_fit(X_train, y_train)
            uploads_by_concept.setdefault(true_concept, []).append(model.get_params())

        if (t + 1) % federation_every == 0 and t < T - 1:
            for concept_id, concept_uploads in uploads_by_concept.items():
                aggregated: dict[str, np.ndarray] = {}
                for key in concept_uploads[0]:
                    stacked = np.stack([params[key] for params in concept_uploads])
                    aggregated[key] = np.mean(stacked, axis=0)
                concept_params[concept_id] = aggregated
                concept_model_bytes = model_bytes(aggregated)
                n_clients = len(concept_uploads)
                total_bytes += n_clients * concept_model_bytes + n_clients * concept_model_bytes

    return ExperimentResult(
        config=config,
        method_name="Oracle",
        accuracy_matrix=acc_matrix,
        concept_tracking_accuracy=1.0,
        predicted_concept_matrix=predicted_matrix,
        true_concept_matrix=dataset.concept_matrix,
        mean_accuracy=float(acc_matrix.mean()),
        final_accuracy=float(acc_matrix[:, -1].mean()),
        forgetting=0.0,
        backward_transfer=0.0,
        total_bytes=total_bytes,
        tracking_metrics=ConceptTrackingMetrics(),
    )
