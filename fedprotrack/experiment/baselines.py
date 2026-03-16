"""Baseline experiment runners for comparison.

Provides simplified runners for:
  - LocalOnly: Each client trains independently, no federation
  - FedAvgBaseline: Standard FedAvg, no drift detection
  - OracleBaseline: Uses ground-truth concept IDs (upper bound)
"""

from __future__ import annotations

import numpy as np

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


def _infer_n_features(generator_type: str) -> int:
    return {"sine": 2, "sea": 3, "circle": 2}[generator_type]


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
    K, T = gc.K, gc.T

    n_features = _infer_n_features(gc.generator_type)
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)

    for k in range(K):
        model = TorchLinearClassifier(
            n_features=n_features, n_classes=2,
            lr=0.1, n_epochs=1, seed=42 + k,
        )

        for t in range(T):
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
        tracking_metrics=ConceptTrackingMetrics(),
    )


def run_fedavg_baseline(
    config: ExperimentConfig,
    dataset: DriftDataset | None = None,
) -> ExperimentResult:
    """Standard FedAvg baseline: average all client models every step.

    No drift detection, no concept awareness. Demonstrates the negative
    interference caused by concept heterogeneity.

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
    K, T = gc.K, gc.T
    n_features = _infer_n_features(gc.generator_type)

    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)

    # Global model on GPU
    global_model = TorchLinearClassifier(
        n_features=n_features, n_classes=2,
        lr=0.1, n_epochs=1, seed=42,
    )
    # Per-client local models on GPU
    client_models = [
        TorchLinearClassifier(
            n_features=n_features, n_classes=2,
            lr=0.1, n_epochs=1, seed=42 + k,
        )
        for k in range(K)
    ]

    for t in range(T):
        client_params_list = []

        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            # Predict with global model
            y_pred = global_model.predict(X_test)
            acc_matrix[k, t] = float(np.mean(y_pred == y_test))

            # Train local model from global init
            if t > 0:
                client_models[k].set_params(global_model.get_params())
            client_models[k].partial_fit(X_train, y_train)
            if t > 0:
                client_models[k].blend_params(global_model.get_params(), alpha=0.5)

            client_params_list.append(client_models[k].get_params())

        # Aggregate: simple average
        global_params: dict[str, np.ndarray] = {}
        for key in client_params_list[0]:
            stacked = np.stack([p[key] for p in client_params_list])
            global_params[key] = np.mean(stacked, axis=0)
        global_model.set_params(global_params)

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
        tracking_metrics=ConceptTrackingMetrics(),
    )


def run_oracle_baseline(
    config: ExperimentConfig,
    dataset: DriftDataset | None = None,
) -> ExperimentResult:
    """Oracle baseline: uses ground-truth concept IDs.

    Maintains separate models per true concept. Represents the upper bound
    for concept-aware methods.

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
    K, T = gc.K, gc.T

    n_features = _infer_n_features(gc.generator_type)
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = dataset.concept_matrix.copy()

    # Per-concept models for each client (on GPU)
    client_models: dict[tuple[int, int], TorchLinearClassifier] = {}

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            true_concept = int(dataset.concept_matrix[k, t])
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            key = (k, true_concept)
            if key in client_models:
                y_pred = client_models[key].predict(X_test)
            else:
                y_pred = np.zeros(len(y_test), dtype=np.int32)

            acc_matrix[k, t] = float(np.mean(y_pred == y_test))

            if key not in client_models:
                client_models[key] = TorchLinearClassifier(
                    n_features=n_features, n_classes=2,
                    lr=0.1, n_epochs=1, seed=42 + k,
                )
            client_models[key].partial_fit(X_train, y_train)

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
        tracking_metrics=ConceptTrackingMetrics(),
    )
