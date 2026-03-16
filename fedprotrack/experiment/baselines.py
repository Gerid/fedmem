"""Baseline experiment runners for comparison.

Provides simplified runners for:
  - LocalOnly: Each client trains independently, no federation
  - FedAvgBaseline: Standard FedAvg, no drift detection
  - OracleBaseline: Uses ground-truth concept IDs (upper bound)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from ..drift_generator import DriftDataset, GeneratorConfig, generate_drift_dataset
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

    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)

    for k in range(K):
        model = SGDClassifier(loss="log_loss", random_state=42)
        initialized = False

        for t in range(T):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            if initialized:
                y_pred = model.predict(X_test)
            else:
                y_pred = np.zeros(len(y_test), dtype=np.int32)

            acc_matrix[k, t] = float(np.mean(y_pred == y_test))
            predicted_matrix[k, t] = 0  # no concept tracking

            model.partial_fit(X_train, y_train, classes=[0, 1])
            initialized = True

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

    # Shared global model params
    global_coef = np.zeros((1, n_features), dtype=np.float64)
    global_intercept = np.zeros(1, dtype=np.float64)

    for t in range(T):
        client_coefs = []
        client_intercepts = []

        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            # Predict with global model
            if t > 0:
                scores = X_test @ global_coef.T + global_intercept
                y_pred = (scores.ravel() > 0).astype(np.int32)
            else:
                y_pred = np.zeros(len(y_test), dtype=np.int32)

            acc_matrix[k, t] = float(np.mean(y_pred == y_test))

            # Train local model from global init
            model = SGDClassifier(loss="log_loss", random_state=42)
            model.partial_fit(X_train, y_train, classes=[0, 1])
            if t > 0:
                model.coef_ = 0.5 * global_coef + 0.5 * model.coef_
                model.intercept_ = 0.5 * global_intercept + 0.5 * model.intercept_

            client_coefs.append(model.coef_.copy())
            client_intercepts.append(model.intercept_.copy())

        # Aggregate: simple average
        global_coef = np.mean(client_coefs, axis=0)
        global_intercept = np.mean(client_intercepts, axis=0)

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

    acc_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = dataset.concept_matrix.copy()

    # Per-concept models for each client
    client_models: dict[tuple[int, int], SGDClassifier] = {}

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
                client_models[key] = SGDClassifier(loss="log_loss", random_state=42)
            client_models[key].partial_fit(X_train, y_train, classes=[0, 1])

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
