"""Experiment runner — end-to-end federated concept drift simulation.

Orchestrates the full pipeline:
  1. Load or generate a DriftDataset
  2. Initialize clients with drift detectors, concept trackers, and learners
  3. Simulate T time steps of federated learning
  4. Collect and return all metrics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..drift_generator import DriftDataset, GeneratorConfig, generate_drift_dataset
from ..drift_detector import BaseDriftDetector, DriftResult
from ..concept_tracker import ConceptTracker, TrackingResult
from ..federation.aggregator import BaseAggregator, FedAvgAggregator
from ..models import TorchLinearClassifier
from ..evaluation.metrics import (
    StreamingAccuracy,
    ConceptTrackingMetrics,
    compute_prequential_accuracy,
    compute_concept_tracking_accuracy,
    compute_forgetting_measure,
    compute_backward_transfer,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # Data
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)

    # Method
    method_name: str = "fedprotrack"
    detector_name: str = "ADWIN"
    similarity_threshold: float = 0.7
    learner_name: str = "sgd"

    # Federation
    aggregator_name: str = "concept_aware"
    federation_every: int = 1  # aggregate every N time steps

    # Output
    results_dir: str = "results"


@dataclass
class ClientState:
    """Internal state for one client during simulation."""

    client_id: int
    detector: BaseDriftDetector
    tracker: ConceptTracker
    accuracy_tracker: StreamingAccuracy = field(
        default_factory=StreamingAccuracy
    )
    predictions: list[np.ndarray] = field(default_factory=list)
    true_labels: list[np.ndarray] = field(default_factory=list)
    predicted_concepts: list[int] = field(default_factory=list)
    drift_flags: list[bool] = field(default_factory=list)
    novel_flags: list[bool] = field(default_factory=list)
    per_concept_accuracies: dict[int, list[float]] = field(default_factory=dict)
    model_params: dict[str, np.ndarray] = field(default_factory=dict)
    torch_model: TorchLinearClassifier | None = None


@dataclass
class ExperimentResult:
    """Full results from one experiment run."""

    config: ExperimentConfig
    method_name: str

    # Per-client, per-step accuracy matrix (K x T)
    accuracy_matrix: np.ndarray  # shape (K, T)

    # Concept tracking
    concept_tracking_accuracy: float
    predicted_concept_matrix: np.ndarray  # shape (K, T)
    true_concept_matrix: np.ndarray  # shape (K, T)

    # Aggregated metrics
    mean_accuracy: float
    final_accuracy: float
    forgetting: float
    backward_transfer: float

    # Detection metrics
    tracking_metrics: ConceptTrackingMetrics
    total_bytes: float | None = None

    def save(self, path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            "method_name": self.method_name,
            "mean_accuracy": self.mean_accuracy,
            "final_accuracy": self.final_accuracy,
            "concept_tracking_accuracy": self.concept_tracking_accuracy,
            "forgetting": self.forgetting,
            "backward_transfer": self.backward_transfer,
            "total_bytes": self.total_bytes,
            "detection_rate": self.tracking_metrics.detection_rate,
            "false_alarm_rate": self.tracking_metrics.false_alarm_rate,
            "identification_accuracy": self.tracking_metrics.identification_accuracy,
            "mean_detection_delay": self.tracking_metrics.mean_detection_delay,
            "accuracy_matrix": self.accuracy_matrix.tolist(),
            "predicted_concept_matrix": self.predicted_concept_matrix.tolist(),
            "true_concept_matrix": self.true_concept_matrix.tolist(),
        }
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @property
    def summary(self) -> dict[str, float]:
        return {
            "mean_accuracy": self.mean_accuracy,
            "final_accuracy": self.final_accuracy,
            "concept_tracking_accuracy": self.concept_tracking_accuracy,
            "forgetting": self.forgetting,
            "backward_transfer": self.backward_transfer,
            "detection_rate": self.tracking_metrics.detection_rate,
            "identification_accuracy": self.tracking_metrics.identification_accuracy,
        }


def _make_detector(name: str) -> BaseDriftDetector:
    """Create a drift detector by name."""
    from ..drift_detector import ADWINDetector, PageHinkleyDetector, KSWINDetector, NoDriftDetector
    detectors = {
        "ADWIN": ADWINDetector,
        "PageHinkley": PageHinkleyDetector,
        "KSWIN": KSWINDetector,
        "NoDrift": NoDriftDetector,
    }
    if name not in detectors:
        raise ValueError(f"Unknown detector: {name}. Choose from {list(detectors)}")
    return detectors[name]()


def _infer_n_features(generator_type: str) -> int:
    """Infer feature dimensionality from generator type."""
    return {"sine": 2, "sea": 3, "circle": 2}[generator_type]


class ExperimentRunner:
    """Runs a complete federated concept drift experiment.

    This is a simplified simulation that uses lightweight linear models
    internally (no external learner module dependency) to keep the runner
    self-contained and testable.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, dataset: DriftDataset | None = None) -> ExperimentResult:
        """Execute the experiment.

        Parameters
        ----------
        dataset : DriftDataset, optional
            Pre-generated dataset. If None, generates from config.

        Returns
        -------
        ExperimentResult
        """
        if dataset is None:
            dataset = generate_drift_dataset(self.config.generator_config)

        gc = self.config.generator_config
        K, T = gc.K, gc.T
        n_features = _infer_n_features(gc.generator_type)

        # Initialize client states
        clients = self._init_clients(K, n_features)

        # Simulate T time steps
        for t in range(T):
            logger.info(f"Time step {t}/{T}")
            self._simulate_step(t, dataset, clients, n_features)

        # Compile results
        return self._compile_results(clients, dataset, K, T)

    def _init_clients(self, K: int, n_features: int) -> list[ClientState]:
        clients = []
        for k in range(K):
            detector = _make_detector(self.config.detector_name)
            tracker = ConceptTracker(
                n_features=n_features,
                n_classes=2,
                similarity_threshold=self.config.similarity_threshold,
            )
            torch_model = TorchLinearClassifier(
                n_features=n_features,
                n_classes=2,
                lr=0.1,
                n_epochs=1,
                seed=42 + k,
            )
            clients.append(ClientState(
                client_id=k, detector=detector, tracker=tracker,
                torch_model=torch_model,
            ))
        return clients

    def _simulate_step(
        self,
        t: int,
        dataset: DriftDataset,
        clients: list[ClientState],
        n_features: int,
    ) -> None:
        """Simulate one time step across all clients."""
        for cs in clients:
            X, y = dataset.data[(cs.client_id, t)]
            true_concept = int(dataset.concept_matrix[cs.client_id, t])

            # Split batch: first half for test (prequential), second for train
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            # Predict using current torch model on GPU
            y_pred = cs.torch_model.predict(X_test)

            acc = float(np.mean(y_pred == y_test)) if len(y_test) > 0 else 0.0
            cs.accuracy_tracker.update(y_test, y_pred)

            # Drift detection: feed errors
            errors = (y_pred != y_test).astype(float)
            is_drift = False
            for e in errors:
                result = cs.detector.update(e)
                if result.is_drift:
                    is_drift = True
                    break

            # Concept tracking
            is_novel = False
            if t == 0:
                pred_concept = cs.tracker.start(X_train, y_train)
            elif is_drift:
                tracking = cs.tracker.on_drift_detected(X_train, y_train)
                pred_concept = tracking.predicted_concept_id
                is_novel = tracking.is_novel
                cs.detector.reset()
            else:
                pred_concept = cs.tracker.active_concept_id or 0
                cs.tracker.observe(X_train, y_train)

            # Train on GPU
            old_params = cs.torch_model.get_params() if cs.torch_model._fitted else None
            cs.torch_model.partial_fit(X_train, y_train)
            if old_params is not None and not is_drift:
                # Warm start: blend with previous params
                cs.torch_model.blend_params(old_params, alpha=0.5)

            cs.model_params = cs.torch_model.get_params()

            # Record
            cs.predictions.append(y_pred)
            cs.true_labels.append(y_test)
            cs.predicted_concepts.append(pred_concept)
            cs.drift_flags.append(is_drift)
            cs.novel_flags.append(is_novel)

            # Per-concept accuracy tracking
            if pred_concept not in cs.per_concept_accuracies:
                cs.per_concept_accuracies[pred_concept] = []
            cs.per_concept_accuracies[pred_concept].append(acc)

    def _compile_results(
        self,
        clients: list[ClientState],
        dataset: DriftDataset,
        K: int,
        T: int,
    ) -> ExperimentResult:
        """Compile all client results into an ExperimentResult."""
        # Accuracy matrix
        acc_matrix = np.zeros((K, T), dtype=np.float64)
        predicted_matrix = np.zeros((K, T), dtype=np.int32)

        all_predicted = []
        all_true = []

        tracking_metrics = ConceptTrackingMetrics()

        for cs in clients:
            for t in range(T):
                y_pred = cs.predictions[t]
                y_true = cs.true_labels[t]
                acc_matrix[cs.client_id, t] = (
                    float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0
                )
                predicted_matrix[cs.client_id, t] = cs.predicted_concepts[t]

            all_predicted.extend(cs.predicted_concepts)
            all_true.extend(
                [int(dataset.concept_matrix[cs.client_id, t]) for t in range(T)]
            )

            # Count drifts
            for t in range(1, T):
                true_drift = (
                    dataset.concept_matrix[cs.client_id, t]
                    != dataset.concept_matrix[cs.client_id, t - 1]
                )
                if true_drift:
                    tracking_metrics.n_true_drifts += 1
                if cs.drift_flags[t]:
                    tracking_metrics.n_detected_drifts += 1
                    tracking_metrics.n_identification_attempts += 1
                    if cs.predicted_concepts[t] != cs.predicted_concepts[t - 1]:
                        if true_drift:
                            tracking_metrics.n_correct_identifications += 1

        # Aggregate metrics
        cta = compute_concept_tracking_accuracy(all_predicted, all_true)

        # Per-concept forgetting
        all_concept_accs: dict[int, list[float]] = {}
        for cs in clients:
            for cid, accs in cs.per_concept_accuracies.items():
                if cid not in all_concept_accs:
                    all_concept_accs[cid] = []
                all_concept_accs[cid].extend(accs)

        forgetting = compute_forgetting_measure(all_concept_accs)
        bwt = compute_backward_transfer(all_concept_accs)

        mean_acc = float(acc_matrix.mean())
        final_acc = float(acc_matrix[:, -1].mean())

        return ExperimentResult(
            config=self.config,
            method_name=self.config.method_name,
            accuracy_matrix=acc_matrix,
            concept_tracking_accuracy=cta,
            predicted_concept_matrix=predicted_matrix,
            true_concept_matrix=dataset.concept_matrix,
            mean_accuracy=mean_acc,
            final_accuracy=final_acc,
            forgetting=forgetting,
            backward_transfer=bwt,
            tracking_metrics=tracking_metrics,
        )
