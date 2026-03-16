"""End-to-end FedProTrack runner using the two-phase protocol.

Orchestrates a complete federated concept drift simulation:
  1. Clients build fingerprints and train local models
  2. Phase A: fingerprint exchange for concept identification
  3. Phase B: model aggregation within concept clusters
  4. Metrics collection and communication byte accounting
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import SGDClassifier

from ..concept_tracker.fingerprint import ConceptFingerprint
from ..drift_detector import BaseDriftDetector
from ..drift_generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from .two_phase_protocol import TwoPhaseConfig, TwoPhaseFedProTrack


def _make_detector(name: str) -> BaseDriftDetector:
    """Create a drift detector by name."""
    from ..drift_detector import (
        ADWINDetector,
        KSWINDetector,
        NoDriftDetector,
        PageHinkleyDetector,
    )

    detectors: dict[str, type] = {
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


@dataclass
class FedProTrackResult:
    """Complete result from a FedProTrack simulation.

    Parameters
    ----------
    accuracy_matrix : np.ndarray
        Shape (K, T). Per-client per-step classification accuracy.
    predicted_concept_matrix : np.ndarray
        Shape (K, T). Predicted concept IDs.
    true_concept_matrix : np.ndarray
        Shape (K, T). Ground-truth concept IDs.
    total_bytes : float
        Total communication cost (Phase A + Phase B).
    phase_a_bytes : float
        Communication cost for Phase A only.
    phase_b_bytes : float
        Communication cost for Phase B only.
    mean_accuracy : float
        Mean accuracy across all clients and steps.
    final_accuracy : float
        Mean accuracy at the last time step.
    method_name : str
        Human-readable name.
    """

    accuracy_matrix: np.ndarray
    predicted_concept_matrix: np.ndarray
    true_concept_matrix: np.ndarray
    total_bytes: float
    phase_a_bytes: float
    phase_b_bytes: float
    mean_accuracy: float
    final_accuracy: float
    method_name: str = "FedProTrack"

    def to_experiment_log(self) -> ExperimentLog:
        """Convert to ExperimentLog for unified metrics computation.

        Returns
        -------
        ExperimentLog
        """
        return ExperimentLog(
            ground_truth=self.true_concept_matrix,
            predicted=self.predicted_concept_matrix,
            accuracy_curve=self.accuracy_matrix,
            total_bytes=self.total_bytes,
            method_name=self.method_name,
        )


class FedProTrackRunner:
    """End-to-end FedProTrack simulation runner.

    Parameters
    ----------
    config : TwoPhaseConfig
        Two-phase protocol configuration.
    federation_every : int
        Run federation (Phase A + Phase B) every this many steps.
    detector_name : str
        Drift detector name ("ADWIN", "PageHinkley", "KSWIN", "NoDrift").
    seed : int
        Base random seed.
    """

    def __init__(
        self,
        config: TwoPhaseConfig | None = None,
        federation_every: int = 1,
        detector_name: str = "ADWIN",
        seed: int = 42,
    ):
        self.config = config or TwoPhaseConfig()
        self.federation_every = federation_every
        self.detector_name = detector_name
        self.seed = seed

    def run(self, dataset: DriftDataset) -> FedProTrackResult:
        """Execute the full FedProTrack simulation.

        Parameters
        ----------
        dataset : DriftDataset
            Pre-generated dataset with concept matrix and data.

        Returns
        -------
        FedProTrackResult
        """
        gc = dataset.config
        K, T = gc.K, gc.T
        n_features = _infer_n_features(gc.generator_type)

        # Update config dimensions
        cfg = TwoPhaseConfig(
            omega=self.config.omega,
            kappa=self.config.kappa,
            novelty_threshold=self.config.novelty_threshold,
            loss_novelty_threshold=self.config.loss_novelty_threshold,
            sticky_dampening=self.config.sticky_dampening,
            sticky_posterior_gate=self.config.sticky_posterior_gate,
            model_loss_weight=self.config.model_loss_weight,
            post_spawn_merge=self.config.post_spawn_merge,
            merge_threshold=self.config.merge_threshold,
            min_count=self.config.min_count,
            max_concepts=self.config.max_concepts,
            merge_every=self.config.merge_every,
            shrink_every=self.config.shrink_every,
            n_features=n_features,
            n_classes=2,
        )

        protocol = TwoPhaseFedProTrack(cfg)

        # Per-client state
        detectors = [_make_detector(self.detector_name) for _ in range(K)]
        model_params: list[dict[str, np.ndarray]] = [{}] * K

        # Results
        accuracy_matrix = np.zeros((K, T), dtype=np.float64)
        predicted_matrix = np.zeros((K, T), dtype=np.int32)

        phase_a_bytes = 0.0
        phase_b_bytes = 0.0
        prev_assignments: dict[int, int] | None = None

        for t in range(T):
            # Per-step fingerprints: built fresh each step from current
            # data only, avoiding cross-concept contamination.
            step_fingerprints = [
                ConceptFingerprint(n_features, 2) for _ in range(K)
            ]

            # --- Per-client: predict, detect, fingerprint, train ---
            for k in range(K):
                X, y = dataset.data[(k, t)]
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]

                # 1. Prequential prediction
                if model_params[k]:
                    coef = model_params[k].get("coef")
                    intercept = model_params[k].get("intercept")
                    if coef is not None and intercept is not None:
                        scores = X_test @ coef.T + intercept
                        y_pred = (scores.ravel() > 0).astype(np.int32)
                    else:
                        y_pred = np.zeros(len(y_test), dtype=np.int32)
                else:
                    y_pred = np.zeros(len(y_test), dtype=np.int32)

                acc = float(np.mean(y_pred == y_test)) if len(y_test) > 0 else 0.0
                accuracy_matrix[k, t] = acc

                # 2. Drift detection
                errors = (y_pred != y_test).astype(float)
                is_drift = False
                for e in errors:
                    result = detectors[k].update(e)
                    if result.is_drift:
                        is_drift = True
                        detectors[k].reset()
                        break

                # 3. Build per-step fingerprint (fresh, no accumulation)
                step_fingerprints[k].update(X_train, y_train)

                # 4. Local SGD training
                model = SGDClassifier(
                    loss="log_loss",
                    random_state=self.seed + k * T + t + 10000,
                )
                model.partial_fit(X_train, y_train, classes=[0, 1])

                if not is_drift and model_params[k].get("coef") is not None:
                    # Warm-start with momentum only when no drift detected
                    old_coef = model_params[k]["coef"]
                    old_int = model_params[k]["intercept"]
                    model.coef_ = 0.5 * old_coef + 0.5 * model.coef_
                    model.intercept_ = 0.5 * old_int + 0.5 * model.intercept_

                model_params[k] = {
                    "coef": model.coef_.copy(),
                    "intercept": model.intercept_.copy(),
                }

            # --- Federation ---
            if (t + 1) % self.federation_every == 0:
                # Phase A: fingerprint exchange (per-step fingerprints)
                # Compute per-client model losses (error rates) as a
                # second channel for novelty gating.
                client_model_losses = {
                    k: 1.0 - accuracy_matrix[k, t] for k in range(K)
                }
                client_fps = {k: step_fingerprints[k] for k in range(K)}
                a_result = protocol.phase_a(
                    client_fps, prev_assignments, client_model_losses,
                )
                phase_a_bytes += a_result.total_bytes
                prev_assignments = a_result.assignments

                # Record concept assignments
                for k in range(K):
                    predicted_matrix[k, t] = a_result.assignments.get(k, 0)

                # Phase B: model aggregation
                client_p = {k: model_params[k] for k in range(K) if model_params[k]}
                if client_p:
                    b_result = protocol.phase_b(client_p, a_result.assignments)
                    phase_b_bytes += b_result.total_bytes

                    # Distribute aggregated models
                    for k in range(K):
                        cid = a_result.assignments.get(k)
                        if cid is not None and cid in b_result.aggregated_params:
                            model_params[k] = {
                                key: arr.copy()
                                for key, arr in b_result.aggregated_params[cid].items()
                            }
            else:
                # No federation this step — carry forward previous assignments
                if prev_assignments:
                    for k in range(K):
                        predicted_matrix[k, t] = prev_assignments.get(k, 0)

        total_bytes = phase_a_bytes + phase_b_bytes

        return FedProTrackResult(
            accuracy_matrix=accuracy_matrix,
            predicted_concept_matrix=predicted_matrix,
            true_concept_matrix=dataset.concept_matrix,
            total_bytes=total_bytes,
            phase_a_bytes=phase_a_bytes,
            phase_b_bytes=phase_b_bytes,
            mean_accuracy=float(accuracy_matrix.mean()),
            final_accuracy=float(accuracy_matrix[:, -1].mean()),
        )
