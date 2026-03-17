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

from ..concept_tracker.fingerprint import ConceptFingerprint
from ..drift_detector import BaseDriftDetector
from ..drift_generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from ..models import TorchLinearClassifier
from .gibbs import PosteriorAssignment
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


def _infer_n_features(generator_type: str, dataset: DriftDataset | None = None) -> int:
    """Infer feature dimensionality from generator type or dataset.

    Parameters
    ----------
    generator_type : str
    dataset : DriftDataset, optional
        If provided, infer from first data sample.

    Returns
    -------
    int
    """
    if dataset is not None and dataset.data:
        first_key = next(iter(dataset.data))
        X, _ = dataset.data[first_key]
        return X.shape[1]
    return {"sine": 2, "sea": 3, "circle": 2}.get(generator_type, 2)


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
    spawned_concepts: int = 0
    merged_concepts: int = 0
    pruned_concepts: int = 0
    active_concepts: int = 0
    soft_assignments: np.ndarray | None = None
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
            total_bytes=self.total_bytes if self.total_bytes > 0 else None,
            soft_assignments=self.soft_assignments,
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
    event_triggered : bool
        If True, Phase A only runs when at least one client detects drift
        (or at the first step). This reduces communication by skipping
        fingerprint exchange during stable periods. Phase B still runs
        on the ``federation_every`` schedule when assignments exist.
    blend_alpha : float
        Weight for momentum blend that pulls local model params toward the
        previous step's params after training. 0.0 disables it. Default 0.5.
    auto_scale : bool
        If True, automatically scale ``loss_novelty_threshold``,
        ``merge_threshold``, ``max_concepts``, and ``blend_alpha`` based
        on feature dimensionality. Higher-dimensional data (e.g. MNIST
        with 20 PCA features) gets higher novelty thresholds and lower
        merge thresholds to prevent over-spawning. Default False.
    """

    def __init__(
        self,
        config: TwoPhaseConfig | None = None,
        federation_every: int = 1,
        detector_name: str = "ADWIN",
        seed: int = 42,
        event_triggered: bool = False,
        blend_alpha: float = 0.5,
        auto_scale: bool = False,
        lr: float = 0.1,
        n_epochs: int = 1,
    ):
        self.config = config or TwoPhaseConfig()
        self.federation_every = federation_every
        self.detector_name = detector_name
        self.seed = seed
        self.event_triggered = event_triggered
        self.blend_alpha = blend_alpha
        self.auto_scale = auto_scale
        self.lr = lr
        self.n_epochs = n_epochs

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
        n_features = _infer_n_features(gc.generator_type, dataset)

        # --- Dimension-adaptive scaling ---
        # Higher-dimensional features produce noisier fingerprints, requiring
        # more conservative spawning (higher loss_novelty_threshold), more
        # aggressive merging (lower merge_threshold), tighter concept caps,
        # and less momentum blending.
        loss_novelty_th = self.config.loss_novelty_threshold
        merge_th = self.config.merge_threshold
        max_concepts = self.config.max_concepts
        blend_alpha = self.blend_alpha

        if self.auto_scale and n_features > 3:
            # Scale factor: 1.0 at dim=2, ~10.0 at dim=20
            scale = n_features / 2.0
            loss_novelty_th = max(loss_novelty_th, 0.02 * scale)
            merge_th = min(merge_th, max(0.80, 0.98 - 0.01 * (n_features - 2)))
            max_concepts = min(max_concepts, max(6, 20 - n_features // 2))
            blend_alpha = max(0.0, blend_alpha - 0.05 * (n_features - 2))

        # Update config dimensions
        cfg = TwoPhaseConfig(
            omega=self.config.omega,
            kappa=self.config.kappa,
            novelty_threshold=self.config.novelty_threshold,
            loss_novelty_threshold=loss_novelty_th,
            sticky_dampening=self.config.sticky_dampening,
            sticky_posterior_gate=self.config.sticky_posterior_gate,
            model_loss_weight=self.config.model_loss_weight,
            post_spawn_merge=self.config.post_spawn_merge,
            merge_threshold=merge_th,
            min_count=self.config.min_count,
            max_concepts=max_concepts,
            merge_every=self.config.merge_every,
            shrink_every=self.config.shrink_every,
            n_features=n_features,
            n_classes=2,
        )

        protocol = TwoPhaseFedProTrack(cfg)

        # Per-client state
        detectors = [_make_detector(self.detector_name) for _ in range(K)]
        models = [
            TorchLinearClassifier(
                n_features=n_features,
                n_classes=2,
                lr=self.lr,
                n_epochs=self.n_epochs,
                seed=self.seed + k,
            )
            for k in range(K)
        ]
        model_params: list[dict[str, np.ndarray]] = [{}] * K

        # Results
        accuracy_matrix = np.zeros((K, T), dtype=np.float64)
        predicted_matrix = np.zeros((K, T), dtype=np.int32)

        # Collect posteriors for soft_assignments: list of (t, posteriors_dict)
        posteriors_log: list[tuple[int, dict[int, PosteriorAssignment]]] = []

        phase_a_bytes = 0.0
        phase_b_bytes = 0.0
        total_spawned = 0
        total_merged = 0
        total_pruned = 0
        prev_assignments: dict[int, int] | None = None
        old_assignments: dict[int, int] | None = None

        for t in range(T):
            # Per-step fingerprints: built fresh each step from current
            # data only, avoiding cross-concept contamination.
            step_fingerprints = [
                ConceptFingerprint(n_features, 2) for _ in range(K)
            ]

            # --- Per-client: predict, detect, fingerprint, train ---
            any_drift = False
            for k in range(K):
                X, y = dataset.data[(k, t)]
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]

                # 1. Prequential prediction (on GPU via TorchLinearClassifier)
                y_pred = models[k].predict(X_test)

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
                if is_drift:
                    any_drift = True

                # 3. Build per-step fingerprint (fresh, no accumulation)
                step_fingerprints[k].update(X_train, y_train)

                # 4. Local training on GPU
                if self.n_epochs > 1:
                    models[k].fit(X_train, y_train)
                else:
                    models[k].partial_fit(X_train, y_train)

                if (
                    blend_alpha > 0.0
                    and not is_drift
                    and model_params[k].get("coef") is not None
                ):
                    # Warm-start with momentum only when no drift detected
                    models[k].blend_params(model_params[k], alpha=blend_alpha)

                model_params[k] = models[k].get_params()

            # --- Federation ---
            is_federation_step = (t + 1) % self.federation_every == 0
            if is_federation_step:
                # Decide whether to run Phase A
                # In event-triggered mode: only run Phase A when drift
                # detected or no assignments exist yet (bootstrap)
                run_phase_a = True
                if self.event_triggered and prev_assignments is not None:
                    run_phase_a = any_drift

                if run_phase_a:
                    # Phase A: fingerprint exchange (per-step fingerprints)
                    client_model_losses = {
                        k: 1.0 - accuracy_matrix[k, t] for k in range(K)
                    }
                    client_fps = {k: step_fingerprints[k] for k in range(K)}
                    a_result = protocol.phase_a(
                        client_fps, prev_assignments, client_model_losses,
                    )
                    phase_a_bytes += a_result.total_bytes
                    total_spawned += a_result.spawned
                    total_merged += a_result.merged
                    total_pruned += a_result.pruned
                    old_assignments = prev_assignments
                    prev_assignments = a_result.assignments
                    posteriors_log.append((t, a_result.posteriors))

                # Record concept assignments
                for k in range(K):
                    predicted_matrix[k, t] = (
                        prev_assignments.get(k, 0) if prev_assignments else 0
                    )

                # Concept-model warm-start on concept SWITCHES only:
                # When a client switches to a different concept (recurrence),
                # restore the stored model for that concept. If the client
                # stays on the same concept, keep its locally trained model
                # (which is already specialised for that concept).
                if prev_assignments and old_assignments is not None:
                    for k in range(K):
                        new_cid = prev_assignments.get(k)
                        old_cid = old_assignments.get(k)
                        if (
                            new_cid is not None
                            and old_cid is not None
                            and new_cid != old_cid
                        ):
                            stored = protocol.memory_bank.get_model_params(new_cid)
                            if stored is not None:
                                models[k].set_params(stored)
                                model_params[k] = {
                                    key: arr.copy()
                                    for key, arr in stored.items()
                                }

                # Phase B: model aggregation (always runs on federation steps
                # when assignments exist, even if Phase A was skipped)
                if prev_assignments:
                    client_p = {k: model_params[k] for k in range(K) if model_params[k]}
                    if client_p:
                        b_result = protocol.phase_b(client_p, prev_assignments)
                        phase_b_bytes += b_result.total_bytes

                        # Distribute aggregated models (load back to GPU)
                        for k in range(K):
                            cid = prev_assignments.get(k)
                            if cid is not None and cid in b_result.aggregated_params:
                                agg_p = b_result.aggregated_params[cid]
                                model_params[k] = {
                                    key: arr.copy() for key, arr in agg_p.items()
                                }
                                models[k].set_params(model_params[k])
            else:
                # No federation this step — carry forward previous assignments
                if prev_assignments:
                    for k in range(K):
                        predicted_matrix[k, t] = prev_assignments.get(k, 0)

        total_bytes = phase_a_bytes + phase_b_bytes

        # Build soft_assignments (K, T, C_max) from collected posteriors
        soft_assignments: np.ndarray | None = None
        if posteriors_log:
            # Collect all concept IDs seen across all posteriors
            all_cids: set[int] = set()
            for _, post_dict in posteriors_log:
                for pa in post_dict.values():
                    all_cids.update(pa.probabilities.keys())
            if all_cids:
                cid_list = sorted(all_cids)
                cid_to_idx = {c: i for i, c in enumerate(cid_list)}
                C = len(cid_list)
                soft_assignments = np.zeros((K, T, C), dtype=np.float64)
                for t_step, post_dict in posteriors_log:
                    for k, pa in post_dict.items():
                        for cid, prob in pa.probabilities.items():
                            if cid in cid_to_idx:
                                soft_assignments[k, t_step, cid_to_idx[cid]] = prob

        return FedProTrackResult(
            accuracy_matrix=accuracy_matrix,
            predicted_concept_matrix=predicted_matrix,
            true_concept_matrix=dataset.concept_matrix,
            total_bytes=total_bytes,
            phase_a_bytes=phase_a_bytes,
            phase_b_bytes=phase_b_bytes,
            mean_accuracy=float(accuracy_matrix.mean()),
            final_accuracy=float(accuracy_matrix[:, -1].mean()),
            spawned_concepts=total_spawned,
            merged_concepts=total_merged,
            pruned_concepts=total_pruned,
            active_concepts=protocol.memory_bank.n_concepts,
            soft_assignments=soft_assignments,
        )
