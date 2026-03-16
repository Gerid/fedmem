from __future__ import annotations

"""Flash baseline: single-model drift adaptation via knowledge distillation.

Flash maintains a single global model across all clients. When drift is
detected on a client, the client's local model is retrained from scratch
on current data while using the old global model as a teacher for
knowledge distillation (soft-label regularisation). This mirrors the core
idea of the Flash framework (Panchal et al., 2023): react to drift by
rapidly adapting a single model rather than maintaining concept-specific
memory.

Simplifications vs. the full Flash paper:
- Uses a PyTorch linear classifier (consistent with other baselines).
- Knowledge distillation is approximated by mixing the old model's
  predicted labels with hard labels (alpha-weighted soft targets).
- Drift detection uses the project's existing ADWIN wrapper.
"""

from dataclasses import dataclass

import numpy as np

from ..drift_detector import ADWINDetector
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


@dataclass
class FlashUpload:
    """Data uploaded by one Flash client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    model_params : dict[str, np.ndarray]
        Mapping ``{"coef": ..., "intercept": ...}`` of model parameters.
    n_samples : int
        Number of training samples seen in the current window.
    has_drifted : bool
        Whether drift was detected since last federation round.
    """

    client_id: int
    model_params: dict[str, np.ndarray]
    n_samples: int
    has_drifted: bool


class FlashClient:
    """Client for the Flash single-model drift adaptation baseline.

    Each client trains a local PyTorch linear model. When drift is detected
    via ADWIN, the model is retrained on the current batch using knowledge
    distillation from the previous global model (soft-label mixing).

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    distill_alpha : float
        Weight of soft labels from old model during distillation.
        0.0 = pure hard labels, 1.0 = pure old-model predictions.
        Default 0.3.
    seed : int
        Random seed. Default 0.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        distill_alpha: float = 0.3,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.distill_alpha = distill_alpha
        self._seed = seed

        self._model = TorchLinearClassifier(
            n_features=n_features, n_classes=n_classes,
            lr=0.01, n_epochs=5, seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._old_model_params: dict[str, np.ndarray] = {}
        self._n_samples: int = 0
        self._drift_detector = ADWINDetector()
        self._has_drifted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit local model on a new batch with optional distillation.

        If drift is detected, the model is retrained using soft targets
        from the old global model mixed with hard labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current batch.
        y : np.ndarray of shape (n_samples,)
            Integer class labels.
        """
        self._n_samples += len(X)

        # Detect drift using prediction error
        if self._model._fitted:
            preds = self._model.predict(X)
            errors = (preds != y).astype(float)
            for err in errors:
                result = self._drift_detector.update(err)
                if result.is_drift:
                    self._has_drifted = True
                    # Store old model params for distillation reference
                    self._old_model_params = {
                        k: v.copy() for k, v in self._model_params.items()
                    }
                    break

        # If drift detected and we have an old model, use distillation
        if self._has_drifted and self._old_model_params:
            y_train = self._distill_labels(X, y)
        else:
            y_train = y

        self._model.fit(X, y_train)
        self._model_params = self._model.get_params()

    def _distill_labels(
        self, X: np.ndarray, y_hard: np.ndarray,
    ) -> np.ndarray:
        """Mix hard labels with old model's predictions for distillation.

        Parameters
        ----------
        X : np.ndarray
        y_hard : np.ndarray

        Returns
        -------
        np.ndarray
            Mixed labels (majority vote between hard and soft).
        """
        if not self._old_model_params:
            return y_hard

        # Build temporary model from old params for soft predictions
        old_model = TorchLinearClassifier(
            n_features=self.n_features, n_classes=self.n_classes,
            seed=self._seed,
        )
        old_model.set_params(self._old_model_params)
        y_soft = old_model.predict(X)

        # Stochastic mixing: with probability (1-alpha) use hard label,
        # with probability alpha use old model prediction
        rng = np.random.RandomState(self._seed + self._n_samples)
        mask = rng.random(len(y_hard)) < self.distill_alpha
        y_mixed = y_hard.copy()
        y_mixed[mask] = y_soft[mask]
        return y_mixed

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using the local model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Federation interface
    # ------------------------------------------------------------------

    def set_model_params(self, params: dict[str, np.ndarray]) -> None:
        """Receive aggregated model parameters from the server.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Must contain ``"coef"`` and ``"intercept"``.
        """
        self._model.set_params(params)
        self._model_params = {k: v.copy() for k, v in params.items()}
        # Reset drift state after global update
        self._has_drifted = False
        self._drift_detector.reset()

    def get_upload(self) -> FlashUpload:
        """Package local model parameters for upload.

        Returns
        -------
        FlashUpload
        """
        return FlashUpload(
            client_id=self.client_id,
            model_params={k: v.copy() for k, v in self._model_params.items()},
            n_samples=self._n_samples,
            has_drifted=self._has_drifted,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute bytes required to upload model to the server.

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        return model_bytes(self._model_params, precision_bits)


class FlashAggregator:
    """Server-side aggregator for Flash.

    Performs weighted FedAvg across all clients. Clients that have
    drifted receive higher weight to steer the global model towards
    the new distribution faster.

    Parameters
    ----------
    drift_weight_boost : float
        Multiplicative weight boost for clients that reported drift.
        Default 2.0.
    """

    def __init__(self, drift_weight_boost: float = 2.0) -> None:
        self.drift_weight_boost = drift_weight_boost

    def aggregate(
        self, uploads: list[FlashUpload],
    ) -> dict[str, np.ndarray]:
        """Aggregate client models into a single global model.

        Parameters
        ----------
        uploads : list[FlashUpload]

        Returns
        -------
        dict[str, np.ndarray]
            Aggregated ``{"coef": ..., "intercept": ...}``.
        """
        if not uploads:
            return {}

        valid = [u for u in uploads if u.model_params]
        if not valid:
            return {}

        weights = []
        for u in valid:
            w = float(max(u.n_samples, 1))
            if u.has_drifted:
                w *= self.drift_weight_boost
            weights.append(w)

        total_w = sum(weights)
        if total_w == 0:
            weights = [1.0] * len(valid)
            total_w = float(len(valid))

        result: dict[str, np.ndarray] = {}
        for key in valid[0].model_params:
            stacked = np.stack([u.model_params[key] for u in valid])
            w_arr = np.array(weights, dtype=np.float64) / total_w
            result[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))

        return result

    def download_bytes(
        self,
        global_params: dict[str, np.ndarray],
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        """Compute bytes required to broadcast global model to all clients.

        Parameters
        ----------
        global_params : dict[str, np.ndarray]
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        return n_clients * model_bytes(global_params, precision_bits)
