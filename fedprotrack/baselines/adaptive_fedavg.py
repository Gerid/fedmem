from __future__ import annotations

"""Adaptive-FedAvg baseline: FedAvg with drift-adaptive learning rate.

Extends standard FedAvg by monitoring local loss via exponential moving
average.  When loss spikes above a threshold (indicating concept drift),
the local learning rate is boosted.  During stable periods, the lr decays
to allow fine-grained convergence.

Communication budget: same as FedAvg (model params up + global model down).
"""

from dataclasses import dataclass

import numpy as np

from ..drift_generator.generator import DriftDataset
from ..metrics.experiment_log import ExperimentLog
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """Unified result container for Adaptive-FedAvg.

    Parameters
    ----------
    method_name : str
        Human-readable name of the method.
    accuracy_matrix : np.ndarray
        Shape (K, T). Per-client per-step classification accuracy.
    predicted_concept_matrix : np.ndarray
        Shape (K, T). Predicted concept IDs (all zeros for this baseline).
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


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Deep-copy a parameter dict."""
    return {k: v.copy() for k, v in params.items()}


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _extract_dims(dataset: DriftDataset) -> tuple[int, int, int, int]:
    """Return (K, T, n_features, n_classes) from a DriftDataset."""
    K = dataset.config.K
    T = dataset.config.T
    X0, _ = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        labels.update(int(v) for v in np.unique(y))
    n_classes = max(labels) + 1
    return K, T, n_features, n_classes


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveFedAvgUpload:
    """Data uploaded by one Adaptive-FedAvg client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    model_params : dict[str, np.ndarray]
        Locally updated model parameters.
    n_samples : int
        Number of training samples seen so far.
    """

    client_id: int
    model_params: dict[str, np.ndarray]
    n_samples: int


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class AdaptiveFedAvgClient:
    """Client for the Adaptive-FedAvg baseline.

    Monitors local loss via an exponential moving average and adjusts the
    learning rate accordingly: boosted when a drift spike is detected,
    decayed during stable periods.

    Parameters
    ----------
    client_id : int
        Unique identifier.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    lr : float
        Initial (base) learning rate.  Default 0.01.
    n_epochs : int
        Local training epochs per round.  Default 5.
    boost_factor : float
        Multiplicative factor applied to lr on drift detection.  Default 2.0.
    decay_factor : float
        Multiplicative factor applied to lr during stable periods.  Default 0.95.
    ema_alpha : float
        Smoothing coefficient for the loss EMA (higher = more reactive).
        Default 0.3.
    drift_threshold : float
        Ratio ``loss / loss_ema`` above which a drift spike is flagged.
        Default 1.5.
    seed : int
        Random seed.  Default 0.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        lr: float = 0.01,
        n_epochs: int = 5,
        boost_factor: float = 2.0,
        decay_factor: float = 0.95,
        ema_alpha: float = 0.3,
        drift_threshold: float = 1.5,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self._base_lr = lr
        self._current_lr = lr
        self._n_epochs = n_epochs
        self._boost_factor = boost_factor
        self._decay_factor = decay_factor
        self._ema_alpha = ema_alpha
        self._drift_threshold = drift_threshold
        self._seed = seed

        self._loss_ema: float | None = None

        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Internal: rebuild optimizer with updated lr
    # ------------------------------------------------------------------

    def _update_model_lr(self) -> None:
        """Synchronise the underlying model's lr with ``_current_lr``.

        ``TorchLinearClassifier.set_params`` rebuilds the SGD optimizer
        using ``self.lr``, so we update that attribute before calling
        ``set_params`` with the existing weights to get a fresh optimizer
        at the new learning rate.
        """
        self._model.lr = self._current_lr
        params = self._model.get_params()
        self._model.set_params(params)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on a local batch with adaptive learning rate.

        Steps:
        1. Compute loss *before* training to gauge drift.
        2. Update loss EMA.
        3. If loss spike detected, boost lr; otherwise decay lr.
        4. Rebuild model optimizer at the new lr and train.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)
        """
        self._n_samples += len(X)

        # 1. Current loss before training (inf when model is unfitted)
        loss = self._model.predict_loss(X, y)

        # 2. Update EMA (first call initialises; skip inf from unfitted model)
        if not np.isfinite(loss):
            # Model is not yet fitted; skip drift logic this round
            pass
        elif self._loss_ema is None or not np.isfinite(self._loss_ema):
            self._loss_ema = loss
        else:
            self._loss_ema = (
                self._ema_alpha * loss
                + (1.0 - self._ema_alpha) * self._loss_ema
            )

        # 3. Drift check and lr adjustment (skip when EMA is not ready)
        if (
            self._loss_ema is not None
            and np.isfinite(self._loss_ema)
            and self._loss_ema > 0.0
            and np.isfinite(loss)
            and loss / self._loss_ema > self._drift_threshold
        ):
            self._current_lr = min(self._current_lr * self._boost_factor, 1.0)
        elif self._loss_ema is not None and np.isfinite(loss):
            self._current_lr = max(
                self._current_lr * self._decay_factor, 1e-5,
            )

        # 4. Apply updated lr and train
        self._update_model_lr()
        self._model.fit(X, y)
        self._model_params = self._model.get_params()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)

        Returns
        -------
        np.ndarray of shape (n,)
        """
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Federation interface
    # ------------------------------------------------------------------

    def set_global_params(self, params: dict[str, np.ndarray]) -> None:
        """Receive global model from the server.

        Parameters
        ----------
        params : dict[str, np.ndarray]
        """
        self._model.set_params(params)
        self._model_params = self._model.get_params()

    def get_upload(self) -> AdaptiveFedAvgUpload:
        """Package local state for upload.

        Returns
        -------
        AdaptiveFedAvgUpload
        """
        return AdaptiveFedAvgUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Communication cost for one upload.

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        return model_bytes(self._model_params, precision_bits=precision_bits)

    @property
    def current_lr(self) -> float:
        """Current adaptive learning rate (read-only diagnostic)."""
        return self._current_lr


# ---------------------------------------------------------------------------
# Server  (standard FedAvg aggregation -- identical to PFedMeServer)
# ---------------------------------------------------------------------------

class AdaptiveFedAvgServer:
    """Server for Adaptive-FedAvg: weighted FedAvg aggregation.

    The server side is identical to vanilla FedAvg -- all adaptivity
    lives on the client.

    Parameters
    ----------
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    seed : int
        Random seed.  Default 42.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed

        rng = np.random.RandomState(seed)
        n_out = 1 if n_classes == 2 else n_classes
        self.global_params: dict[str, np.ndarray] = {
            "coef": (rng.randn(n_out * n_features) * 0.01).astype(np.float64),
            "intercept": np.zeros(n_out, dtype=np.float64),
        }

    def aggregate(
        self, uploads: list[AdaptiveFedAvgUpload],
    ) -> dict[str, np.ndarray]:
        """Weighted average of client models.

        Parameters
        ----------
        uploads : list[AdaptiveFedAvgUpload]

        Returns
        -------
        dict[str, np.ndarray]
            Updated global model.
        """
        if not uploads:
            return _copy_params(self.global_params)

        valid = [u for u in uploads if u.model_params]
        if not valid:
            return _copy_params(self.global_params)

        weights = [float(max(u.n_samples, 1)) for u in valid]
        total_w = sum(weights)
        if total_w <= 0:
            weights = [1.0] * len(valid)
            total_w = float(len(valid))

        agg: dict[str, np.ndarray] = {}
        for key in valid[0].model_params:
            stacked = np.stack([u.model_params[key] for u in valid])
            w_arr = np.array(weights, dtype=np.float64) / total_w
            agg[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))

        self.global_params = _copy_params(agg)
        return _copy_params(self.global_params)

    def download_bytes(
        self, n_clients: int, precision_bits: int = 32,
    ) -> float:
        """Communication cost for broadcasting global model.

        Parameters
        ----------
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        return float(n_clients) * model_bytes(
            self.global_params, precision_bits=precision_bits,
        )


# ---------------------------------------------------------------------------
# Full runner
# ---------------------------------------------------------------------------

def run_adaptive_fedavg_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    lr: float = 0.01,
    n_epochs: int = 5,
    boost_factor: float = 2.0,
    decay_factor: float = 0.95,
) -> MethodResult:
    """Run Adaptive-FedAvg end-to-end on a DriftDataset.

    Parameters
    ----------
    dataset : DriftDataset
        Synthetic or real drift dataset.
    federation_every : int
        Federation period (aggregate every N rounds).
    lr : float
        Initial (base) learning rate.  Default 0.01.
    n_epochs : int
        Local training epochs per round.  Default 5.
    boost_factor : float
        LR boost multiplier on drift detection.  Default 2.0.
    decay_factor : float
        LR decay multiplier during stable periods.  Default 0.95.

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        AdaptiveFedAvgClient(
            k,
            n_features,
            n_classes,
            lr=lr,
            n_epochs=n_epochs,
            boost_factor=boost_factor,
            decay_factor=decay_factor,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = AdaptiveFedAvgServer(
        n_features=n_features, n_classes=n_classes, seed=42,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_concept_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        # Evaluate before training
        for k in range(K):
            X, y = dataset.data[(k, t)]
            accuracy_matrix[k, t] = _accuracy(y, clients[k].predict(X))

        # Local training
        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        # Federation
        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [client.get_upload() for client in clients]
            upload_bytes = sum(client.upload_bytes() for client in clients)
            global_params = server.aggregate(uploads)
            download_bytes = server.download_bytes(K)
            total_bytes += upload_bytes + download_bytes
            for client in clients:
                client.set_global_params(global_params)

    return MethodResult(
        method_name="Adaptive-FedAvg",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
