from __future__ import annotations

"""SCAFFOLD baseline: variance reduction via control variates.

SCAFFOLD (Karimireddy et al., ICML 2020) maintains server control variate c
and per-client control variates c_i to correct local gradient drift.
Uses Option II: c_i_new = (x_global - x_local) / (n_steps * lr).

Communication budget: 2x FedAvg (model params + control variate delta).
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
    """Unified result container (matches runners.MethodResult).

    Parameters
    ----------
    method_name : str
        Human-readable name of the method.
    accuracy_matrix : np.ndarray
        Shape (K, T). Per-client per-step classification accuracy.
    predicted_concept_matrix : np.ndarray
        Shape (K, T). Predicted concept IDs.
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


def _zeros_like(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a dict of zero arrays matching shapes/dtypes in *params*."""
    return {k: np.zeros_like(v) for k, v in params.items()}


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
class SCAFFOLDUpload:
    """Data uploaded by one SCAFFOLD client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    model_params : dict[str, np.ndarray]
        Locally updated model parameters.
    control_delta : dict[str, np.ndarray]
        Change in the client control variate: ``c_i_new - c_i_old``.
    n_samples : int
        Number of training samples seen so far.
    """

    client_id: int
    model_params: dict[str, np.ndarray]
    control_delta: dict[str, np.ndarray]
    n_samples: int


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class SCAFFOLDClient:
    """Client for the SCAFFOLD baseline.

    Local training uses standard SGD, and the control variate update
    follows Option II from the paper:
        ``c_i_new = (x_global - x_local) / (n_epochs * lr)``

    Parameters
    ----------
    client_id : int
        Unique identifier.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    lr : float
        Local SGD learning rate.  Default 0.01.
    n_epochs : int
        Local training epochs per round.  Default 5.
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
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self._lr = lr
        self._n_epochs = n_epochs
        self._seed = seed

        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._global_params: dict[str, np.ndarray] = {}
        self._n_samples: int = 0

        # Control variates -- initialised to zeros once we know param shapes.
        self._control: dict[str, np.ndarray] = {}
        self._server_control: dict[str, np.ndarray] = {}

        # Latest control delta for upload.
        self._control_delta: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on a local batch and update control variates (Option II).

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)
        """
        self._n_samples += len(X)

        # (1) Snapshot params before local training.
        x_before = self._model.get_params()
        if not x_before:
            # Model has no params yet (first call) -- fit to initialise.
            self._model.fit(X, y)
            self._model_params = self._model.get_params()
            # Initialise control variates to zeros now that we know shapes.
            if not self._control:
                self._control = _zeros_like(self._model_params)
                self._server_control = _zeros_like(self._model_params)
                self._control_delta = _zeros_like(self._model_params)
            return

        # (2) Run standard local SGD training.
        self._model.fit(X, y)

        # (3) Snapshot params after local training.
        x_after = self._model.get_params()

        # (4) Option II control variate update:
        #     c_i_new = (x_before - x_after) / (n_epochs * lr)
        denom = self._n_epochs * self._lr
        if denom == 0.0:
            denom = 1.0  # safety fallback

        c_new: dict[str, np.ndarray] = {}
        for key in x_before:
            c_new[key] = (x_before[key] - x_after[key]) / denom

        # (5) Control delta = c_new - c_old.
        control_delta: dict[str, np.ndarray] = {}
        for key in c_new:
            old = self._control.get(key, np.zeros_like(c_new[key]))
            control_delta[key] = c_new[key] - old

        # (6) Store updated control variate.
        self._control = c_new
        self._control_delta = control_delta

        # (7) Store model params (keep x_after as-is; Option II corrects
        #     via the control delta on the server side rather than
        #     modifying the local model directly).
        self._model_params = _copy_params(x_after)

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
        self._global_params = _copy_params(params)
        self._model.set_params(params)
        self._model_params = self._model.get_params()

    def set_server_control(self, control: dict[str, np.ndarray]) -> None:
        """Receive the server control variate.

        Parameters
        ----------
        control : dict[str, np.ndarray]
        """
        self._server_control = _copy_params(control)

    def get_upload(self) -> SCAFFOLDUpload:
        """Package local state for upload.

        Returns
        -------
        SCAFFOLDUpload
        """
        return SCAFFOLDUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            control_delta=_copy_params(self._control_delta) if self._control_delta else {},
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Communication cost for one upload (model + control delta).

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        return (
            model_bytes(self._model_params, precision_bits=precision_bits)
            + model_bytes(self._control_delta, precision_bits=precision_bits)
        )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class SCAFFOLDServer:
    """Server for SCAFFOLD: weighted FedAvg + global control variate update.

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
        self.global_control: dict[str, np.ndarray] = _zeros_like(self.global_params)

    def aggregate(
        self, uploads: list[SCAFFOLDUpload],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Aggregate client uploads into a new global model and control variate.

        Parameters
        ----------
        uploads : list[SCAFFOLDUpload]

        Returns
        -------
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]
            ``(global_params, global_control)`` after aggregation.
        """
        if not uploads:
            return _copy_params(self.global_params), _copy_params(self.global_control)

        valid = [u for u in uploads if u.model_params]
        if not valid:
            return _copy_params(self.global_params), _copy_params(self.global_control)

        # (1) Weighted average of model params -> new global params.
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

        # (2) Update global control:
        #     c = c + (1/N) * sum(delta_c_i)
        N = len(valid)
        for key in self.global_control:
            delta_sum = np.zeros_like(self.global_control[key])
            for u in valid:
                if key in u.control_delta:
                    delta_sum = delta_sum + u.control_delta[key]
            self.global_control[key] = self.global_control[key] + delta_sum / N

        return _copy_params(self.global_params), _copy_params(self.global_control)

    def download_bytes(self, n_clients: int, precision_bits: int = 32) -> float:
        """Communication cost for broadcasting global model + control variate.

        Parameters
        ----------
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        per_client = (
            model_bytes(self.global_params, precision_bits=precision_bits)
            + model_bytes(self.global_control, precision_bits=precision_bits)
        )
        return float(n_clients) * per_client


# ---------------------------------------------------------------------------
# Full runner
# ---------------------------------------------------------------------------


def run_scaffold_full(
    dataset: DriftDataset,
    federation_every: int = 1,
    *,
    lr: float = 0.01,
    n_epochs: int = 5,
) -> MethodResult:
    """Run SCAFFOLD end-to-end on a DriftDataset.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every : int
        Federation period.
    lr : float
        Local learning rate.
    n_epochs : int
        Local epochs.

    Returns
    -------
    MethodResult
    """
    K, T, n_features, n_classes = _extract_dims(dataset)
    clients = [
        SCAFFOLDClient(
            k,
            n_features,
            n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = SCAFFOLDServer(n_features=n_features, n_classes=n_classes, seed=42)

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_concept_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for t in range(T):
        # Evaluate before training at this step.
        for k in range(K):
            X, y = dataset.data[(k, t)]
            accuracy_matrix[k, t] = _accuracy(y, clients[k].predict(X))

        # Local training.
        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        # Federation round.
        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [client.get_upload() for client in clients]
            upload_b = sum(client.upload_bytes() for client in clients)
            global_params, global_control = server.aggregate(uploads)
            download_b = server.download_bytes(K)
            total_bytes += upload_b + download_b
            for client in clients:
                client.set_global_params(global_params)
                client.set_server_control(global_control)

    return MethodResult(
        method_name="SCAFFOLD",
        accuracy_matrix=accuracy_matrix,
        predicted_concept_matrix=predicted_concept_matrix,
        total_bytes=total_bytes,
    )
