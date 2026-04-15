from __future__ import annotations

"""CompressedFedAvg baseline: budget-matched compressed model exchange.

Standard FedAvg with top-k sparsification and error feedback. This
serves as a matched-budget control: instead of spending communication
budget on concept tracking overhead, it spends the *same* budget on
transmitting a compressed full model more frequently.

The compression scheme:
- **Top-k sparsification**: only the top-k% largest-magnitude entries of
  the model update (delta from global model) are transmitted.
- **Error feedback**: residual (un-transmitted) entries accumulate in a
  local error buffer and are added to the next round's delta.

This is the canonical compressed communication baseline from
Stich et al. (2018) / Alistarh et al. (2018).
"""

from dataclasses import dataclass

import numpy as np

from ..federation.aggregator import FedAvgAggregator
from ..models import TorchLinearClassifier
from ..models.factory import create_model
from .comm_tracker import model_bytes


@dataclass
class CompressedUpload:
    """Data uploaded by one CompressedFedAvg client per round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    sparse_delta : dict[str, np.ndarray]
        Sparsified model delta (zero entries are not transmitted).
    n_nonzero : int
        Total number of non-zero entries across all parameter arrays.
    n_total : int
        Total number of entries (for byte accounting).
    n_samples : int
        Number of training samples.
    """

    client_id: int
    sparse_delta: dict[str, np.ndarray]
    n_nonzero: int
    n_total: int
    n_samples: int


class CompressedFedAvgClient:
    """Client for the CompressedFedAvg baseline.

    Trains a local PyTorch linear model, computes delta from global model,
    applies top-k sparsification with error feedback, uploads the sparse
    delta.

    Parameters
    ----------
    client_id : int
        Unique identifier.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of classes.
    topk_fraction : float
        Fraction of entries to transmit (0.0 < topk_fraction <= 1.0).
        Default 0.3 (30% sparsification).
    seed : int
        Random seed. Default 0.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        topk_fraction: float = 0.3,
        seed: int = 0,
        lr: float = 0.01,
        n_epochs: int = 5,
        model_type: str = "linear",
    ) -> None:
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError(
                f"topk_fraction must be in (0, 1], got {topk_fraction}",
            )
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.topk_fraction = topk_fraction
        self._seed = seed

        self._model = create_model(
            model_type, n_features, n_classes,
            lr=lr, n_epochs=n_epochs, seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        # Global model params (received from server)
        self._global_params: dict[str, np.ndarray] = {}
        # Error feedback buffers
        self._error_buffer: dict[str, np.ndarray] = {}
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit local model on a new batch.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        self._n_samples += len(X)
        self._model.fit(X, y)
        self._model_params = self._model.get_params()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray

        Returns
        -------
        np.ndarray
        """
        return self._model.predict(X)

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def _topk_sparsify(
        self, delta: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], int, int]:
        """Apply top-k sparsification with error feedback.

        Parameters
        ----------
        delta : dict[str, np.ndarray]
            Raw model delta (local - global).

        Returns
        -------
        sparse_delta : dict[str, np.ndarray]
            Sparsified delta (non-selected entries set to 0).
        n_nonzero : int
            Number of non-zero entries transmitted.
        n_total : int
            Total number of entries.
        """
        # Add error feedback from previous round
        corrected: dict[str, np.ndarray] = {}
        for key, arr in delta.items():
            err = self._error_buffer.get(key, np.zeros_like(arr))
            corrected[key] = arr + err

        # Flatten all corrected deltas to find global top-k
        all_vals = np.concatenate([v.flatten() for v in corrected.values()])
        n_total = len(all_vals)
        k = max(1, int(np.ceil(n_total * self.topk_fraction)))

        # Find the k-th largest magnitude
        abs_vals = np.abs(all_vals)
        if k >= n_total:
            threshold = 0.0
        else:
            partition_idx = n_total - k
            threshold = float(np.partition(abs_vals, partition_idx)[partition_idx])

        # Build sparse delta and update error buffer
        sparse_delta: dict[str, np.ndarray] = {}
        n_nonzero = 0

        for key, arr in corrected.items():
            mask = np.abs(arr) >= threshold
            sparse = np.where(mask, arr, 0.0)
            sparse_delta[key] = sparse
            n_nonzero += int(np.count_nonzero(sparse))
            self._error_buffer[key] = arr - sparse

        return sparse_delta, n_nonzero, n_total

    # ------------------------------------------------------------------
    # Federation interface
    # ------------------------------------------------------------------

    def set_model_params(self, params: dict[str, np.ndarray]) -> None:
        """Receive aggregated model parameters from the server.

        Parameters
        ----------
        params : dict[str, np.ndarray]
        """
        self._model.set_params(params)
        self._model_params = {k: v.copy() for k, v in params.items()}
        self._global_params = {k: v.copy() for k, v in params.items()}

    def get_upload(self) -> CompressedUpload:
        """Compute sparse delta and package for upload.

        Returns
        -------
        CompressedUpload
        """
        if not self._model_params:
            return CompressedUpload(
                client_id=self.client_id,
                sparse_delta={},
                n_nonzero=0,
                n_total=0,
                n_samples=self._n_samples,
            )

        # Compute delta from global model
        delta: dict[str, np.ndarray] = {}
        for key in self._model_params:
            local = self._model_params[key]
            glob = self._global_params.get(key, np.zeros_like(local))
            delta[key] = local - glob

        sparse_delta, n_nonzero, n_total = self._topk_sparsify(delta)

        return CompressedUpload(
            client_id=self.client_id,
            sparse_delta=sparse_delta,
            n_nonzero=n_nonzero,
            n_total=n_total,
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Estimate bytes for the compressed upload without mutating state.

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        if not self._model_params:
            return 0.0
        n_total = sum(arr.size for arr in self._model_params.values())
        n_nonzero = max(1, int(np.ceil(n_total * self.topk_fraction)))
        value_bytes = n_nonzero * precision_bits / 8
        index_bytes = n_nonzero * 4
        return float(value_bytes + index_bytes)

    def upload_bytes_from_upload(
        self, upload: CompressedUpload, precision_bits: int = 32,
    ) -> float:
        """Compute bytes from a pre-computed upload.

        Parameters
        ----------
        upload : CompressedUpload
        precision_bits : int

        Returns
        -------
        float
        """
        if upload.n_nonzero == 0:
            return 0.0
        value_bytes = upload.n_nonzero * precision_bits / 8
        index_bytes = upload.n_nonzero * 4
        return float(value_bytes + index_bytes)


class CompressedFedAvgServer:
    """Server for the CompressedFedAvg baseline.

    Receives sparse deltas from clients, reconstructs full deltas,
    averages them, and applies to the global model.

    Parameters
    ----------
    n_features : int
    n_classes : int
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        expected_coef_elems = (1 if n_classes == 2 else n_classes) * n_features
        expected_intercept_elems = 1 if n_classes == 2 else n_classes
        self.global_params: dict[str, np.ndarray] = {
            "coef": np.zeros(expected_coef_elems, dtype=np.float64),
            "intercept": np.zeros(expected_intercept_elems, dtype=np.float64),
        }

    def aggregate(
        self, uploads: list[CompressedUpload],
    ) -> dict[str, np.ndarray]:
        """Aggregate sparse deltas and update global model.

        Parameters
        ----------
        uploads : list[CompressedUpload]

        Returns
        -------
        dict[str, np.ndarray]
            Updated global model parameters.
        """
        if not uploads:
            return self.global_params

        valid = [u for u in uploads if u.sparse_delta]
        if not valid:
            return self.global_params

        weights = [float(max(u.n_samples, 1)) for u in valid]
        total_w = sum(weights)
        if total_w == 0:
            weights = [1.0] * len(valid)
            total_w = float(len(valid))

        avg_delta: dict[str, np.ndarray] = {}
        for key in valid[0].sparse_delta:
            stacked = np.stack([u.sparse_delta[key] for u in valid])
            w_arr = np.array(weights, dtype=np.float64) / total_w
            avg_delta[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))

        for key in avg_delta:
            if key in self.global_params:
                self.global_params[key] = self.global_params[key] + avg_delta[key]
            else:
                self.global_params[key] = avg_delta[key]

        return self.global_params

    def download_bytes(
        self,
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        """Compute bytes to broadcast global model.

        Parameters
        ----------
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        return n_clients * model_bytes(self.global_params, precision_bits)
