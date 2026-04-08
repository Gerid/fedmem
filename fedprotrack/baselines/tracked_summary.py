from __future__ import annotations

"""TrackedSummary baseline: fingerprint-routed federated learning.

Each client shares a ConceptFingerprint alongside its model weights.
The server clusters clients by fingerprint cosine similarity and performs
FedAvg within each cluster, avoiding aggregation across heterogeneous
distributions.
"""

from dataclasses import dataclass, field

import numpy as np

from ..concept_tracker.fingerprint import ConceptFingerprint
from ..federation.aggregator import FedAvgAggregator
from ..models import TorchLinearClassifier
from ..models.factory import create_model
from .comm_tracker import model_bytes, fingerprint_bytes


@dataclass
class TrackedUpload:
    """Data uploaded by one TrackedSummary client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    fingerprint_vector : np.ndarray
        Flattened fingerprint summary from ``ConceptFingerprint.to_vector()``,
        shape ``(n_features + n_classes,)``.
    model_params : dict[str, np.ndarray]
        Mapping ``{"coef": ..., "intercept": ...}`` of model parameters.
    n_samples : int
        Total number of training samples seen by this client.
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of class labels.
    """

    client_id: int
    fingerprint_vector: np.ndarray
    model_params: dict[str, np.ndarray]
    n_samples: int
    n_features: int
    n_classes: int


class TrackedSummaryClient:
    """Client for the TrackedSummary method.

    Maintains a ConceptFingerprint and a PyTorch linear model.
    At each step the client fits local data and updates its fingerprint.
    During federation it uploads the fingerprint vector together with model
    parameters so the server can cluster clients before aggregation.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    seed : int
        Random seed. Default 0.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        seed: int = 0,
        lr: float = 0.01,
        n_epochs: int = 5,
        model_type: str = "linear",
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed

        self._fingerprint = ConceptFingerprint(n_features=n_features, n_classes=n_classes)
        self._model = create_model(
            model_type, n_features, n_classes,
            lr=lr, n_epochs=n_epochs, seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit local model on a new batch and update the fingerprint.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current batch.
        y : np.ndarray of shape (n_samples,)
            Integer class labels.
        """
        # Update incremental fingerprint
        self._fingerprint.update(X, y)
        self._n_samples += len(X)

        # Train on GPU
        self._model.fit(X, y)
        self._model_params = self._model.get_params()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using the local model.

        Returns an array of zeros if the model has not been fitted yet.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to classify.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
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

    def get_upload(self) -> TrackedUpload:
        """Package local fingerprint and model parameters for upload.

        Returns
        -------
        TrackedUpload
            Snapshot of the current fingerprint vector and model parameters.
        """
        return TrackedUpload(
            client_id=self.client_id,
            fingerprint_vector=self._fingerprint.to_vector().copy(),
            model_params={k: v.copy() for k, v in self._model_params.items()},
            n_samples=self._n_samples,
            n_features=self.n_features,
            n_classes=self.n_classes,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute bytes required to upload fingerprint and model to the server.

        Parameters
        ----------
        precision_bits : int
            Bit-width per scalar element (default 32).

        Returns
        -------
        float
            Sum of fingerprint bytes and model bytes.
        """
        fp_b = fingerprint_bytes(self.n_features, self.n_classes, precision_bits)
        mdl_b = model_bytes(self._model_params, precision_bits) if self._model_params else 0.0
        return fp_b + mdl_b


class TrackedSummaryServer:
    """Server for the TrackedSummary method.

    Receives fingerprint vectors from clients, clusters them by cosine
    similarity using a greedy single-pass algorithm, then performs FedAvg
    within each cluster independently.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity between a client's fingerprint vector and
        a cluster centroid for the client to join that cluster. Default 0.5.
    """

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.similarity_threshold = similarity_threshold
        self._aggregator = FedAvgAggregator()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def aggregate(
        self,
        uploads: list[TrackedUpload],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Cluster clients by fingerprint similarity, aggregate within clusters.

        Parameters
        ----------
        uploads : list[TrackedUpload]

        Returns
        -------
        dict[int, dict[str, np.ndarray]]
            Maps each ``client_id`` to the aggregated model parameters.
        """
        if not uploads:
            return {}

        clusters: list[dict] = []

        for upload in uploads:
            fv = upload.fingerprint_vector
            best_idx = -1
            best_sim = -2.0
            for i, cluster in enumerate(clusters):
                sim = self._cosine_sim(fv, cluster["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_idx >= 0 and best_sim >= self.similarity_threshold:
                clusters[best_idx]["members"].append(upload)
                member_vecs = np.stack(
                    [m.fingerprint_vector for m in clusters[best_idx]["members"]]
                )
                clusters[best_idx]["centroid"] = member_vecs.mean(axis=0)
            else:
                clusters.append({"members": [upload], "centroid": fv.copy()})

        client_params: dict[int, dict[str, np.ndarray]] = {}

        for cluster in clusters:
            members = cluster["members"]
            valid = [m for m in members if m.model_params]
            if not valid:
                for m in members:
                    client_params[m.client_id] = {}
                continue

            params_list = [m.model_params for m in valid]
            weights = [float(m.n_samples) for m in valid]
            if sum(weights) == 0:
                weights = [1.0] * len(valid)

            aggregated = self._aggregator.aggregate(params_list, weights)
            for m in members:
                client_params[m.client_id] = aggregated

        return client_params

    def download_bytes(
        self,
        uploads: list[TrackedUpload],
        precision_bits: int = 32,
    ) -> float:
        """Compute bytes to send aggregated models back to all clients.

        Parameters
        ----------
        uploads : list[TrackedUpload]
        precision_bits : int

        Returns
        -------
        float
        """
        total = 0.0
        for upload in uploads:
            total += model_bytes(upload.model_params, precision_bits)
        return total
