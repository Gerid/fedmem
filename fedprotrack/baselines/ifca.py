from __future__ import annotations

"""IFCA baseline: Iterative Federated Clustering Algorithm.

IFCA (Ghosh et al., 2020) maintains a fixed number of cluster models on
the server. Each client evaluates all cluster models on its local data
and selects the one with the lowest loss. The server then aggregates
client updates within each cluster via FedAvg.

This provides a multi-model alternative to FedProTrack that does *not*
explicitly track concept identity over time — instead it relies on
loss-based cluster selection each round.

Simplifications vs. the full IFCA paper:
- Uses a PyTorch linear classifier instead of neural networks.
- Loss is cross-entropy evaluated on the local batch.
- No random restarts or multi-init; deterministic initialisation.
"""

from dataclasses import dataclass

import numpy as np

from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


@dataclass
class IFCAUpload:
    """Data uploaded by one IFCA client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    selected_cluster : int
        Index of the cluster model selected by this client.
    model_params : dict[str, np.ndarray]
        Locally updated parameters of the selected cluster model.
    n_samples : int
        Number of training samples in the current batch.
    """

    client_id: int
    selected_cluster: int
    model_params: dict[str, np.ndarray]
    n_samples: int


class IFCAClient:
    """Client for the IFCA baseline.

    Evaluates all cluster models, selects the best, trains locally on
    that model, and uploads the updated parameters.

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
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed

        # Cluster models received from server
        self._cluster_models: list[dict[str, np.ndarray]] = []
        self._selected_cluster: int = 0
        self._model = TorchLinearClassifier(
            n_features=n_features, n_classes=n_classes,
            lr=0.01, n_epochs=5, seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Cluster selection
    # ------------------------------------------------------------------

    def _evaluate_model(
        self, params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray,
    ) -> float:
        """Evaluate log-loss of a model on given data.

        Parameters
        ----------
        params : dict[str, np.ndarray]
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        float
            Mean log-loss (lower is better).
        """
        if not params:
            return float("inf")

        # Use a temporary model to evaluate
        temp = TorchLinearClassifier(
            n_features=self.n_features, n_classes=self.n_classes,
            seed=self._seed,
        )
        temp.set_params(params)
        return temp.predict_loss(X, y)

    def select_cluster(self, X: np.ndarray, y: np.ndarray) -> int:
        """Select the best cluster model based on loss.

        Parameters
        ----------
        X : np.ndarray
        y : np.ndarray

        Returns
        -------
        int
            Index of the selected cluster.
        """
        if not self._cluster_models:
            return 0

        losses = [
            self._evaluate_model(params, X, y)
            for params in self._cluster_models
        ]
        self._selected_cluster = int(np.argmin(losses))
        return self._selected_cluster

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Select best cluster, then train locally.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        self._n_samples += len(X)

        # Select cluster based on current data
        self.select_cluster(X, y)

        # Initialise model from selected cluster params
        if self._cluster_models and self._selected_cluster < len(self._cluster_models):
            selected_params = self._cluster_models[self._selected_cluster]
            if selected_params:
                old_params = {k: v.copy() for k, v in selected_params.items()}
                self._model.set_params(selected_params)

        # Train locally on GPU
        self._model.fit(X, y)

        # Blend with cluster model for stability
        if self._cluster_models and self._selected_cluster < len(self._cluster_models):
            selected_params = self._cluster_models[self._selected_cluster]
            if selected_params:
                self._model.blend_params(selected_params, alpha=0.5)

        self._model_params = self._model.get_params()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the current model.

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

    def set_cluster_models(self, cluster_models: list[dict[str, np.ndarray]]) -> None:
        """Receive all cluster models from the server.

        Parameters
        ----------
        cluster_models : list[dict[str, np.ndarray]]
            One parameter dict per cluster.
        """
        self._cluster_models = [
            {k: v.copy() for k, v in cm.items()} if cm else {}
            for cm in cluster_models
        ]
        # Also update active model if cluster params available
        if (
            self._cluster_models
            and self._selected_cluster < len(self._cluster_models)
            and self._cluster_models[self._selected_cluster]
        ):
            self._model.set_params(
                self._cluster_models[self._selected_cluster],
            )

    def get_upload(self) -> IFCAUpload:
        """Package locally trained model for upload.

        Returns
        -------
        IFCAUpload
        """
        return IFCAUpload(
            client_id=self.client_id,
            selected_cluster=self._selected_cluster,
            model_params={k: v.copy() for k, v in self._model_params.items()},
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute bytes required to upload the model.

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        return model_bytes(self._model_params, precision_bits)


class IFCAServer:
    """Server for the IFCA baseline.

    Maintains ``n_clusters`` model parameter sets. Each round, aggregates
    client updates within each cluster via weighted average.

    Parameters
    ----------
    n_clusters : int
        Number of cluster models. Default 3.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of classes.
    seed : int
        Random seed for initialisation. Default 42.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_features: int = 2,
        n_classes: int = 2,
        seed: int = 42,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {n_clusters}")
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_classes = n_classes

        # Initialise cluster models with small random weights
        rng = np.random.RandomState(seed)
        expected_coef_elems = (1 if n_classes == 2 else n_classes) * n_features
        expected_intercept_elems = 1 if n_classes == 2 else n_classes

        self.cluster_models: list[dict[str, np.ndarray]] = []
        for _ in range(n_clusters):
            self.cluster_models.append({
                "coef": rng.randn(expected_coef_elems).astype(np.float64) * 0.01,
                "intercept": np.zeros(expected_intercept_elems, dtype=np.float64),
            })

    def aggregate(
        self, uploads: list[IFCAUpload],
    ) -> list[dict[str, np.ndarray]]:
        """Aggregate client updates within each cluster.

        Parameters
        ----------
        uploads : list[IFCAUpload]

        Returns
        -------
        list[dict[str, np.ndarray]]
            Updated cluster models (one per cluster).
        """
        if not uploads:
            return self.cluster_models

        # Group by selected cluster
        cluster_uploads: dict[int, list[IFCAUpload]] = {}
        for u in uploads:
            cid = u.selected_cluster
            if cid not in cluster_uploads:
                cluster_uploads[cid] = []
            cluster_uploads[cid].append(u)

        # Update each cluster with its members
        for cid, members in cluster_uploads.items():
            if cid >= self.n_clusters:
                continue

            valid = [m for m in members if m.model_params]
            if not valid:
                continue

            weights = [float(max(m.n_samples, 1)) for m in valid]
            total_w = sum(weights)
            if total_w == 0:
                weights = [1.0] * len(valid)
                total_w = float(len(valid))

            new_params: dict[str, np.ndarray] = {}
            for key in valid[0].model_params:
                stacked = np.stack([m.model_params[key] for m in valid])
                w_arr = np.array(weights, dtype=np.float64) / total_w
                new_params[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))

            self.cluster_models[cid] = new_params

        return self.cluster_models

    def download_bytes(
        self,
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        """Compute bytes to broadcast all cluster models to all clients.

        Each client receives all cluster models (needed for selection).

        Parameters
        ----------
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        per_model = sum(
            model_bytes(cm, precision_bits) for cm in self.cluster_models
        )
        return n_clients * per_model
