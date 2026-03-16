from __future__ import annotations

"""FedDrift baseline: multi-model per-client with drift-triggered branching.

Each client maintains a *set* of models, one per detected concept. When
drift is detected the client spawns a new model. The server clusters
clients by their active model similarity and performs FedAvg within each
cluster. This mirrors the FedDrift framework (Canonical et al., 2021):
each concept gets its own model trajectory, and the server groups
clients that share the same current concept.

Simplifications vs. the full FedDrift paper:
- Uses LogisticRegression instead of neural networks.
- Clustering is based on cosine similarity of model parameter vectors.
- Concept re-identification uses a simple nearest-model heuristic.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..drift_detector import ADWINDetector
from ..federation.aggregator import FedAvgAggregator
from .comm_tracker import model_bytes


@dataclass
class FedDriftUpload:
    """Data uploaded by one FedDrift client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    active_concept_id : int
        ID of the currently active concept on this client.
    model_params : dict[str, np.ndarray]
        Parameters of the currently active model.
    model_vector : np.ndarray
        Flattened parameter vector for clustering.
    n_samples : int
        Number of training samples in the current concept window.
    """

    client_id: int
    active_concept_id: int
    model_params: dict[str, np.ndarray]
    model_vector: np.ndarray
    n_samples: int


class FedDriftClient:
    """Client for the FedDrift multi-model baseline.

    Maintains multiple models (one per detected concept). Drift detection
    triggers spawning of a new model or re-identification of a previously
    seen concept via parameter similarity.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    similarity_threshold : float
        Cosine similarity threshold for concept re-identification.
        If a new model's parameters are sufficiently similar to a stored
        concept's model, re-use that concept. Default 0.5.
    seed : int
        Random seed. Default 0.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        similarity_threshold: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.similarity_threshold = similarity_threshold
        self._seed = seed

        # Per-concept models: concept_id -> (model, params_dict)
        self._models: dict[int, tuple[LogisticRegression, dict[str, np.ndarray]]] = {}
        self._active_concept: int = 0
        self._next_concept_id: int = 1

        self._drift_detector = ADWINDetector()
        self._n_samples: int = 0
        self._current_model: LogisticRegression | None = None
        self._current_params: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _params_to_vector(self, params: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten model params into a single vector."""
        if not params:
            return np.zeros(1)
        return np.concatenate([v.flatten() for v in params.values()])

    def _fit_model(
        self, X: np.ndarray, y: np.ndarray,
    ) -> tuple[LogisticRegression, dict[str, np.ndarray]]:
        """Fit a fresh LogisticRegression and return (model, params)."""
        classes = np.arange(self.n_classes)
        if len(np.unique(y)) < self.n_classes:
            missing = [c for c in classes if c not in np.unique(y)]
            X_aug = np.vstack([X] + [X[[0]] * 0.0 for _ in missing])
            y_aug = np.concatenate([y, np.array(missing, dtype=y.dtype)])
        else:
            X_aug, y_aug = X, y

        model = LogisticRegression(
            max_iter=200,
            random_state=self._seed + self._active_concept,
            solver="lbfgs",
        )
        model.fit(X_aug, y_aug)
        params = {
            "coef": model.coef_.flatten().copy(),
            "intercept": model.intercept_.flatten().copy(),
        }
        return model, params

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the active model on a new batch, detecting drift.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        self._n_samples += len(X)
        drift_detected = False

        # Detect drift
        if self._current_model is not None:
            preds = self._current_model.predict(X)
            errors = (preds != y).astype(float)
            for err in errors:
                result = self._drift_detector.update(err)
                if result.is_drift:
                    drift_detected = True
                    break

        if drift_detected:
            # Save current model under its concept
            if self._current_model is not None:
                self._models[self._active_concept] = (
                    self._current_model, self._current_params,
                )

            # Train a fresh model on new data
            new_model, new_params = self._fit_model(X, y)
            new_vec = self._params_to_vector(new_params)

            # Try to re-identify an existing concept
            best_concept = -1
            best_sim = -1.0
            for cid, (_, cparams) in self._models.items():
                cvec = self._params_to_vector(cparams)
                sim = self._cosine_sim(new_vec, cvec)
                if sim > best_sim:
                    best_sim = sim
                    best_concept = cid

            if best_concept >= 0 and best_sim >= self.similarity_threshold:
                # Re-use existing concept
                self._active_concept = best_concept
                # Re-train existing concept model with new data
                self._current_model, self._current_params = self._fit_model(X, y)
            else:
                # Spawn new concept
                self._active_concept = self._next_concept_id
                self._next_concept_id += 1
                self._current_model = new_model
                self._current_params = new_params

            self._drift_detector.reset()
        else:
            # No drift — train current model
            self._current_model, self._current_params = self._fit_model(X, y)

        # Update stored model
        self._models[self._active_concept] = (
            self._current_model, self._current_params,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the currently active model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if self._current_model is None:
            return np.zeros(len(X), dtype=np.int64)
        return self._current_model.predict(X).astype(np.int64)

    @property
    def active_concept_id(self) -> int:
        """Return the currently active concept ID."""
        return self._active_concept

    # ------------------------------------------------------------------
    # Federation interface
    # ------------------------------------------------------------------

    def set_model_params(self, params: dict[str, np.ndarray]) -> None:
        """Receive aggregated model parameters for the active concept.

        Parameters
        ----------
        params : dict[str, np.ndarray]
        """
        coef = params["coef"]
        intercept = params["intercept"]
        expected_rows = 1 if self.n_classes == 2 else self.n_classes
        coef_2d = coef.reshape(expected_rows, self.n_features)

        if self._current_model is None:
            model = LogisticRegression(
                max_iter=200, random_state=self._seed, solver="lbfgs",
            )
            model.classes_ = np.arange(self.n_classes)
            model.coef_ = coef_2d.copy()
            model.intercept_ = intercept.copy()
            self._current_model = model
        else:
            self._current_model.coef_ = coef_2d.copy()
            self._current_model.intercept_ = intercept.copy()

        self._current_params = {"coef": coef.copy(), "intercept": intercept.copy()}
        self._models[self._active_concept] = (
            self._current_model, self._current_params,
        )

    def get_upload(self) -> FedDriftUpload:
        """Package active model parameters for upload.

        Returns
        -------
        FedDriftUpload
        """
        return FedDriftUpload(
            client_id=self.client_id,
            active_concept_id=self._active_concept,
            model_params={k: v.copy() for k, v in self._current_params.items()},
            model_vector=self._params_to_vector(self._current_params),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute bytes required to upload the active model.

        Parameters
        ----------
        precision_bits : int

        Returns
        -------
        float
        """
        return model_bytes(self._current_params, precision_bits)


class FedDriftServer:
    """Server for the FedDrift baseline.

    Clusters clients by their active model's parameter vector similarity,
    then performs FedAvg within each cluster.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity to join an existing cluster. Default 0.5.
    """

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.similarity_threshold = similarity_threshold
        self._aggregator = FedAvgAggregator()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def aggregate(
        self, uploads: list[FedDriftUpload],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Cluster clients and aggregate within clusters.

        Parameters
        ----------
        uploads : list[FedDriftUpload]

        Returns
        -------
        dict[int, dict[str, np.ndarray]]
            Maps each ``client_id`` to aggregated model parameters.
        """
        if not uploads:
            return {}

        # Greedy clustering by model vector similarity
        clusters: list[dict] = []

        for upload in uploads:
            mv = upload.model_vector
            best_idx = -1
            best_sim = -2.0
            for i, cluster in enumerate(clusters):
                sim = self._cosine_sim(mv, cluster["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_idx >= 0 and best_sim >= self.similarity_threshold:
                clusters[best_idx]["members"].append(upload)
                vecs = np.stack([m.model_vector for m in clusters[best_idx]["members"]])
                clusters[best_idx]["centroid"] = vecs.mean(axis=0)
            else:
                clusters.append({"members": [upload], "centroid": mv.copy()})

        # FedAvg within each cluster
        client_params: dict[int, dict[str, np.ndarray]] = {}

        for cluster in clusters:
            members = cluster["members"]
            valid = [m for m in members if m.model_params]
            if not valid:
                for m in members:
                    client_params[m.client_id] = {}
                continue

            params_list = [m.model_params for m in valid]
            weights = [float(max(m.n_samples, 1)) for m in valid]
            total_w = sum(weights)
            if total_w == 0:
                weights = [1.0] * len(valid)

            aggregated = self._aggregator.aggregate(params_list, weights)
            for m in members:
                client_params[m.client_id] = aggregated

        return client_params

    def download_bytes(
        self,
        uploads: list[FedDriftUpload],
        precision_bits: int = 32,
    ) -> float:
        """Compute bytes to send aggregated models back.

        Parameters
        ----------
        uploads : list[FedDriftUpload]
        precision_bits : int

        Returns
        -------
        float
        """
        total = 0.0
        for upload in uploads:
            total += model_bytes(upload.model_params, precision_bits)
        return total
