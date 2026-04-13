from __future__ import annotations

"""FedDrift baseline: multi-model per-client with loss-based concept selection.

Each client maintains a *set* of models, one per detected concept. Every
round, the client evaluates all stored models on its current local data,
selects the lowest-loss concept, and only spawns a fresh model when no
stored concept achieves sufficiently low loss. The server clusters
clients by their active model similarity and performs FedAvg within each
cluster.

Simplifications vs. the full FedDrift paper:
- Uses a PyTorch linear classifier instead of neural networks.
- Server-side clustering is based on cosine similarity of model parameter
  vectors.
- Client-side concept selection evaluates all stored models on the
  current local data and branches when none achieves sufficiently low
  loss.
"""

from dataclasses import dataclass

import numpy as np

from ..drift_detector import ADWINDetector
from ..federation.aggregator import FedAvgAggregator
from ..models import TorchLinearClassifier
from ..models.factory import create_model
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

    Maintains multiple models (one per detected concept). Each round the
    client evaluates all stored concept models by local loss, reuses the
    best-fitting stored model when possible, and otherwise spawns a fresh
    concept model.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    similarity_threshold : float
        Public interface kept for compatibility. On the client this is
        used as the loss threshold for spawning a new concept model; on
        the server the same value is still used as the cosine-similarity
        threshold for clustering. Default 0.5.
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
        lr: float = 0.01,
        n_epochs: int = 5,
        model_type: str = "linear",
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.similarity_threshold = similarity_threshold
        self._loss_threshold = float(similarity_threshold)
        self._seed = seed
        self._lr = lr
        self._n_epochs = n_epochs
        self._model_type = model_type

        # Per-concept models: concept_id -> params dict
        self._model_params_store: dict[int, dict[str, np.ndarray]] = {}
        self._active_concept: int = 0
        self._next_concept_id: int = 1

        self._drift_detector = ADWINDetector()
        self._n_samples: int = 0
        self._model = create_model(
            model_type, n_features, n_classes,
            lr=lr, n_epochs=n_epochs, seed=seed,
        )
        self._current_params: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _params_to_vector(self, params: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten model params into a single vector."""
        if not params:
            return np.zeros(1)
        return np.concatenate([v.flatten() for v in params.values()])

    def _make_model(self, concept_id: int):
        """Create a model instance for a given concept ID."""
        return create_model(
            self._model_type, self.n_features, self.n_classes,
            lr=self._lr, n_epochs=self._n_epochs,
            seed=self._seed + concept_id,
        )

    def _copy_params(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in params.items()}

    def _evaluate_model(
        self,
        params: dict[str, np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Evaluate a stored model on the current local data."""
        if not params:
            return float("inf")
        temp = create_model(
            self._model_type,
            self.n_features,
            self.n_classes,
            lr=self._lr,
            n_epochs=self._n_epochs,
            seed=self._seed,
        )
        temp.set_params(params)
        try:
            return temp.predict_loss(X, y)
        except (IndexError, RuntimeError):
            # Model output dim may not match current label set
            # (e.g. disjoint label splits across concepts).
            return float("inf")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """FedDrift client training with pool-based model selection.

        Algorithm (Jothimurugesan et al., AISTATS 2023):
        1. Detect drift on current active model.
        2. If drift detected: evaluate ALL pool models on new data,
           pick the best; spawn new only if pool is empty or best
           model has error rate > 50% (random-guess level).
        3. If no drift: continue with current active model.
        4. Fine-tune the selected model on new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        self._n_samples += len(X)

        # --- Step 1: drift detection on current model ---
        drift_detected = False
        if self._model._fitted:
            preds = self._model.predict(X)
            errors = (preds != y).astype(float)
            for err in errors:
                result = self._drift_detector.update(err)
                if result.is_drift:
                    drift_detected = True
                    break

        # --- Step 2: on drift (or first round), select from pool ---
        need_selection = drift_detected or not self._model._fitted

        if need_selection:
            # Save current model before switching
            if self._model._fitted:
                self._model_params_store[self._active_concept] = (
                    self._copy_params(self._model.get_params())
                )

            # Evaluate all stored models on current data
            best_concept: int | None = None
            best_loss = float("inf")
            for cid, params in self._model_params_store.items():
                loss = self._evaluate_model(params, X, y)
                if loss < best_loss:
                    best_loss = loss
                    best_concept = cid

            # Compute random-guess loss as spawn threshold:
            # cross-entropy of uniform prediction = log(n_classes)
            n_unique = max(len(np.unique(y)), 2)
            random_loss = float(np.log(n_unique))

            if best_concept is not None and best_loss < random_loss:
                # Re-use existing concept: load its parameters
                self._active_concept = best_concept
                stored = self._model_params_store[self._active_concept]
                self._model = self._make_model(self._active_concept)
                self._model.set_params(stored)
                self._current_params = self._copy_params(stored)
            else:
                # Spawn new concept
                concept_id = (0 if not self._model_params_store
                              else self._next_concept_id)
                if concept_id == self._next_concept_id:
                    self._next_concept_id += 1
                self._active_concept = concept_id
                self._model = self._make_model(concept_id)

            self._drift_detector.reset()

        # --- Step 3: fine-tune the selected model ---
        self._model.fit(X, y)
        self._current_params = self._copy_params(self._model.get_params())
        self._model_params_store[self._active_concept] = (
            self._copy_params(self._current_params)
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
        return self._model.predict(X)

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
        self._model.set_params(params)
        self._current_params = {k: v.copy() for k, v in params.items()}
        self._model_params_store[self._active_concept] = {
            k: v.copy() for k, v in params.items()
        }

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
