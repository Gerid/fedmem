from __future__ import annotations

"""FedCCFA-Impl: Classifier Clustering and Feature Alignment for FL.

Re-implementation of the core FedCCFA algorithm (NeurIPS 2024):
  1. **Classifier clustering**: After local training, the server collects
     classifier weight matrices from all clients.  Classifier *rows*
     (one per class) are clustered using DBSCAN to discover concept groups.
  2. **Feature alignment**: Cluster centroid features are computed and an
     alignment loss nudges local features toward the cluster centroid
     during local training, weighted by a configurable gamma.
  3. **Drift handling**: Concept drift is detected when the cluster
     structure (number of clusters or client-to-cluster mapping) changes
     across consecutive federation rounds.

The existing ``fedccfa.py`` already provides a lighter label-prototype
variant.  This file adds the full DBSCAN-row-clustering +
feature-alignment-loss formulation, keeping the same federation
interface conventions (Client / Server / Upload / runner).

Communication budget:
  - Upload: model params + classifier weight rows.
  - Download: aggregated model + cluster assignment int per client.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN

from ..device import get_device, to_numpy, to_tensor
from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Deep-copy a parameter dict."""
    return {k: v.copy() for k, v in params.items()}


def _classifier_rows(
    params: dict[str, np.ndarray],
    n_features: int,
    n_classes: int,
) -> np.ndarray:
    """Extract per-class classifier weight rows as (n_classes, n_features).

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Model parameter dict (``coef``, ``intercept``).
    n_features : int
    n_classes : int

    Returns
    -------
    np.ndarray
        Shape ``(n_classes, n_features)``.
    """
    coef = np.asarray(params["coef"], dtype=np.float64).reshape(-1)
    if n_classes == 2:
        if coef.size == n_features:
            row = coef.reshape(1, n_features)
            return np.vstack([-row, row])
        if coef.size == 2 * n_features:
            return coef.reshape(2, n_features)
    return coef.reshape(n_classes, n_features)


def _cluster_rows_dbscan(
    all_rows: np.ndarray,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """Cluster a set of row vectors using DBSCAN.

    Parameters
    ----------
    all_rows : np.ndarray
        Shape ``(N, D)`` -- one row per (client, class) pair.
    eps : float
        DBSCAN neighbourhood radius.
    min_samples : int
        Minimum samples for a core point.

    Returns
    -------
    np.ndarray
        Integer cluster labels of shape ``(N,)``.
    """
    if all_rows.shape[0] == 0:
        return np.array([], dtype=np.int64)
    if all_rows.shape[0] == 1:
        return np.zeros(1, dtype=np.int64)

    # Normalise rows before computing distances
    norms = np.linalg.norm(all_rows, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = all_rows / norms
    # Cosine distance matrix
    sim = normed @ normed.T
    dist = 1.0 - np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    dist = np.maximum(dist, 0.0)

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(dist)

    # Relabel noise points (-1) as singleton clusters
    max_label = int(labels.max()) if labels.size > 0 else -1
    for i in range(len(labels)):
        if labels[i] == -1:
            max_label += 1
            labels[i] = max_label
    return labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FedCCFAImplUpload:
    """Data uploaded by one FedCCFA-Impl client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    model_params : dict[str, np.ndarray]
        Locally updated model parameters.
    classifier_rows : np.ndarray
        Per-class classifier weight rows, shape ``(n_classes, n_features)``.
    feature_centroids : dict[int, np.ndarray]
        Per-class mean feature vector from the local batch.
    n_samples : int
        Number of training samples in the current round.
    """

    client_id: int
    model_params: dict[str, np.ndarray]
    classifier_rows: np.ndarray
    feature_centroids: dict[int, np.ndarray]
    n_samples: int


@dataclass
class FedCCFAImplUpdate:
    """Server -> client personalised update after clustering.

    Parameters
    ----------
    aggregated_params : dict[str, np.ndarray]
        Aggregated model parameters from same-cluster clients.
    cluster_id : int
        Assigned concept cluster for this client.
    cluster_centroids : dict[int, np.ndarray]
        Per-class feature centroid of the assigned cluster.
    """

    aggregated_params: dict[str, np.ndarray] = field(default_factory=dict)
    cluster_id: int = 0
    cluster_centroids: dict[int, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FedCCFAImplClient:
    """Client for the FedCCFA-Impl baseline.

    Parameters
    ----------
    client_id : int
        Unique identifier.
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    gamma : float
        Feature-alignment loss weight.  Default 20.0.
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
        gamma: float = 20.0,
        lr: float = 0.01,
        n_epochs: int = 5,
        seed: int = 0,
    ) -> None:
        if gamma < 0.0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.gamma = gamma
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
        self._cluster_centroids: dict[int, np.ndarray] = {}
        self._feature_centroids: dict[int, np.ndarray] = {}
        self._cluster_id: int = 0
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Local feature centroids
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_feature_centroids(
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Compute per-class mean feature vectors.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)

        Returns
        -------
        dict[int, np.ndarray]
        """
        centroids: dict[int, np.ndarray] = {}
        for label in np.unique(y.astype(np.int64)):
            mask = y == label
            centroids[int(label)] = X[mask].mean(axis=0).astype(np.float64)
        return centroids

    # ------------------------------------------------------------------
    # Training with optional feature-alignment loss
    # ------------------------------------------------------------------

    def _fit_with_alignment(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the local model with an optional alignment regulariser.

        When ``self._cluster_centroids`` is non-empty, an L2 alignment
        penalty pulls per-class feature means toward the cluster centroids.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)
        """
        device = self._model.device
        X_t = to_tensor(X, device=device)
        y_t = to_tensor(y, dtype=torch.long, device=device)

        linear = self._model._linear
        n_out = self._model._n_out
        optimizer = self._model._optimizer

        # Pre-compute cluster centroid tensors for alignment
        centroid_tensors: dict[int, torch.Tensor] = {}
        if self.gamma > 0.0 and self._cluster_centroids:
            for label, centroid in self._cluster_centroids.items():
                centroid_tensors[label] = to_tensor(
                    centroid.reshape(1, -1),
                    device=device,
                )

        linear.train()
        for _ in range(self._n_epochs):
            optimizer.zero_grad()
            logits = linear(X_t)

            # Classification loss
            if n_out == 1:
                cls_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.squeeze(-1), y_t.float(),
                )
            else:
                cls_loss = nn.functional.cross_entropy(logits, y_t)

            # Feature-alignment loss
            align_loss = torch.tensor(0.0, device=device)
            if centroid_tensors:
                for label, centroid_t in centroid_tensors.items():
                    mask = (y_t == label)
                    if mask.sum() == 0:
                        continue
                    local_mean = X_t[mask].mean(dim=0, keepdim=True)
                    align_loss = align_loss + nn.functional.mse_loss(
                        local_mean, centroid_t,
                    )

            loss = cls_loss + self.gamma * align_loss
            loss.backward()
            optimizer.step()

        self._model._fitted = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on a local batch with optional alignment regulariser.

        Parameters
        ----------
        X : np.ndarray of shape (n, d)
        y : np.ndarray of shape (n,)
        """
        self._n_samples += len(X)
        self._feature_centroids = self._compute_feature_centroids(X, y)

        if self.gamma > 0.0 and self._cluster_centroids:
            self._fit_with_alignment(X, y)
        else:
            self._model.fit(X, y)

        self._model_params = self._model.get_params()

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

    def set_update(self, update: FedCCFAImplUpdate) -> None:
        """Apply server-side personalised update.

        Parameters
        ----------
        update : FedCCFAImplUpdate
        """
        if update.aggregated_params:
            self._model.set_params(update.aggregated_params)
            self._model_params = self._model.get_params()
        self._cluster_id = update.cluster_id
        self._cluster_centroids = {
            k: v.copy() for k, v in update.cluster_centroids.items()
        }

    def get_upload(self) -> FedCCFAImplUpload:
        """Package local state for upload.

        Returns
        -------
        FedCCFAImplUpload
        """
        rows = _classifier_rows(
            self._model_params, self.n_features, self.n_classes,
        ) if self._model_params else np.zeros(
            (self.n_classes, self.n_features), dtype=np.float64,
        )
        return FedCCFAImplUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            classifier_rows=rows.copy(),
            feature_centroids={k: v.copy() for k, v in self._feature_centroids.items()},
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
        # model params + classifier rows (already included in coef, but
        # counted explicitly for conceptual clarity)
        return model_bytes(self._model_params, precision_bits)

    @property
    def cluster_id(self) -> int:
        """Current cluster assignment."""
        return self._cluster_id


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class FedCCFAImplServer:
    """Server for FedCCFA-Impl: DBSCAN row clustering + feature alignment.

    Parameters
    ----------
    n_features : int
        Feature dimensionality.
    n_classes : int
        Number of class labels.
    n_clusters_max : int
        Upper bound on cluster count (informational; DBSCAN is
        non-parametric).  Default 10.
    dbscan_eps : float
        DBSCAN neighbourhood radius.  Default 0.1.
    dbscan_min_samples : int
        DBSCAN core-point threshold.  Default 1.
    seed : int
        Random seed.  Default 42.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        n_clusters_max: int = 10,
        dbscan_eps: float = 0.1,
        dbscan_min_samples: int = 1,
        seed: int = 42,
    ) -> None:
        if dbscan_eps <= 0.0:
            raise ValueError(f"dbscan_eps must be > 0, got {dbscan_eps}")
        if dbscan_min_samples < 1:
            raise ValueError(
                f"dbscan_min_samples must be >= 1, got {dbscan_min_samples}",
            )

        self.n_features = n_features
        self.n_classes = n_classes
        self.n_clusters_max = n_clusters_max
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self._seed = seed

        # Tracking cluster structure changes for drift detection
        self._prev_cluster_map: dict[int, int] = {}
        self._drift_detected: bool = False

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_clients(
        self,
        uploads: list[FedCCFAImplUpload],
    ) -> dict[int, int]:
        """Assign each client to a cluster by DBSCAN on classifier rows.

        For each client, all per-class classifier rows are concatenated
        into a single vector before clustering.

        Parameters
        ----------
        uploads : list[FedCCFAImplUpload]

        Returns
        -------
        dict[int, int]
            Mapping from client_id to cluster label.
        """
        if not uploads:
            return {}

        # Build per-client feature vector: concat all classifier rows
        vectors: list[np.ndarray] = []
        for upload in uploads:
            rows = upload.classifier_rows  # (n_classes, n_features)
            vectors.append(rows.reshape(-1))

        all_vectors = np.stack(vectors, axis=0)  # (n_clients, n_classes * n_features)
        labels = _cluster_rows_dbscan(
            all_vectors, self.dbscan_eps, self.dbscan_min_samples,
        )

        return {
            upload.client_id: int(labels[i])
            for i, upload in enumerate(uploads)
        }

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        uploads: list[FedCCFAImplUpload],
    ) -> dict[int, FedCCFAImplUpdate]:
        """Cluster clients, aggregate within clusters, compute centroids.

        Parameters
        ----------
        uploads : list[FedCCFAImplUpload]

        Returns
        -------
        dict[int, FedCCFAImplUpdate]
            Per-client personalised updates.
        """
        if not uploads:
            return {}

        cluster_map = self._cluster_clients(uploads)

        # Detect drift: cluster structure change
        self._drift_detected = cluster_map != self._prev_cluster_map
        self._prev_cluster_map = dict(cluster_map)

        # Group uploads by cluster
        cluster_groups: dict[int, list[FedCCFAImplUpload]] = {}
        for upload in uploads:
            cid = cluster_map[upload.client_id]
            cluster_groups.setdefault(cid, []).append(upload)

        # Per-cluster aggregation
        cluster_agg_params: dict[int, dict[str, np.ndarray]] = {}
        cluster_centroids: dict[int, dict[int, np.ndarray]] = {}

        for cid, members in cluster_groups.items():
            # Weighted average of model params
            valid = [m for m in members if m.model_params]
            if not valid:
                continue

            weights = [float(max(m.n_samples, 1)) for m in valid]
            total_w = sum(weights)
            if total_w <= 0:
                weights = [1.0] * len(valid)
                total_w = float(len(valid))

            agg: dict[str, np.ndarray] = {}
            for key in valid[0].model_params:
                stacked = np.stack([m.model_params[key] for m in valid])
                w_arr = np.array(weights, dtype=np.float64) / total_w
                agg[key] = np.tensordot(w_arr, stacked, axes=([0], [0]))
            cluster_agg_params[cid] = agg

            # Per-class feature centroid within cluster
            centroids: dict[int, np.ndarray] = {}
            for label in range(self.n_classes):
                centroid_sum = np.zeros(self.n_features, dtype=np.float64)
                count = 0.0
                for m in valid:
                    fc = m.feature_centroids.get(label)
                    if fc is not None:
                        w = float(max(m.n_samples, 1))
                        centroid_sum += w * np.asarray(fc, dtype=np.float64)
                        count += w
                if count > 0:
                    centroids[label] = centroid_sum / count
            cluster_centroids[cid] = centroids

        # Build per-client updates
        updates: dict[int, FedCCFAImplUpdate] = {}
        for upload in uploads:
            cid = cluster_map[upload.client_id]
            updates[upload.client_id] = FedCCFAImplUpdate(
                aggregated_params=_copy_params(cluster_agg_params.get(cid, {})),
                cluster_id=cid,
                cluster_centroids={
                    k: v.copy()
                    for k, v in cluster_centroids.get(cid, {}).items()
                },
            )

        return updates

    @property
    def drift_detected(self) -> bool:
        """Whether the last aggregation detected a cluster-structure change."""
        return self._drift_detected

    def download_bytes(
        self,
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        """Communication cost for broadcasting updates to all clients.

        Each client receives: aggregated model params + cluster assignment
        (int) + per-class feature centroids.

        Parameters
        ----------
        n_clients : int
        precision_bits : int

        Returns
        -------
        float
        """
        if precision_bits <= 0:
            raise ValueError(f"precision_bits must be > 0, got {precision_bits}")

        # Model params: n_classes * n_features (coef) + n_classes (intercept)
        n_out = 1 if self.n_classes == 2 else self.n_classes
        model_elems = n_out * self.n_features + n_out
        # Centroids: n_classes * n_features
        centroid_elems = self.n_classes * self.n_features
        # Cluster assignment: 1 int (4 bytes)
        cluster_assign_bytes = 4.0

        per_client = float(
            (model_elems + centroid_elems) * precision_bits / 8
        ) + cluster_assign_bytes
        return float(n_clients) * per_client
