from __future__ import annotations

"""FedProto baseline: nearest-prototype federated learning.

Reference: Tan et al., "FedProto: Federated Prototype Learning across
Heterogeneous Clients" (AAAI 2022). Each client uploads per-class feature
prototypes (means) instead of full model weights. The server aggregates them
via weighted averaging; clients classify by nearest-prototype lookup.
"""

from dataclasses import dataclass, field

import numpy as np

from .comm_tracker import prototype_bytes as _prototype_bytes


@dataclass
class ClientProtoUpload:
    """Data uploaded by one client per federation round.

    Parameters
    ----------
    client_id : int
        Identifier of the uploading client.
    prototypes : dict[int, np.ndarray]
        Mapping from class label to mean feature vector of shape
        ``(n_features,)``.
    class_counts : dict[int, int]
        Number of training samples per class seen by this client.
    n_samples : int
        Total number of training samples seen by this client.
    """

    client_id: int
    prototypes: dict[int, np.ndarray]
    class_counts: dict[int, int]
    n_samples: int


class FedProtoClient:
    """FedProto client: maintains per-class feature prototypes.

    Classifies new samples by finding the global prototype (received from
    the server) with the smallest Euclidean distance to the query point.
    Local prototypes are computed as running per-class means and uploaded
    each round.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client.
    n_features : int
        Dimensionality of the feature space.
    n_classes : int
        Number of possible class labels.
    """

    def __init__(self, client_id: int, n_features: int, n_classes: int) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes

        # Local per-class running statistics
        self._prototypes: dict[int, np.ndarray] = {}
        self._class_counts: dict[int, int] = {}

        # Global prototypes received from server (used for prediction)
        self._global_prototypes: dict[int, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update local prototypes incrementally (compute per-class mean).

        Each call re-computes the running mean for every class present in
        ``y`` by folding the new batch into the existing aggregate.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix for the current batch.
        y : np.ndarray of shape (n_samples,)
            Integer class labels for each sample.
        """
        for cls in np.unique(y):
            cls_int = int(cls)
            mask = y == cls
            X_cls = X[mask]
            new_count = int(mask.sum())

            if cls_int in self._prototypes:
                old_count = self._class_counts[cls_int]
                old_mean = self._prototypes[cls_int]
                total_count = old_count + new_count
                # Weighted combination of old and new means
                updated_mean = (
                    old_mean * old_count + X_cls.sum(axis=0)
                ) / total_count
                self._prototypes[cls_int] = updated_mean
                self._class_counts[cls_int] = total_count
            else:
                self._prototypes[cls_int] = X_cls.mean(axis=0)
                self._class_counts[cls_int] = new_count

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Nearest-prototype classification using global prototypes.

        If no global prototypes have been received from the server yet,
        returns an array of zeros with length ``len(X)``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix to classify.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels (integers).
        """
        if self._global_prototypes is None or len(self._global_prototypes) == 0:
            return np.zeros(len(X), dtype=np.int64)

        classes = list(self._global_prototypes.keys())
        proto_matrix = np.stack(
            [self._global_prototypes[c] for c in classes], axis=0
        )  # (n_classes, n_features)

        # Squared Euclidean distances: (n_samples, n_classes)
        diffs = X[:, np.newaxis, :] - proto_matrix[np.newaxis, :, :]
        sq_dists = (diffs ** 2).sum(axis=2)

        nearest_idx = np.argmin(sq_dists, axis=1)
        return np.array([classes[i] for i in nearest_idx], dtype=np.int64)

    # ------------------------------------------------------------------
    # Federation interface
    # ------------------------------------------------------------------

    def set_global_prototypes(self, global_protos: dict[int, np.ndarray]) -> None:
        """Receive aggregated global prototypes from the server.

        Parameters
        ----------
        global_protos : dict[int, np.ndarray]
            Mapping from class label to global mean prototype vector of
            shape ``(n_features,)``.
        """
        self._global_prototypes = {k: v.copy() for k, v in global_protos.items()}

    def get_upload(self) -> ClientProtoUpload:
        """Package local prototypes for upload to the server.

        Returns
        -------
        ClientProtoUpload
            Snapshot of current local prototypes and class counts.
        """
        n_samples = sum(self._class_counts.values())
        return ClientProtoUpload(
            client_id=self.client_id,
            prototypes={k: v.copy() for k, v in self._prototypes.items()},
            class_counts=dict(self._class_counts),
            n_samples=n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        """Compute bytes required to upload local prototypes to the server.

        Parameters
        ----------
        precision_bits : int
            Bit-width per scalar element (default 32).

        Returns
        -------
        float
            Upload size in bytes.
        """
        return _prototype_bytes(self._prototypes, precision_bits=precision_bits)


class FedProtoAggregator:
    """Server-side FedProto aggregation.

    Computes a weighted average of per-class prototypes across all clients,
    weighting each client's prototype by the number of samples of that class
    it has seen.
    """

    def aggregate(
        self,
        uploads: list[ClientProtoUpload],
    ) -> dict[int, np.ndarray]:
        """Aggregate prototypes via weighted mean by class counts.

        For each class label present in any upload, the global prototype is
        the weighted mean of all client prototypes for that class, where
        the weight is each client's ``class_counts[class]``.

        Parameters
        ----------
        uploads : list[ClientProtoUpload]
            One entry per participating client.

        Returns
        -------
        global_protos : dict[int, np.ndarray]
            Per-class global mean prototype vector of shape ``(n_features,)``.
        """
        if not uploads:
            return {}

        # Collect all class labels across uploads
        all_classes: set[int] = set()
        for upload in uploads:
            all_classes.update(upload.prototypes.keys())

        global_protos: dict[int, np.ndarray] = {}
        for cls in all_classes:
            weighted_sum: np.ndarray | None = None
            total_count = 0

            for upload in uploads:
                if cls not in upload.prototypes:
                    continue
                count = upload.class_counts.get(cls, 0)
                if count == 0:
                    continue
                proto = upload.prototypes[cls]
                if weighted_sum is None:
                    weighted_sum = proto * count
                else:
                    weighted_sum = weighted_sum + proto * count
                total_count += count

            if weighted_sum is not None and total_count > 0:
                global_protos[cls] = weighted_sum / total_count

        return global_protos

    def download_bytes(
        self,
        global_protos: dict[int, np.ndarray],
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        """Compute bytes to broadcast global prototypes to all clients.

        Parameters
        ----------
        global_protos : dict[int, np.ndarray]
            The aggregated global prototypes to be broadcast.
        n_clients : int
            Number of clients that will receive the broadcast.
        precision_bits : int
            Bit-width per scalar element (default 32).

        Returns
        -------
        float
            Total download volume in bytes (one copy per client).

        Raises
        ------
        ValueError
            If ``n_clients`` < 0.
        """
        if n_clients < 0:
            raise ValueError(f"n_clients must be >= 0, got {n_clients}")
        per_client = _prototype_bytes(global_protos, precision_bits=precision_bits)
        return per_client * n_clients
