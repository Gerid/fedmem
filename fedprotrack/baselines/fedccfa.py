from __future__ import annotations

"""FedCCFA baseline: classifier clustering with prototype-aware adaptation.

This module ports the core idea of FedCCFA (NeurIPS 2024) into the
project's lightweight linear-model setting:

1. Cluster classifier parameters *per label* across clients.
2. Aggregate label-specific prototypes/anchors within each cluster.
3. Re-distribute personalised label parameters and cluster-specific
   prototypes back to the participating clients.

The original paper uses a deep encoder with decoupled representation and
classifier training. This baseline keeps the classifier-clustering +
prototype-alignment logic, but adapts it to the existing
``TorchLinearClassifier`` interface used throughout the repository.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN

from ..models import TorchLinearClassifier
from ..models.factory import create_model
from .comm_tracker import model_bytes, prototype_bytes


def _zero_model_params(n_features: int, n_classes: int) -> dict[str, np.ndarray]:
    n_out = 1 if n_classes == 2 else n_classes
    return {
        "coef": np.zeros(n_out * n_features, dtype=np.float64),
        "intercept": np.zeros(n_out, dtype=np.float64),
    }


def _split_label_params(
    params: dict[str, np.ndarray],
    n_features: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert project parameter dicts into per-label weight/bias arrays."""
    if not params:
        W = np.zeros((n_classes, n_features), dtype=np.float64)
        b = np.zeros(n_classes, dtype=np.float64)
        return W, b

    coef = np.asarray(params["coef"], dtype=np.float64).reshape(-1)
    intercept = np.asarray(params["intercept"], dtype=np.float64).reshape(-1)

    if n_classes == 2:
        if coef.size == n_features and intercept.size == 1:
            row = coef.reshape(1, n_features)
            W = np.vstack([-row, row])
            b = np.array([-intercept[0], intercept[0]], dtype=np.float64)
            return W, b
        if coef.size == 2 * n_features and intercept.size == 2:
            return coef.reshape(2, n_features), intercept.reshape(2)
        raise ValueError(
            "Binary classifier params must have shapes "
            f"({n_features},)/(1,) or ({2 * n_features},)/(2,), "
            f"got coef={coef.shape}, intercept={intercept.shape}",
        )

    expected_coef = n_classes * n_features
    if coef.size != expected_coef or intercept.size != n_classes:
        raise ValueError(
            f"Expected coef/intercept sizes {expected_coef}/{n_classes}, "
            f"got {coef.size}/{intercept.size}",
        )
    return coef.reshape(n_classes, n_features), intercept.reshape(n_classes)


def _merge_label_params(
    W: np.ndarray,
    b: np.ndarray,
    n_features: int,
    n_classes: int,
) -> dict[str, np.ndarray]:
    """Convert per-label weight/bias arrays back into project params."""
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if n_classes == 2:
        coef = 0.5 * (W[1] - W[0])
        intercept = np.array([0.5 * (b[1] - b[0])], dtype=np.float64)
        return {
            "coef": coef.reshape(n_features).copy(),
            "intercept": intercept,
        }

    return {
        "coef": W.reshape(n_classes * n_features).copy(),
        "intercept": b.reshape(n_classes).copy(),
    }


def _label_update_bytes(
    label_vectors: dict[int, np.ndarray],
    label_biases: dict[int, float],
    precision_bits: int = 32,
) -> float:
    if precision_bits <= 0:
        raise ValueError(f"precision_bits must be > 0, got {precision_bits}")
    total_elements = sum(vec.size for vec in label_vectors.values()) + len(label_biases)
    return float(total_elements * precision_bits / 8)


@dataclass
class FedCCFAUpload:
    """Data uploaded by one client for classifier clustering."""

    client_id: int
    model_params: dict[str, np.ndarray]
    local_prototypes: dict[int, np.ndarray]
    label_counts: dict[int, int]
    n_samples: int


@dataclass
class FedCCFAUpdate:
    """Personalised label-wise update sent from server to one client."""

    label_vectors: dict[int, np.ndarray] = field(default_factory=dict)
    label_biases: dict[int, float] = field(default_factory=dict)
    global_prototypes: dict[int, np.ndarray] = field(default_factory=dict)
    label_cluster_ids: dict[int, int] = field(default_factory=dict)


class FedCCFAClient:
    """Client for the FedCCFA baseline.

    The client trains a local linear classifier, extracts per-label
    prototypes from the current batch, and lightly nudges label weights
    toward the server-provided prototypes. The server then clusters
    label-specific classifier heads and redistributes personalised label
    parameters and prototypes.
    """

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        prototype_mix: float = 0.20,
        lr: float = 0.01,
        n_epochs: int = 5,
        seed: int = 0,
        model_type: str = "linear",
    ) -> None:
        if prototype_mix < 0.0 or prototype_mix > 1.0:
            raise ValueError(
                f"prototype_mix must be in [0, 1], got {prototype_mix}",
            )

        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.prototype_mix = prototype_mix
        self._seed = seed
        self._model_type = model_type

        self._model = create_model(
            model_type,
            n_features,
            n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._global_prototypes: dict[int, np.ndarray] = {}
        self._local_prototypes: dict[int, np.ndarray] = {}
        self._label_counts: dict[int, int] = {}
        self._label_cluster_ids: dict[int, int] = {
            label: 0 for label in range(n_classes)
        }
        self._n_samples: int = 0

    def _prototype_alignment_mix(self, y: np.ndarray) -> float:
        if self.prototype_mix <= 0.0 or len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(np.int64), minlength=self.n_classes)
        probs = counts[counts > 0].astype(np.float64)
        if probs.size <= 1:
            return 0.0
        probs /= probs.sum()
        entropy = float(-(probs * np.log(probs)).sum())
        max_entropy = float(np.log(max(self.n_classes, 2)))
        if max_entropy <= 1e-12:
            return 0.0
        return float(np.clip(self.prototype_mix * entropy / max_entropy, 0.0, self.prototype_mix))

    @staticmethod
    def _compute_label_counts(y: np.ndarray) -> dict[int, int]:
        labels, counts = np.unique(y.astype(np.int64), return_counts=True)
        return {int(label): int(count) for label, count in zip(labels, counts)}

    @staticmethod
    def _compute_local_prototypes(
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[int, np.ndarray]:
        prototypes: dict[int, np.ndarray] = {}
        for label in np.unique(y.astype(np.int64)):
            mask = y == label
            prototypes[int(label)] = X[mask].mean(axis=0).astype(np.float64)
        return prototypes

    def _align_classifier_to_prototypes(
        self,
        params: dict[str, np.ndarray],
        observed_labels: set[int],
        mix: float,
    ) -> dict[str, np.ndarray]:
        if mix <= 0.0 or not self._global_prototypes:
            return params

        W, b = _split_label_params(params, self.n_features, self.n_classes)
        for label in observed_labels:
            proto = self._global_prototypes.get(label)
            if proto is None:
                continue
            proto_vec = np.asarray(proto, dtype=np.float64).reshape(-1)
            if proto_vec.size != self.n_features:
                continue
            proto_norm = np.linalg.norm(proto_vec)
            if proto_norm < 1e-12:
                continue
            row = W[label]
            row_scale = max(np.linalg.norm(row), 1.0)
            aligned = proto_vec / proto_norm * row_scale
            W[label] = (1.0 - mix) * row + mix * aligned
        return _merge_label_params(W, b, self.n_features, self.n_classes)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_samples += len(X)
        self._label_counts = self._compute_label_counts(y)
        self._local_prototypes = self._compute_local_prototypes(X, y)

        self._model.fit(X, y)
        params = self._model.get_params()

        mix = self._prototype_alignment_mix(y)
        params = self._align_classifier_to_prototypes(
            params,
            observed_labels=set(self._label_counts),
            mix=mix,
        )
        self._model.set_params(params)
        self._model_params = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def set_personalized_state(self, update: FedCCFAUpdate) -> None:
        base_params = self._model_params or _zero_model_params(
            self.n_features,
            self.n_classes,
        )
        W, b = _split_label_params(base_params, self.n_features, self.n_classes)

        for label, vec in update.label_vectors.items():
            W[label] = np.asarray(vec, dtype=np.float64).copy()
        for label, bias in update.label_biases.items():
            b[label] = float(bias)

        new_params = _merge_label_params(W, b, self.n_features, self.n_classes)
        self._model.set_params(new_params)
        self._model_params = self._model.get_params()

        for label, proto in update.global_prototypes.items():
            self._global_prototypes[label] = np.asarray(proto, dtype=np.float64).copy()
        for label, cluster_id in update.label_cluster_ids.items():
            self._label_cluster_ids[label] = int(cluster_id)

    def get_upload(self) -> FedCCFAUpload:
        return FedCCFAUpload(
            client_id=self.client_id,
            model_params={k: v.copy() for k, v in self._model_params.items()},
            local_prototypes={k: v.copy() for k, v in self._local_prototypes.items()},
            label_counts=dict(self._label_counts),
            n_samples=self._n_samples,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        return model_bytes(self._model_params, precision_bits) + prototype_bytes(
            self._local_prototypes,
            precision_bits,
        )

    @property
    def cluster_signature(self) -> tuple[int, ...]:
        return tuple(
            int(self._label_cluster_ids.get(label, 0))
            for label in range(self.n_classes)
        )


class FedCCFAServer:
    """Server-side classifier clustering and prototype aggregation."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        eps: float = 0.35,
        reid_similarity_threshold: float = 0.85,
        memory_momentum: float = 0.5,
    ) -> None:
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if not 0.0 <= reid_similarity_threshold <= 1.0:
            raise ValueError(
                "reid_similarity_threshold must be in [0, 1], got "
                f"{reid_similarity_threshold}",
            )
        if not 0.0 <= memory_momentum <= 1.0:
            raise ValueError(
                f"memory_momentum must be in [0, 1], got {memory_momentum}",
            )

        self.n_features = n_features
        self.n_classes = n_classes
        self.eps = eps
        self.reid_similarity_threshold = reid_similarity_threshold
        self.memory_momentum = memory_momentum

        self._label_cluster_memory: dict[int, dict[int, np.ndarray]] = {
            label: {} for label in range(n_classes)
        }
        self._next_label_cluster_id: dict[int, int] = {
            label: 0 for label in range(n_classes)
        }

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _madd(self, vecs: np.ndarray) -> np.ndarray:
        num = len(vecs)
        res = np.zeros((num, num), dtype=np.float64)
        if num <= 2:
            return res

        for i in range(num):
            for j in range(i + 1, num):
                dist = 0.0
                for z in range(num):
                    if z == i or z == j:
                        continue
                    dist += abs(
                        self._cosine_sim(vecs[i], vecs[z])
                        - self._cosine_sim(vecs[j], vecs[z])
                    )
                res[i, j] = res[j, i] = dist / (num - 2)
        return res

    def _distance_matrix(self, vecs: np.ndarray) -> np.ndarray:
        num = len(vecs)
        if num <= 1:
            return np.zeros((num, num), dtype=np.float64)

        if num == 2:
            sim = self._cosine_sim(vecs[0], vecs[1])
            dist = np.zeros((2, 2), dtype=np.float64)
            dist[0, 1] = dist[1, 0] = 1.0 - sim
            return dist

        dist = self._madd(vecs)
        dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
        np.fill_diagonal(dist, 0.0)
        return dist

    def _cluster_indices(self, vecs: list[np.ndarray]) -> list[list[int]]:
        if len(vecs) <= 1:
            return [list(range(len(vecs)))]

        dist = self._distance_matrix(np.asarray(vecs, dtype=np.float64))
        labels = DBSCAN(
            eps=self.eps,
            min_samples=1,
            metric="precomputed",
        ).fit_predict(dist)

        groups: list[list[int]] = []
        for cluster_id in sorted(set(labels)):
            indices = np.where(labels == cluster_id)[0]
            groups.append(indices.tolist())
        return groups

    def _aggregate_label_params(
        self,
        label: int,
        uploads: list[FedCCFAUpload],
    ) -> tuple[np.ndarray, float]:
        total_weight = 0.0
        row_sum = np.zeros(self.n_features, dtype=np.float64)
        bias_sum = 0.0

        for upload in uploads:
            W, b = _split_label_params(
                upload.model_params,
                self.n_features,
                self.n_classes,
            )
            weight = float(max(upload.label_counts.get(label, 0), 1))
            row_sum += weight * W[label]
            bias_sum += weight * float(b[label])
            total_weight += weight

        if total_weight <= 0.0:
            return row_sum, bias_sum
        return row_sum / total_weight, bias_sum / total_weight

    def _aggregate_label_prototype(
        self,
        label: int,
        uploads: list[FedCCFAUpload],
    ) -> np.ndarray | None:
        proto_sum = np.zeros(self.n_features, dtype=np.float64)
        total_weight = 0.0

        for upload in uploads:
            proto = upload.local_prototypes.get(label)
            if proto is None:
                continue
            weight = float(upload.label_counts.get(label, 0))
            if weight <= 0.0:
                continue
            proto_sum += weight * np.asarray(proto, dtype=np.float64)
            total_weight += weight

        if total_weight <= 0.0:
            return None
        return proto_sum / total_weight

    def _cluster_representation(
        self,
        label_row: np.ndarray,
        label_bias: float,
        label_proto: np.ndarray | None,
    ) -> np.ndarray:
        if label_proto is None:
            label_proto = np.zeros(self.n_features, dtype=np.float64)
        return np.concatenate([
            np.asarray(label_row, dtype=np.float64).reshape(-1),
            np.array([label_bias], dtype=np.float64),
            np.asarray(label_proto, dtype=np.float64).reshape(-1),
        ])

    def _assign_persistent_cluster_id(
        self,
        label: int,
        representation: np.ndarray,
    ) -> int:
        memory = self._label_cluster_memory[label]
        if not memory:
            cluster_id = self._next_label_cluster_id[label]
            self._next_label_cluster_id[label] += 1
            memory[cluster_id] = representation.copy()
            return cluster_id

        best_id = -1
        best_sim = -1.0
        for cluster_id, ref in memory.items():
            sim = self._cosine_sim(representation, ref)
            if sim > best_sim:
                best_sim = sim
                best_id = cluster_id

        if best_id >= 0 and best_sim >= self.reid_similarity_threshold:
            memory[best_id] = (
                self.memory_momentum * memory[best_id]
                + (1.0 - self.memory_momentum) * representation
            )
            return best_id

        cluster_id = self._next_label_cluster_id[label]
        self._next_label_cluster_id[label] += 1
        memory[cluster_id] = representation.copy()
        return cluster_id

    def aggregate(
        self,
        uploads: list[FedCCFAUpload],
    ) -> dict[int, FedCCFAUpdate]:
        updates = {
            upload.client_id: FedCCFAUpdate()
            for upload in uploads
        }
        if not uploads:
            return updates

        for label in range(self.n_classes):
            label_uploads = [
                upload for upload in uploads
                if upload.label_counts.get(label, 0) > 0 and upload.model_params
            ]
            if not label_uploads:
                continue

            label_vectors: list[np.ndarray] = []
            for upload in label_uploads:
                W, b = _split_label_params(
                    upload.model_params,
                    self.n_features,
                    self.n_classes,
                )
                label_vectors.append(
                    np.concatenate([W[label], np.array([b[label]], dtype=np.float64)]),
                )

            for indices in self._cluster_indices(label_vectors):
                members = [label_uploads[i] for i in indices]
                row, bias = self._aggregate_label_params(label, members)
                proto = self._aggregate_label_prototype(label, members)
                representation = self._cluster_representation(row, bias, proto)
                persistent_id = self._assign_persistent_cluster_id(
                    label,
                    representation,
                )

                for member in members:
                    update = updates[member.client_id]
                    update.label_vectors[label] = row.copy()
                    update.label_biases[label] = float(bias)
                    update.label_cluster_ids[label] = persistent_id
                    if proto is not None:
                        update.global_prototypes[label] = proto.copy()

        return updates

    def download_bytes(
        self,
        updates: dict[int, FedCCFAUpdate],
        precision_bits: int = 32,
    ) -> float:
        total = 0.0
        for update in updates.values():
            total += _label_update_bytes(
                update.label_vectors,
                update.label_biases,
                precision_bits,
            )
            total += prototype_bytes(update.global_prototypes, precision_bits)
        return total
