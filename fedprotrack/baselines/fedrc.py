from __future__ import annotations

"""FedRC baseline: robust clustering under distribution shifts.

This is a lightweight port of the original LINs-lab FedRC client update
rules. The upstream implementation maintains soft cluster assignments and
uses label-aware weighting when recomputing per-client responsibilities.
Here we adapt that logic to a single ``TorchLinearClassifier`` per client
and a small server-side ensemble of cluster models.
"""

from dataclasses import dataclass

import numpy as np

from ..federation.aggregator import FedAvgAggregator
from ..models import TorchLinearClassifier
from ..models.factory import create_model
from .comm_tracker import model_bytes


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(initial=0.0)
    ex = np.exp(x)
    denom = ex.sum()
    if denom <= 0:
        return np.ones_like(x) / len(x)
    return ex / denom


def _label_histogram(y: np.ndarray, n_classes: int) -> np.ndarray:
    hist = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return np.ones(n_classes, dtype=np.float64) / max(n_classes, 1)
    return hist / total


def _fresh_params(
    n_features: int,
    n_classes: int,
    seed: int,
    model_type: str = "linear",
) -> dict[str, np.ndarray]:
    model = create_model(model_type, n_features, n_classes, seed=seed)
    return model.get_params()


@dataclass
class FedRCUpload:
    """Client-side contribution to one FedRC round."""

    client_id: int
    model_params: dict[str, np.ndarray]
    cluster_probs: np.ndarray
    label_hist: np.ndarray
    n_samples: int
    batch_size: int
    selected_cluster: int


class FedRCClient:
    """Client for the lightweight FedRC adapter."""

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        n_clusters: int = 3,
        lr: float = 0.1,
        n_epochs: int = 10,
        label_prior_strength: float = 0.5,
        prior_strength: float = 0.1,
        seed: int = 0,
        model_type: str = "linear",
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_clusters = n_clusters
        self.label_prior_strength = label_prior_strength
        self.prior_strength = prior_strength
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
        self._cluster_models: list[dict[str, np.ndarray]] = []
        self._cluster_label_hists: list[np.ndarray] = []
        self._cluster_probs = np.ones(n_clusters, dtype=np.float64) / max(n_clusters, 1)
        self._selected_cluster = 0
        self._model_params: dict[str, np.ndarray] = {}
        self._label_hist = np.ones(n_classes, dtype=np.float64) / max(n_classes, 1)
        self._n_samples = 0

    def set_cluster_state(
        self,
        cluster_models: list[dict[str, np.ndarray]],
        cluster_label_hists: list[np.ndarray] | None = None,
    ) -> None:
        self._cluster_models = [_copy_params(params) for params in cluster_models]
        if cluster_label_hists is None:
            self._cluster_label_hists = [
                np.ones(self.n_classes, dtype=np.float64) / max(self.n_classes, 1)
                for _ in cluster_models
            ]
        else:
            self._cluster_label_hists = [hist.copy() for hist in cluster_label_hists]
        if self._cluster_models and self._selected_cluster < len(self._cluster_models):
            self._model.set_params(_copy_params(self._cluster_models[self._selected_cluster]))

    def _evaluate_cluster_models(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        losses = []
        for params in self._cluster_models:
            if not params:
                losses.append(float("inf"))
                continue
            temp = create_model(
                self._model_type, self.n_features, self.n_classes,
                seed=self._seed,
            )
            temp.set_params(_copy_params(params))
            losses.append(temp.predict_loss(X, y))
        return np.asarray(losses, dtype=np.float64)

    def _cluster_score(self, losses: np.ndarray, label_hist: np.ndarray) -> np.ndarray:
        if losses.size == 0:
            return np.ones(self.n_clusters, dtype=np.float64) / max(self.n_clusters, 1)

        prior = self._cluster_probs
        if prior.shape[0] != losses.shape[0]:
            prior = np.ones_like(losses) / max(len(losses), 1)

        label_scores = np.zeros_like(losses)
        for idx, cluster_hist in enumerate(self._cluster_label_hists[: len(losses)]):
            label_scores[idx] = float(np.dot(label_hist, cluster_hist))

        scores = (
            -losses
            + self.label_prior_strength * label_scores
            + self.prior_strength * np.log(prior + 1e-12)
        )
        return _softmax(scores)

    def _blend_cluster_models(self, probs: np.ndarray) -> dict[str, np.ndarray]:
        valid = [
            (params, float(prob))
            for params, prob in zip(self._cluster_models, probs)
            if params and prob > 0.0
        ]
        if not valid:
            return {}
        params_list = [params for params, _ in valid]
        weights = [weight for _, weight in valid]
        return FedAvgAggregator().aggregate(params_list, weights)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._n_samples += int(len(X))
        self._batch_size = int(len(X))
        label_hist = _label_histogram(y, self.n_classes)
        self._label_hist = label_hist

        if not self._cluster_models:
            self._model.fit(X, y)
            self._model_params = self._model.get_params()
            self._cluster_probs = np.ones(self.n_clusters, dtype=np.float64) / max(self.n_clusters, 1)
            self._selected_cluster = 0
            return

        losses = self._evaluate_cluster_models(X, y)
        probs = self._cluster_score(losses, label_hist)
        self._cluster_probs = probs
        self._selected_cluster = int(np.argmax(probs))

        blended = self._blend_cluster_models(probs)
        if blended:
            self._model.set_params(blended)
        elif self._selected_cluster < len(self._cluster_models):
            self._model.set_params(_copy_params(self._cluster_models[self._selected_cluster]))

        self._model.fit(X, y)
        self._model_params = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_upload(self) -> FedRCUpload:
        return FedRCUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            cluster_probs=self._cluster_probs.copy(),
            label_hist=self._label_hist.copy(),
            n_samples=self._n_samples,
            batch_size=getattr(self, "_batch_size", self._n_samples),
            selected_cluster=self._selected_cluster,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        return model_bytes(self._model_params, precision_bits)


class FedRCServer:
    """Server for the lightweight FedRC adapter."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_clusters: int = 3,
        seed: int = 0,
        model_type: str = "linear",
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_clusters = n_clusters
        self.seed = seed
        self._aggregator = FedAvgAggregator()
        self.cluster_models: list[dict[str, np.ndarray]] = [
            _fresh_params(n_features, n_classes, seed + idx, model_type=model_type)
            for idx in range(n_clusters)
        ]
        self.cluster_label_hists: list[np.ndarray] = [
            np.ones(n_classes, dtype=np.float64) / max(n_classes, 1)
            for _ in range(n_clusters)
        ]

    def broadcast(self) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
        return (
            [_copy_params(params) for params in self.cluster_models],
            [hist.copy() for hist in self.cluster_label_hists],
        )

    def aggregate(self, uploads: list[FedRCUpload]) -> list[dict[str, np.ndarray]]:
        if not uploads:
            return [_copy_params(params) for params in self.cluster_models]

        cluster_groups: dict[int, list[FedRCUpload]] = {idx: [] for idx in range(len(self.cluster_models))}
        for upload in uploads:
            probs = upload.cluster_probs
            if probs.size != len(self.cluster_models):
                probs = np.ones(len(self.cluster_models), dtype=np.float64) / max(len(self.cluster_models), 1)
            cluster_id = int(np.argmax(probs))
            cluster_groups[cluster_id].append(upload)

        for cluster_id, members in cluster_groups.items():
            valid = [m for m in members if m.model_params]
            if not valid:
                continue

            weights = [float(max(m.batch_size, 1)) * float(max(m.cluster_probs[cluster_id], 1e-6)) for m in valid]
            self.cluster_models[cluster_id] = self._aggregator.aggregate(
                [m.model_params for m in valid],
                weights,
            )

            label_hists = np.stack([m.label_hist for m in valid])
            w = np.asarray(weights, dtype=np.float64)
            if np.sum(w) > 0:
                w = w / np.sum(w)
                self.cluster_label_hists[cluster_id] = np.tensordot(w, label_hists, axes=([0], [0]))

        return [_copy_params(params) for params in self.cluster_models]

    def download_bytes(self, n_clients: int, precision_bits: int = 32) -> float:
        per_client = sum(model_bytes(params, precision_bits) for params in self.cluster_models)
        return float(n_clients) * float(per_client)

