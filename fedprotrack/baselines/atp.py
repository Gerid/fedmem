from __future__ import annotations

"""ATP baseline: adaptive test-time personalization for linear models.

This is a lightweight adaptation of the official ATP implementation
from Bao et al. (NeurIPS 2023) to the repository's
``TorchLinearClassifier`` / ``DriftDataset`` stack.

Core idea preserved from the original code:
- maintain a per-parameter adaptation-rate vector;
- take an entropy-minimisation step on the current batch;
- update the adaptation rates from the agreement between the entropy
  gradient and a supervised gradient;
- optionally federate the adapted model and rate vector across clients.

The original ATP repo operates on deep networks with batch-norm modules
and explicit test-time personalization. Here we keep the same training
logic, but collapse it to the two trainable tensors of the linear
classifier (``coef`` and ``intercept``).
"""

from dataclasses import dataclass

import numpy as np
import torch

from ..models import TorchLinearClassifier
from .comm_tracker import model_bytes


@dataclass
class ATPUpload:
    """Payload uploaded by one ATP client."""

    client_id: int
    model_params: dict[str, np.ndarray]
    adaptation_rates: np.ndarray
    n_samples: int
    entropy: float


@dataclass
class ATPUpdate:
    """Aggregated state broadcast back to clients."""

    model_params: dict[str, np.ndarray]
    adaptation_rates: np.ndarray


@dataclass
class ATPResult:
    """Compact result returned by ``run_atp_full``."""

    method_name: str
    accuracy_matrix: np.ndarray
    total_bytes: float


def _copy_params(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: v.copy() for k, v in params.items()}


def _entropy_from_logits(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if n_classes == 2 and logits.shape[-1] == 1:
        probs = torch.sigmoid(logits.squeeze(-1))
        probs = torch.clamp(probs, 1e-6, 1 - 1e-6)
        entropy = -(
            probs * torch.log(probs)
            + (1 - probs) * torch.log(1 - probs)
        )
        return entropy.mean()

    probs = torch.softmax(logits, dim=-1)
    probs = torch.clamp(probs, 1e-6, 1.0)
    return -(probs * torch.log(probs)).sum(dim=-1).mean()


def _apply_scaled_step(
    params: list[torch.nn.Parameter],
    grads: list[torch.Tensor],
    rates: np.ndarray,
) -> None:
    with torch.no_grad():
        for param, grad, rate in zip(params, grads, rates, strict=True):
            if grad is None:
                continue
            param.add_(grad, alpha=-float(rate))


def _copy_trainable_grads(
    params: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    grads: list[torch.Tensor] = []
    for param in params:
        if param.grad is None:
            grads.append(torch.zeros_like(param))
        else:
            grads.append(param.grad.detach().clone())
    return grads


def _zero_grads(params: list[torch.nn.Parameter]) -> None:
    for param in params:
        if param.grad is not None:
            param.grad.zero_()


class ATPClient:
    """Client-side ATP adaptation for ``TorchLinearClassifier``."""

    def __init__(
        self,
        client_id: int,
        n_features: int,
        n_classes: int,
        *,
        base_lr: float = 0.05,
        meta_lr: float = 0.15,
        supervised_steps: int = 1,
        seed: int = 0,
    ) -> None:
        self.client_id = client_id
        self.n_features = n_features
        self.n_classes = n_classes
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        self.supervised_steps = supervised_steps
        self._seed = seed

        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=0.1,
            n_epochs=1,
            seed=seed,
        )
        self._adaptation_rates = np.full(
            len(list(self._model._linear.parameters())),
            base_lr,
            dtype=np.float64,
        )
        self._model_params: dict[str, np.ndarray] = {}
        self._n_samples = 0
        self._last_entropy = float("inf")

    def _train_supervised(self, X: np.ndarray, y: np.ndarray) -> None:
        for _ in range(self.supervised_steps):
            self._model.partial_fit(X, y)

    def _entropy_step(self, X: np.ndarray) -> tuple[list[torch.Tensor], float]:
        params = list(self._model._linear.parameters())
        self._model._linear.train()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self._model.device)
        logits = self._model._linear(X_t)
        loss = _entropy_from_logits(logits, self.n_classes)
        loss.backward()
        grads = _copy_trainable_grads(params)
        _zero_grads(params)
        _apply_scaled_step(params, grads, self._adaptation_rates)
        self._model._fitted = True
        return grads, float(loss.item())

    def _update_rates(
        self,
        X: np.ndarray,
        y: np.ndarray,
        unsup_grads: list[torch.Tensor],
    ) -> None:
        params = list(self._model._linear.parameters())
        self._model._linear.train()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self._model.device)
        y_t = torch.as_tensor(y, dtype=torch.long, device=self._model.device)
        logits = self._model._linear(X_t)
        if self.n_classes == 2 and logits.shape[-1] == 1:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), y_t.float(),
            )
        else:
            loss = torch.nn.functional.cross_entropy(logits, y_t)
        loss.backward()
        sup_grads = _copy_trainable_grads(params)
        _zero_grads(params)

        for idx, (sup, unsup) in enumerate(zip(sup_grads, unsup_grads, strict=True)):
            score = float(torch.dot(sup.flatten(), unsup.flatten()).item())
            score /= max(1, sup.numel())
            self._adaptation_rates[idx] = max(
                0.0,
                float(self._adaptation_rates[idx] + self.meta_lr * score),
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Update the local model on one batch."""
        self._n_samples += len(X)

        if not self._model._fitted:
            self._train_supervised(X, y)
            self._model_params = self._model.get_params()
            return

        unsup_grads, entropy = self._entropy_step(X)
        self._last_entropy = entropy
        self._update_rates(X, y, unsup_grads)
        self._train_supervised(X, y)
        self._model_params = self._model.get_params()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    @property
    def adaptation_rates(self) -> np.ndarray:
        return self._adaptation_rates.copy()

    def set_model_state(
        self,
        params: dict[str, np.ndarray],
        adaptation_rates: np.ndarray | None = None,
    ) -> None:
        self._model.set_params(params)
        self._model_params = _copy_params(params)
        if adaptation_rates is not None:
            self._adaptation_rates = np.asarray(adaptation_rates, dtype=np.float64).copy()

    def get_upload(self) -> ATPUpload:
        return ATPUpload(
            client_id=self.client_id,
            model_params=_copy_params(self._model_params),
            adaptation_rates=self._adaptation_rates.copy(),
            n_samples=self._n_samples,
            entropy=self._last_entropy,
        )

    def upload_bytes(self, precision_bits: int = 32) -> float:
        rate_bytes = float(self._adaptation_rates.size * precision_bits / 8)
        return model_bytes(self._model_params, precision_bits=precision_bits) + rate_bytes


class ATPServer:
    """Federates ATP clients by averaging both model weights and rates."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        base_lr: float = 0.05,
        meta_lr: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        self._seed = seed
        self._model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=0.1,
            n_epochs=1,
            seed=seed,
        )
        self._global_params: dict[str, np.ndarray] = {}
        self._global_rates = np.full(
            len(list(self._model._linear.parameters())),
            base_lr,
            dtype=np.float64,
        )

    def aggregate(self, uploads: list[ATPUpload]) -> ATPUpdate:
        if not uploads:
            return ATPUpdate({}, self._global_rates.copy())

        total = sum(max(1, u.n_samples) for u in uploads)
        weighted_params: dict[str, np.ndarray] = {}
        for key in uploads[0].model_params:
            accum = None
            for upload in uploads:
                weight = max(1, upload.n_samples)
                arr = upload.model_params[key]
                accum = arr * weight if accum is None else accum + arr * weight
            weighted_params[key] = accum / total

        rates = np.zeros_like(self._global_rates)
        for upload in uploads:
            rates += upload.adaptation_rates * max(1, upload.n_samples)
        rates /= total

        self._global_params = _copy_params(weighted_params)
        self._global_rates = rates.copy()
        return ATPUpdate(weighted_params, rates)

    def download_bytes(
        self,
        update: ATPUpdate,
        n_clients: int,
        precision_bits: int = 32,
    ) -> float:
        rate_bytes = float(update.adaptation_rates.size * precision_bits / 8)
        return (
            model_bytes(update.model_params, precision_bits=precision_bits) * max(1, n_clients)
            + rate_bytes * max(1, n_clients)
        )

    def broadcast(self, clients: list[ATPClient]) -> None:
        if not self._global_params:
            return
        for client in clients:
            client.set_model_state(self._global_params, self._global_rates)


def run_atp_full(
    dataset,
    federation_every: int = 1,
    *,
    base_lr: float = 0.05,
    meta_lr: float = 0.15,
) -> ATPResult:
    """Run a small ATP simulation on a ``DriftDataset``."""

    K = dataset.config.K
    T = dataset.config.T
    X0, _ = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    all_labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    clients = [
        ATPClient(
            k,
            n_features,
            n_classes,
            base_lr=base_lr,
            meta_lr=meta_lr,
            seed=42 + k,
        )
        for k in range(K)
    ]
    server = ATPServer(
        n_features,
        n_classes,
        base_lr=base_lr,
        meta_lr=meta_lr,
        seed=42,
    )

    accuracy = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy[k, t] = float(np.mean(preds == y)) if len(y) else 0.0

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T - 1:
            uploads = [client.get_upload() for client in clients]
            upload_bytes = sum(client.upload_bytes() for client in clients)
            update = server.aggregate(uploads)
            total_bytes += upload_bytes + server.download_bytes(update, K)
            server.broadcast(clients)

    return ATPResult(
        method_name="ATP",
        accuracy_matrix=accuracy,
        total_bytes=total_bytes,
    )
