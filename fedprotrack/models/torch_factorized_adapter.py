from __future__ import annotations

"""Factorized shared/private routed model for Stage-2 adapter experiments."""

import re

import numpy as np
import torch
import torch.nn as nn

from ..device import get_device, to_numpy, to_tensor

_SLOT_KEY_RE = re.compile(r"^expert\.(?P<slot>\d+)\.(?P<name>.+)$")


class TorchFactorizedAdapterClassifier:
    """Classifier with a shared encoder and per-slot private encoders/heads."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 64,
        adapter_dim: int = 16,
        lr: float = 0.05,
        n_epochs: int = 5,
        seed: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.private_dim = adapter_dim
        self.adapter_dim = adapter_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self._seed = seed
        self._n_out = 1 if n_classes == 2 else n_classes

        base_params = (
            n_features * hidden_dim
            + hidden_dim
            + n_features * adapter_dim
            + adapter_dim
            + (hidden_dim + adapter_dim) * self._n_out
            + self._n_out
        )
        self.device = device or get_device(n_params=base_params)
        torch.manual_seed(seed)

        self._shared_encoder = nn.Linear(n_features, hidden_dim).to(self.device)
        self._private_encoders: dict[int, nn.Linear] = {}
        self._heads: dict[int, nn.Linear] = {}
        self._optimizers: dict[int, torch.optim.Optimizer] = {}
        self._fitted = False
        self._active_slot = 0
        self._ensure_slot(0)

    def _ensure_slot(self, slot_id: int) -> None:
        if slot_id in self._private_encoders:
            return
        private_encoder = nn.Linear(self.n_features, self.private_dim).to(self.device)
        head = nn.Linear(self.hidden_dim + self.private_dim, self._n_out).to(self.device)
        self._private_encoders[slot_id] = private_encoder
        self._heads[slot_id] = head
        params = (
            list(self._shared_encoder.parameters())
            + list(private_encoder.parameters())
            + list(head.parameters())
        )
        self._optimizers[slot_id] = torch.optim.SGD(params, lr=self.lr)

    def _shared_embed_tensor(self, X_t: torch.Tensor) -> torch.Tensor:
        return torch.relu(self._shared_encoder(X_t))

    def _private_embed_tensor(self, X_t: torch.Tensor, slot_id: int) -> torch.Tensor:
        self._ensure_slot(slot_id)
        private_encoder = self._private_encoders[slot_id]
        return torch.relu(private_encoder(X_t))

    def _combined_embed_tensor(self, X_t: torch.Tensor, slot_id: int) -> torch.Tensor:
        shared_hidden = self._shared_embed_tensor(X_t)
        private_hidden = self._private_embed_tensor(X_t, slot_id)
        return torch.cat([shared_hidden, private_hidden], dim=-1)

    def _forward_slot(self, X_t: torch.Tensor, slot_id: int) -> torch.Tensor:
        self._ensure_slot(slot_id)
        head = self._heads[slot_id]
        features = self._combined_embed_tensor(X_t, slot_id)
        return head(features)

    def _loss(self, logits: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        if self._n_out == 1:
            return nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), y_t.float(),
            )
        return nn.functional.cross_entropy(logits, y_t)

    def _normalise_training_weights(
        self,
        slot_id: int,
        slot_weights: dict[int, float] | None = None,
    ) -> dict[int, float]:
        if not slot_weights:
            self._ensure_slot(slot_id)
            return {int(slot_id): 1.0}

        filtered = {
            int(current_slot): float(weight)
            for current_slot, weight in slot_weights.items()
            if float(weight) > 0.01
        }
        if not filtered:
            self._ensure_slot(slot_id)
            return {int(slot_id): 1.0}

        total = float(sum(filtered.values()))
        normalized = {
            current_slot: weight / total
            for current_slot, weight in filtered.items()
        }
        for current_slot in normalized:
            self._ensure_slot(int(current_slot))
        return normalized

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
        update_shared: bool = True,
        update_private: bool = True,
        update_head: bool = True,
    ) -> None:
        training_weights = self._normalise_training_weights(slot_id, slot_weights)
        self._active_slot = max(training_weights, key=training_weights.get)
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        for _ in range(self.n_epochs):
            for current_slot, weight in training_weights.items():
                self._step(
                    X_t,
                    y_t,
                    current_slot,
                    loss_scale=weight,
                    update_shared=update_shared,
                    update_private=update_private,
                    update_head=update_head,
                )
        self._fitted = True

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
        update_shared: bool = True,
        update_private: bool = True,
        update_head: bool = True,
    ) -> None:
        training_weights = self._normalise_training_weights(slot_id, slot_weights)
        self._active_slot = max(training_weights, key=training_weights.get)
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        for current_slot, weight in training_weights.items():
            self._step(
                X_t,
                y_t,
                current_slot,
                loss_scale=weight,
                update_shared=update_shared,
                update_private=update_private,
                update_head=update_head,
            )
        self._fitted = True

    def _step(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        slot_id: int,
        *,
        loss_scale: float = 1.0,
        update_shared: bool = True,
        update_private: bool = True,
        update_head: bool = True,
    ) -> None:
        self._shared_encoder.train()
        self._private_encoders[slot_id].train()
        self._heads[slot_id].train()
        optimizer = self._optimizers[slot_id]
        optimizer.zero_grad()
        logits = self._forward_slot(X_t, slot_id)
        loss = self._loss(logits, y_t) * float(loss_scale)
        loss.backward()
        if not update_shared:
            for param in self._shared_encoder.parameters():
                param.grad = None
        if not update_private:
            for param in self._private_encoders[slot_id].parameters():
                param.grad = None
        if not update_head:
            for param in self._heads[slot_id].parameters():
                param.grad = None
        optimizer.step()

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
    ) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X), dtype=np.int64)
        logits = self._predict_logits(X, slot_id=slot_id, slot_weights=slot_weights)
        if self._n_out == 1:
            preds = (logits.squeeze(-1) > 0).long()
        else:
            preds = logits.argmax(dim=-1)
        return to_numpy(preds).astype(np.int64)

    @torch.no_grad()
    def predict_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
    ) -> float:
        if not self._fitted:
            return float("inf")
        logits = self._predict_logits(X, slot_id=slot_id, slot_weights=slot_weights)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        return float(self._loss(logits, y_t).item())

    @torch.no_grad()
    def embed(
        self,
        X: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
        representation: str = "post_adapter",
    ) -> np.ndarray:
        if not self._fitted:
            width = self.hidden_dim if representation == "pre_adapter" else self.hidden_dim + self.private_dim
            return np.zeros((len(X), width), dtype=np.float32)
        hidden = self._predict_hidden(
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation=representation,
        )
        return to_numpy(hidden).astype(np.float32, copy=False)

    @torch.no_grad()
    def _predict_hidden(
        self,
        X: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
        representation: str = "post_adapter",
    ) -> torch.Tensor:
        if representation not in {"post_adapter", "pre_adapter"}:
            raise ValueError(
                "representation must be one of ['post_adapter', 'pre_adapter']"
            )
        X_t = to_tensor(X, device=self.device)
        self._shared_encoder.eval()
        shared_hidden = self._shared_embed_tensor(X_t)
        if representation == "pre_adapter":
            return shared_hidden

        if slot_weights:
            total = float(sum(slot_weights.values()))
            if total <= 0.0:
                raise ValueError("slot_weights must sum to a positive value")
            blended = None
            for current_slot, weight in slot_weights.items():
                self._ensure_slot(int(current_slot))
                private_hidden = self._private_embed_tensor(X_t, int(current_slot))
                hidden = torch.cat([shared_hidden, private_hidden], dim=-1)
                scaled = (float(weight) / total) * hidden
                blended = scaled if blended is None else blended + scaled
            assert blended is not None
            return blended

        self._ensure_slot(slot_id)
        return self._combined_embed_tensor(X_t, slot_id)

    @torch.no_grad()
    def _predict_logits(
        self,
        X: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
    ) -> torch.Tensor:
        X_t = to_tensor(X, device=self.device)
        self._shared_encoder.eval()
        if slot_weights:
            total = float(sum(slot_weights.values()))
            if total <= 0.0:
                raise ValueError("slot_weights must sum to a positive value")
            blended = None
            for current_slot, weight in slot_weights.items():
                self._ensure_slot(int(current_slot))
                self._private_encoders[int(current_slot)].eval()
                self._heads[int(current_slot)].eval()
                logits = self._forward_slot(X_t, int(current_slot))
                scaled = (float(weight) / total) * logits
                blended = scaled if blended is None else blended + scaled
            assert blended is not None
            return blended

        self._ensure_slot(slot_id)
        self._private_encoders[slot_id].eval()
        self._heads[slot_id].eval()
        return self._forward_slot(X_t, slot_id)

    def get_params(self, slot_id: int | None = None) -> dict[str, np.ndarray]:
        if slot_id is None:
            slot_id = self._active_slot
        self._ensure_slot(slot_id)
        private_encoder = self._private_encoders[slot_id]
        head = self._heads[slot_id]
        return {
            "shared.encoder.weight": to_numpy(self._shared_encoder.weight.data).copy(),
            "shared.encoder.bias": to_numpy(self._shared_encoder.bias.data).copy(),
            f"expert.{slot_id}.private.weight": to_numpy(private_encoder.weight.data).copy(),
            f"expert.{slot_id}.private.bias": to_numpy(private_encoder.bias.data).copy(),
            f"expert.{slot_id}.head.weight": to_numpy(head.weight.data).copy(),
            f"expert.{slot_id}.head.bias": to_numpy(head.bias.data).copy(),
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            self._shared_encoder.weight.copy_(
                to_tensor(params["shared.encoder.weight"], device=self.device)
            )
            self._shared_encoder.bias.copy_(
                to_tensor(params["shared.encoder.bias"], device=self.device)
            )
            slot_groups: dict[int, dict[str, np.ndarray]] = {}
            for key, value in params.items():
                if key.startswith("shared."):
                    continue
                match = _SLOT_KEY_RE.match(key)
                if match is None:
                    slot_groups.setdefault(0, {})[
                        f"expert.0.{key[len('expert.'):]}"
                        if key.startswith("expert.")
                        else key
                    ] = value
                    continue
                slot_id = int(match.group("slot"))
                slot_groups.setdefault(slot_id, {})[key] = value

            for slot_id, slot_params in slot_groups.items():
                self._ensure_slot(slot_id)
                self._active_slot = slot_id
                private_encoder = self._private_encoders[slot_id]
                head = self._heads[slot_id]
                private_encoder.weight.copy_(
                    to_tensor(
                        slot_params[f"expert.{slot_id}.private.weight"],
                        device=self.device,
                    )
                )
                private_encoder.bias.copy_(
                    to_tensor(
                        slot_params[f"expert.{slot_id}.private.bias"],
                        device=self.device,
                    )
                )
                head.weight.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.head.weight"], device=self.device)
                )
                head.bias.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.head.bias"], device=self.device)
                )

        for slot_id in list(self._private_encoders):
            params_for_slot = (
                list(self._shared_encoder.parameters())
                + list(self._private_encoders[slot_id].parameters())
                + list(self._heads[slot_id].parameters())
            )
            self._optimizers[slot_id] = torch.optim.SGD(params_for_slot, lr=self.lr)
        self._fitted = True

    def blend_params(self, other_params: dict[str, np.ndarray], alpha: float = 0.5) -> None:
        own = self.get_params()
        common_keys = set(own) & set(other_params)
        blended = {key: own[key].copy() for key in own}
        for key in common_keys:
            blended[key] = alpha * other_params[key] + (1.0 - alpha) * own[key]
        self.set_params(blended)

    def clone_fresh(self, seed: int | None = None) -> TorchFactorizedAdapterClassifier:
        return TorchFactorizedAdapterClassifier(
            n_features=self.n_features,
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim,
            adapter_dim=self.private_dim,
            lr=self.lr,
            n_epochs=self.n_epochs,
            seed=seed if seed is not None else self._seed + 1,
            device=self.device,
        )
