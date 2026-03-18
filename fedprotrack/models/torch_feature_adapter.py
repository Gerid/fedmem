from __future__ import annotations

"""Feature-space shared-trunk + slot-expert model for Plan C experiments."""

import re

import numpy as np
import torch
import torch.nn as nn

from ..device import get_device, to_numpy, to_tensor

_SLOT_KEY_RE = re.compile(r"^expert\.(?P<slot>\d+)\.(?P<name>.+)$")


class _AdapterBlock(nn.Module):
    def __init__(self, hidden_dim: int, adapter_dim: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(torch.relu(self.down(x)))


class TorchFeatureAdapterClassifier:
    """Feature-space classifier with a shared trunk and per-slot experts."""

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
        self.adapter_dim = adapter_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self._seed = seed
        self._n_out = 1 if n_classes == 2 else n_classes

        base_params = (
            n_features * hidden_dim
            + hidden_dim
            + hidden_dim * adapter_dim
            + adapter_dim
            + adapter_dim * hidden_dim
            + hidden_dim
            + hidden_dim * self._n_out
            + self._n_out
        )
        self.device = device or get_device(n_params=base_params)
        torch.manual_seed(seed)

        self._trunk = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
        ).to(self.device)
        self._experts: dict[int, tuple[_AdapterBlock, nn.Linear]] = {}
        self._optimizers: dict[int, torch.optim.Optimizer] = {}
        self._fitted = False
        self._active_slot = 0
        self._ensure_slot(0)

    def _ensure_slot(self, slot_id: int) -> None:
        if slot_id in self._experts:
            return
        adapter = _AdapterBlock(self.hidden_dim, self.adapter_dim).to(self.device)
        head = nn.Linear(self.hidden_dim, self._n_out).to(self.device)
        self._experts[slot_id] = (adapter, head)
        params = list(self._trunk.parameters()) + list(adapter.parameters()) + list(head.parameters())
        self._optimizers[slot_id] = torch.optim.SGD(params, lr=self.lr)

    def _forward_slot(self, X_t: torch.Tensor, slot_id: int) -> torch.Tensor:
        self._ensure_slot(slot_id)
        adapter, head = self._experts[slot_id]
        hidden = self._trunk(X_t)
        hidden = adapter(hidden)
        return head(hidden)

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
        normalised = {
            current_slot: weight / total
            for current_slot, weight in filtered.items()
        }
        for current_slot in normalised:
            self._ensure_slot(int(current_slot))
        return normalised

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
    ) -> None:
        training_weights = self._normalise_training_weights(slot_id, slot_weights)
        self._active_slot = max(training_weights, key=training_weights.get)
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        for _ in range(self.n_epochs):
            for current_slot, weight in training_weights.items():
                self._step(X_t, y_t, current_slot, loss_scale=weight)
        self._fitted = True

    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
    ) -> None:
        training_weights = self._normalise_training_weights(slot_id, slot_weights)
        self._active_slot = max(training_weights, key=training_weights.get)
        X_t = to_tensor(X, device=self.device)
        y_t = to_tensor(y, dtype=torch.long, device=self.device)
        for current_slot, weight in training_weights.items():
            self._step(X_t, y_t, current_slot, loss_scale=weight)
        self._fitted = True

    def _step(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        slot_id: int,
        *,
        loss_scale: float = 1.0,
    ) -> None:
        self._trunk.train()
        adapter, head = self._experts[slot_id]
        adapter.train()
        head.train()
        optimizer = self._optimizers[slot_id]
        optimizer.zero_grad()
        logits = self._forward_slot(X_t, slot_id)
        loss = self._loss(logits, y_t) * float(loss_scale)
        loss.backward()
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
    def _predict_logits(
        self,
        X: np.ndarray,
        slot_id: int = 0,
        slot_weights: dict[int, float] | None = None,
    ) -> torch.Tensor:
        X_t = to_tensor(X, device=self.device)
        self._trunk.eval()
        if slot_weights:
            total = float(sum(slot_weights.values()))
            if total <= 0.0:
                raise ValueError("slot_weights must sum to a positive value")
            blended = None
            for current_slot, weight in slot_weights.items():
                self._ensure_slot(int(current_slot))
                adapter, head = self._experts[int(current_slot)]
                adapter.eval()
                head.eval()
                logits = self._forward_slot(X_t, int(current_slot))
                scaled = (float(weight) / total) * logits
                blended = scaled if blended is None else blended + scaled
            assert blended is not None
            return blended

        self._ensure_slot(slot_id)
        adapter, head = self._experts[slot_id]
        adapter.eval()
        head.eval()
        return self._forward_slot(X_t, slot_id)

    def get_params(self, slot_id: int | None = None) -> dict[str, np.ndarray]:
        if slot_id is None:
            slot_id = self._active_slot
        self._ensure_slot(slot_id)
        adapter, head = self._experts[slot_id]
        return {
            "shared.trunk.weight": to_numpy(self._trunk[0].weight.data).copy(),
            "shared.trunk.bias": to_numpy(self._trunk[0].bias.data).copy(),
            f"expert.{slot_id}.adapter.down.weight": to_numpy(adapter.down.weight.data).copy(),
            f"expert.{slot_id}.adapter.down.bias": to_numpy(adapter.down.bias.data).copy(),
            f"expert.{slot_id}.adapter.up.weight": to_numpy(adapter.up.weight.data).copy(),
            f"expert.{slot_id}.adapter.up.bias": to_numpy(adapter.up.bias.data).copy(),
            f"expert.{slot_id}.head.weight": to_numpy(head.weight.data).copy(),
            f"expert.{slot_id}.head.bias": to_numpy(head.bias.data).copy(),
        }

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            self._trunk[0].weight.copy_(
                to_tensor(params["shared.trunk.weight"], device=self.device)
            )
            self._trunk[0].bias.copy_(
                to_tensor(params["shared.trunk.bias"], device=self.device)
            )
            slot_groups: dict[int, dict[str, np.ndarray]] = {}
            for key, value in params.items():
                if key.startswith("shared."):
                    continue
                match = _SLOT_KEY_RE.match(key)
                if match is None:
                    # Backward-compatible single-slot expert payload.
                    slot_groups.setdefault(0, {})[f"expert.0.{key[len('expert.'):]}" if key.startswith("expert.") else key] = value
                    continue
                slot_id = int(match.group("slot"))
                slot_groups.setdefault(slot_id, {})[key] = value

            for slot_id, slot_params in slot_groups.items():
                self._ensure_slot(slot_id)
                self._active_slot = slot_id
                adapter, head = self._experts[slot_id]
                adapter.down.weight.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.adapter.down.weight"], device=self.device)
                )
                adapter.down.bias.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.adapter.down.bias"], device=self.device)
                )
                adapter.up.weight.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.adapter.up.weight"], device=self.device)
                )
                adapter.up.bias.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.adapter.up.bias"], device=self.device)
                )
                head.weight.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.head.weight"], device=self.device)
                )
                head.bias.copy_(
                    to_tensor(slot_params[f"expert.{slot_id}.head.bias"], device=self.device)
                )

        for slot_id in list(self._experts):
            adapter, head = self._experts[slot_id]
            params_for_slot = list(self._trunk.parameters()) + list(adapter.parameters()) + list(head.parameters())
            self._optimizers[slot_id] = torch.optim.SGD(params_for_slot, lr=self.lr)
        self._fitted = True

    def blend_params(self, other_params: dict[str, np.ndarray], alpha: float = 0.5) -> None:
        own = self.get_params()
        common_keys = set(own) & set(other_params)
        blended = {key: own[key].copy() for key in own}
        for key in common_keys:
            blended[key] = alpha * other_params[key] + (1.0 - alpha) * own[key]
        self.set_params(blended)

    def clone_fresh(self, seed: int | None = None) -> TorchFeatureAdapterClassifier:
        return TorchFeatureAdapterClassifier(
            n_features=self.n_features,
            n_classes=self.n_classes,
            hidden_dim=self.hidden_dim,
            adapter_dim=self.adapter_dim,
            lr=self.lr,
            n_epochs=self.n_epochs,
            seed=seed if seed is not None else self._seed + 1,
            device=self.device,
        )
