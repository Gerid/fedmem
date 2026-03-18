"""Federated aggregation strategies.

Provides standard FedAvg and the concept-aware variant used by FedProTrack.
The concept-aware aggregator only averages models that share the same predicted
concept identity, avoiding destructive interference between heterogeneous
distributions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import re

import numpy as np

_EXPERT_PREFIX_RE = re.compile(r"^expert\.(?P<slot>\d+)\.(?P<name>.+)$")


@dataclass
class NamespacedAggregationResult:
    """Aggregation result for shared + expert parameter payloads."""

    shared_params: dict[str, np.ndarray] = field(default_factory=dict)
    expert_params: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)

    def to_flat_params(self) -> dict[str, np.ndarray]:
        """Flatten shared and expert payloads into a single dict."""
        flat = {key: value.copy() for key, value in self.shared_params.items()}
        for slot_id, slot_params in self.expert_params.items():
            prefix = f"expert.{slot_id}."
            for local_key, value in slot_params.items():
                name = local_key[len(prefix):] if local_key.startswith(prefix) else local_key
                flat[f"{prefix}{name}"] = value.copy()
        return flat


def split_param_namespaces(
    params: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Split params into shared/expert/other buckets."""
    shared: dict[str, np.ndarray] = {}
    expert: dict[str, np.ndarray] = {}
    other: dict[str, np.ndarray] = {}
    for key, value in params.items():
        if key.startswith("shared."):
            shared[key] = value.copy()
        elif key.startswith("expert."):
            expert[key] = value.copy()
        else:
            other[key] = value.copy()
    return shared, expert, other


def has_namespaced_params(params: dict[str, np.ndarray]) -> bool:
    """Return whether a payload uses shared/expert namespaces."""
    return any(
        key.startswith("shared.") or key.startswith("expert.")
        for key in params
    )


def merge_param_namespaces(
    shared: dict[str, np.ndarray] | None = None,
    expert: dict[str, np.ndarray] | None = None,
    other: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Merge namespace buckets back into a flat payload."""
    merged: dict[str, np.ndarray] = {}
    if shared:
        merged.update({key: value.copy() for key, value in shared.items()})
    if expert:
        merged.update({key: value.copy() for key, value in expert.items()})
    if other:
        merged.update({key: value.copy() for key, value in other.items()})
    return merged


class BaseAggregator(ABC):
    """Interface for federated model aggregation."""

    @abstractmethod
    def aggregate(
        self,
        client_params: list[dict[str, np.ndarray]],
        weights: list[float] | None = None,
        concept_ids: list[int] | None = None,
    ) -> dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]]:
        """Aggregate model parameters from multiple clients.

        Parameters
        ----------
        client_params : list of dict
            Each dict maps parameter names to numpy arrays.
        weights : list of float, optional
            Per-client weights (e.g., proportional to dataset size).
        concept_ids : list of int, optional
            Predicted concept ID for each client (used by concept-aware methods).

        Returns
        -------
        Aggregated parameters. For standard methods, a single dict.
        For concept-aware methods, a dict mapping concept_id to params.
        """

    @property
    @abstractmethod
    def name(self) -> str: ...


class FedAvgAggregator(BaseAggregator):
    """Standard Federated Averaging (McMahan et al., 2017).

    Computes the weighted average of all client model parameters,
    regardless of concept heterogeneity.
    """

    def aggregate(
        self,
        client_params: list[dict[str, np.ndarray]],
        weights: list[float] | None = None,
        concept_ids: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        if not client_params:
            return {}

        if weights is None:
            weights = [1.0 / len(client_params)] * len(client_params)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        result: dict[str, np.ndarray] = {}
        for key in client_params[0]:
            stacked = np.stack([p[key] for p in client_params])
            w = np.array(weights).reshape(-1, *([1] * (stacked.ndim - 1)))
            result[key] = np.sum(stacked * w, axis=0)

        return result

    @property
    def name(self) -> str:
        return "FedAvg"


class ConceptAwareFedAvgAggregator(BaseAggregator):
    """Concept-aware Federated Averaging (FedProTrack contribution).

    Groups clients by their predicted concept ID and performs FedAvg
    within each group independently. This prevents averaging models
    trained on fundamentally different distributions.

    When a concept group has only one client, no aggregation occurs
    (the client's model is used as-is).
    """

    def aggregate(
        self,
        client_params: list[dict[str, np.ndarray]],
        weights: list[float] | None = None,
        concept_ids: list[int] | None = None,
    ) -> dict[int, dict[str, np.ndarray]]:
        if not client_params:
            return {}

        if concept_ids is None:
            raise ValueError(
                "ConceptAwareFedAvgAggregator requires concept_ids"
            )

        if weights is None:
            weights = [1.0] * len(client_params)

        # Group by concept
        groups: dict[int, list[tuple[dict[str, np.ndarray], float]]] = defaultdict(list)
        for params, w, cid in zip(client_params, weights, concept_ids):
            groups[cid].append((params, w))

        # Aggregate within each group
        fedavg = FedAvgAggregator()
        result: dict[int, dict[str, np.ndarray]] = {}
        for cid, group in groups.items():
            g_params = [p for p, _ in group]
            g_weights = [w for _, w in group]
            result[cid] = fedavg.aggregate(g_params, g_weights)

        return result

    @property
    def name(self) -> str:
        return "ConceptAwareFedAvg"


class NamespacedExpertAggregator(BaseAggregator):
    """Aggregate shared params globally and expert params by slot."""

    def aggregate(
        self,
        client_params: list[dict[str, np.ndarray]],
        weights: list[float] | None = None,
        concept_ids: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        result = self.aggregate_namespaced(client_params, weights=weights)
        return result.to_flat_params()

    def aggregate_namespaced(
        self,
        client_params: list[dict[str, np.ndarray]],
        weights: list[float] | None = None,
        expert_weights: list[dict[int, float]] | None = None,
    ) -> NamespacedAggregationResult:
        """Aggregate a namespaced payload.

        ``shared.*`` keys are averaged across all clients. ``expert.<slot>.*``
        keys are averaged only among clients contributing to the same slot.
        Optional ``expert_weights`` can down/up-weight a client's contribution
        to each slot independently.
        """
        if not client_params:
            return NamespacedAggregationResult()

        norm_weights = self._normalise_weights(weights, len(client_params))
        split_payloads = [self.split_params(params) for params in client_params]

        shared_payloads = [shared for shared, _ in split_payloads if shared]
        shared_weights = [
            weight for weight, (shared, _) in zip(norm_weights, split_payloads)
            if shared
        ]
        shared_params = (
            FedAvgAggregator().aggregate(shared_payloads, shared_weights)
            if shared_payloads else {}
        )

        expert_params: dict[int, dict[str, np.ndarray]] = {}
        slot_ids = sorted({
            slot_id
            for _, experts in split_payloads
            for slot_id in experts
        })

        for slot_id in slot_ids:
            payloads: list[dict[str, np.ndarray]] = []
            slot_weights: list[float] = []
            for idx, (_, experts) in enumerate(split_payloads):
                if slot_id not in experts:
                    continue
                weight = norm_weights[idx]
                if expert_weights is not None and idx < len(expert_weights):
                    weight *= float(expert_weights[idx].get(slot_id, 0.0))
                if weight <= 0.0:
                    continue
                payloads.append(experts[slot_id])
                slot_weights.append(weight)
            if payloads:
                expert_params[slot_id] = FedAvgAggregator().aggregate(payloads, slot_weights)

        return NamespacedAggregationResult(
            shared_params=shared_params,
            expert_params=expert_params,
        )

    @staticmethod
    def split_params(
        params: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
        """Split a flat payload into shared and per-slot expert params."""
        shared: dict[str, np.ndarray] = {}
        experts: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
        for key, value in params.items():
            match = _EXPERT_PREFIX_RE.match(key)
            if key.startswith("shared."):
                shared[key] = value.copy()
            elif match is not None:
                slot_id = int(match.group("slot"))
                experts[slot_id][key] = value.copy()
            else:
                shared[key] = value.copy()
        return shared, dict(experts)

    @staticmethod
    def merge_params(
        shared_params: dict[str, np.ndarray],
        expert_params: dict[int, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Merge shared and per-slot params back into a flat payload."""
        return NamespacedAggregationResult(
            shared_params=shared_params,
            expert_params=expert_params,
        ).to_flat_params()

    @staticmethod
    def _normalise_weights(
        weights: list[float] | None,
        n_clients: int,
    ) -> list[float]:
        if n_clients == 0:
            return []
        if weights is None:
            return [1.0 / n_clients] * n_clients
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        return [float(weight) / total for weight in weights]

    @property
    def name(self) -> str:
        return "NamespacedExpertFedAvg"
