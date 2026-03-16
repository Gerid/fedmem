"""Federated aggregation strategies.

Provides standard FedAvg and the concept-aware variant used by FedProTrack.
The concept-aware aggregator only averages models that share the same predicted
concept identity, avoiding destructive interference between heterogeneous
distributions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


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
