"""Federation server that coordinates the federated learning process."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aggregator import BaseAggregator


@dataclass
class RoundResult:
    """Result from one federation round (one time step)."""

    timestamp: int
    participating_clients: list[int]
    concept_ids: list[int]
    per_client_accuracy: list[float]
    mean_accuracy: float
    n_drift_events: int
    n_novel_concepts: int


class FederationServer:
    """Coordinates federated learning across clients.

    Manages the communication rounds: collects model updates from clients,
    aggregates them, and distributes the result back.

    Parameters
    ----------
    aggregator : BaseAggregator
        Strategy for combining model parameters.
    """

    def __init__(self, aggregator: BaseAggregator):
        self.aggregator = aggregator
        self._history: list[RoundResult] = []

    @property
    def history(self) -> list[RoundResult]:
        return list(self._history)

    def aggregate_round(
        self,
        client_params: list[dict[str, np.ndarray]],
        client_ids: list[int],
        concept_ids: list[int],
        accuracies: list[float],
        timestamp: int,
        weights: list[float] | None = None,
        drift_flags: list[bool] | None = None,
        novel_flags: list[bool] | None = None,
    ) -> dict[str, np.ndarray] | dict[int, dict[str, np.ndarray]]:
        """Execute one federation round.

        Parameters
        ----------
        client_params : list of dict
            Model parameters from each participating client.
        client_ids : list of int
            Client identifiers.
        concept_ids : list of int
            Predicted concept ID for each client at this time step.
        accuracies : list of float
            Per-client accuracy for this round.
        timestamp : int
            Current time step index.
        weights : list of float, optional
            Aggregation weights per client.
        drift_flags : list of bool, optional
            Whether each client detected drift this round.
        novel_flags : list of bool, optional
            Whether each client encountered a novel concept.

        Returns
        -------
        Aggregated parameters (format depends on the aggregator).
        """
        if drift_flags is None:
            drift_flags = [False] * len(client_ids)
        if novel_flags is None:
            novel_flags = [False] * len(client_ids)

        aggregated = self.aggregator.aggregate(
            client_params, weights=weights, concept_ids=concept_ids,
        )

        result = RoundResult(
            timestamp=timestamp,
            participating_clients=list(client_ids),
            concept_ids=list(concept_ids),
            per_client_accuracy=list(accuracies),
            mean_accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
            n_drift_events=sum(drift_flags),
            n_novel_concepts=sum(novel_flags),
        )
        self._history.append(result)

        return aggregated
