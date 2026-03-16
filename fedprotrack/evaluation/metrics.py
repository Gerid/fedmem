"""Evaluation metrics for federated concept drift experiments.

Covers three dimensions:
  1. Predictive performance (prequential accuracy, per-concept accuracy)
  2. Concept tracking quality (identification accuracy, detection delay)
  3. Knowledge transfer (backward transfer, forgetting measure)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class StreamingAccuracy:
    """Online accuracy tracker with optional fading factor.

    Parameters
    ----------
    fading_factor : float
        Exponential decay for older observations. 1.0 = no decay.
    """

    def __init__(self, fading_factor: float = 1.0):
        self.fading_factor = fading_factor
        self._correct: float = 0.0
        self._total: float = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Update with a batch and return current accuracy.

        Parameters
        ----------
        y_true : np.ndarray of shape (n,)
        y_pred : np.ndarray of shape (n,)

        Returns
        -------
        accuracy : float
            Current accumulated accuracy.
        """
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        self._correct = self._correct * self.fading_factor + correct
        self._total = self._total * self.fading_factor + total
        return self.accuracy

    @property
    def accuracy(self) -> float:
        if self._total == 0:
            return 0.0
        return float(self._correct / self._total)

    def reset(self) -> None:
        self._correct = 0.0
        self._total = 0.0


@dataclass
class ConceptTrackingMetrics:
    """Aggregated metrics for concept identification quality."""

    n_true_drifts: int = 0
    n_detected_drifts: int = 0
    n_correct_identifications: int = 0
    n_identification_attempts: int = 0
    detection_delays: list[int] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        """Fraction of true drifts that were detected."""
        if self.n_true_drifts == 0:
            return 1.0
        return self.n_detected_drifts / self.n_true_drifts

    @property
    def false_alarm_rate(self) -> float:
        """Fraction of detected drifts that were false alarms."""
        if self.n_detected_drifts == 0:
            return 0.0
        false_alarms = max(0, self.n_detected_drifts - self.n_true_drifts)
        return false_alarms / self.n_detected_drifts

    @property
    def identification_accuracy(self) -> float:
        """Fraction of identification attempts that were correct."""
        if self.n_identification_attempts == 0:
            return 0.0
        return self.n_correct_identifications / self.n_identification_attempts

    @property
    def mean_detection_delay(self) -> float:
        """Average delay (in time steps) between true drift and detection."""
        if not self.detection_delays:
            return 0.0
        return float(np.mean(self.detection_delays))


def compute_prequential_accuracy(
    predictions: list[np.ndarray],
    true_labels: list[np.ndarray],
) -> np.ndarray:
    """Compute per-step prequential accuracy.

    Parameters
    ----------
    predictions : list of np.ndarray
        Predictions at each time step.
    true_labels : list of np.ndarray
        True labels at each time step.

    Returns
    -------
    accuracies : np.ndarray of shape (T,)
        Accuracy at each time step.
    """
    accs = []
    for y_pred, y_true in zip(predictions, true_labels):
        if len(y_true) == 0:
            accs.append(0.0)
        else:
            accs.append(float(np.mean(y_pred == y_true)))
    return np.array(accs, dtype=np.float64)


def compute_concept_tracking_accuracy(
    predicted_concepts: list[int],
    true_concepts: list[int],
) -> float:
    """Compute concept identification accuracy (up to permutation).

    Uses a greedy matching strategy: maps each predicted concept ID to the
    true concept ID it most frequently co-occurs with.

    Parameters
    ----------
    predicted_concepts : list of int
        Predicted concept IDs over time.
    true_concepts : list of int
        Ground-truth concept IDs over time.

    Returns
    -------
    accuracy : float
        Fraction of time steps where predicted concept matches true
        concept (after optimal remapping).
    """
    if not predicted_concepts:
        return 0.0

    pred = np.array(predicted_concepts, dtype=np.int32)
    true = np.array(true_concepts, dtype=np.int32)

    # Build co-occurrence matrix
    unique_pred = np.unique(pred)
    unique_true = np.unique(true)
    cooccur = np.zeros((len(unique_pred), len(unique_true)), dtype=np.int32)
    pred_idx = {v: i for i, v in enumerate(unique_pred)}
    true_idx = {v: i for i, v in enumerate(unique_true)}

    for p, t in zip(pred, true):
        cooccur[pred_idx[p], true_idx[t]] += 1

    # Greedy matching: assign each predicted concept to its best true concept
    mapping: dict[int, int] = {}
    used_true: set[int] = set()
    # Sort by max co-occurrence descending for greedy assignment
    for _ in range(min(len(unique_pred), len(unique_true))):
        best_val = -1
        best_pi = -1
        best_ti = -1
        for pi in range(len(unique_pred)):
            if unique_pred[pi] in mapping:
                continue
            for ti in range(len(unique_true)):
                if unique_true[ti] in used_true:
                    continue
                if cooccur[pi, ti] > best_val:
                    best_val = cooccur[pi, ti]
                    best_pi = pi
                    best_ti = ti
        if best_pi >= 0:
            mapping[unique_pred[best_pi]] = unique_true[best_ti]
            used_true.add(unique_true[best_ti])

    # Compute accuracy under mapping
    correct = sum(
        1 for p, t in zip(pred, true)
        if mapping.get(p, -1) == t
    )
    return correct / len(pred)


def compute_forgetting_measure(
    per_concept_accuracies: dict[int, list[float]],
) -> float:
    """Compute average forgetting across concepts.

    Forgetting for a concept = max past accuracy - final accuracy.
    Averaged across all concepts that were seen at least twice.

    Parameters
    ----------
    per_concept_accuracies : dict
        Mapping from concept_id to list of accuracy values (one per
        encounter with that concept).

    Returns
    -------
    forgetting : float
        Average forgetting. Lower is better. Negative means improvement.
    """
    forgettings = []
    for cid, accs in per_concept_accuracies.items():
        if len(accs) >= 2:
            forgettings.append(max(accs[:-1]) - accs[-1])
    if not forgettings:
        return 0.0
    return float(np.mean(forgettings))


def compute_backward_transfer(
    per_concept_accuracies: dict[int, list[float]],
) -> float:
    """Compute backward transfer (BWT).

    BWT measures how learning new concepts affects performance on old ones.
    Positive BWT means revisiting a concept improved performance.

    Parameters
    ----------
    per_concept_accuracies : dict
        Mapping from concept_id to list of accuracy values.

    Returns
    -------
    bwt : float
        Average backward transfer. Positive = beneficial.
    """
    transfers = []
    for cid, accs in per_concept_accuracies.items():
        if len(accs) >= 2:
            transfers.append(accs[-1] - accs[0])
    if not transfers:
        return 0.0
    return float(np.mean(transfers))
