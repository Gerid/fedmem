from __future__ import annotations

import numpy as np

from .budget_metrics import (
    budget_normalized_score,
    compute_accuracy_auc,
    fedavg_total_bytes,
)
from .concept_metrics import (
    assignment_entropy,
    concept_re_id_accuracy,
    wrong_memory_reuse_rate,
)
from .drift_metrics import find_drift_points, worst_window_dip_recovery
from .experiment_log import ExperimentLog, MetricsResult
from .hungarian import align_predictions
from .phase_diagram import PhaseDiagramData, build_phase_diagram, load_results_from_dir
from .visualization import plot_budget_frontier, plot_metric_comparison, plot_phase_diagram


def compute_all_metrics(log: ExperimentLog) -> MetricsResult:
    """Compute all available metrics from an ExperimentLog.

    Parameters
    ----------
    log : ExperimentLog
        Contains ground truth, predictions, and optional accuracy/budget data.

    Returns
    -------
    MetricsResult
        All computed metrics; optional fields are None when input is missing.
    """
    # Concept identity metrics
    acc, per_client, per_ts = concept_re_id_accuracy(log.ground_truth, log.predicted)

    n_concepts = int(log.ground_truth.max()) + 1
    ent = assignment_entropy(log.soft_assignments, log.predicted, n_concepts)

    wmrr = wrong_memory_reuse_rate(log.ground_truth, log.predicted)

    # Drift window metrics (optional)
    dip: float | None = None
    recovery: int | None = None
    if log.accuracy_curve is not None:
        dip, recovery = worst_window_dip_recovery(log.accuracy_curve, log.ground_truth)

    # Budget-normalized score (optional)
    bns: float | None = None
    if log.accuracy_curve is not None and log.total_bytes is not None:
        bns = budget_normalized_score(log.accuracy_curve, log.total_bytes)

    return MetricsResult(
        concept_re_id_accuracy=acc,
        assignment_entropy=ent,
        wrong_memory_reuse_rate=wmrr,
        worst_window_dip=dip,
        worst_window_recovery=recovery,
        budget_normalized_score=bns,
        per_client_re_id=per_client,
        per_timestep_re_id=per_ts,
    )


__all__ = [
    # Core orchestrator
    "compute_all_metrics",
    # Data structures
    "ExperimentLog",
    "MetricsResult",
    # Hungarian alignment
    "align_predictions",
    # Concept metrics
    "concept_re_id_accuracy",
    "assignment_entropy",
    "wrong_memory_reuse_rate",
    # Drift metrics
    "find_drift_points",
    "worst_window_dip_recovery",
    # Budget metrics
    "budget_normalized_score",
    "compute_accuracy_auc",
    "fedavg_total_bytes",
    # Phase diagram
    "PhaseDiagramData",
    "build_phase_diagram",
    "load_results_from_dir",
    # Visualization
    "plot_phase_diagram",
    "plot_metric_comparison",
    "plot_budget_frontier",
]
