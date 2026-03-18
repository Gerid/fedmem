from __future__ import annotations

import numpy as np

from .budget_metrics import (
    budget_normalized_score,
    compute_accuracy_auc,
    fedavg_total_bytes,
)
from .concept_metrics import (
    assignment_entropy,
    assignment_switch_rate,
    avg_clients_per_concept,
    concept_re_id_accuracy,
    memory_reuse_rate,
    routing_consistency,
    singleton_group_ratio,
    wrong_memory_reuse_rate,
)
from .drift_metrics import find_drift_points, worst_window_dip_recovery
from .experiment_log import ExperimentLog, MetricsResult
from .hungarian import align_predictions
from .phase_diagram import PhaseDiagramData, build_phase_diagram, load_results_from_dir
from .visualization import plot_budget_frontier, plot_metric_comparison, plot_phase_diagram


def compute_all_metrics(
    log: ExperimentLog,
    identity_capable: bool = True,
) -> MetricsResult:
    """Compute all available metrics from an ExperimentLog.

    Parameters
    ----------
    log : ExperimentLog
        Contains ground truth, predictions, and optional accuracy/budget data.
    identity_capable : bool
        If False, identity metrics (concept_re_id_accuracy,
        assignment_entropy, wrong_memory_reuse_rate, per_client_re_id,
        per_timestep_re_id) are set to ``None`` instead of being computed.
        This should be set to False for methods that do not perform
        concept identity inference (e.g. FedAvg, FedProto, Flash).

    Returns
    -------
    MetricsResult
        All computed metrics; optional fields are None when input is missing
        or when the method does not support that metric.
    """
    # Concept identity metrics -- only compute when the method supports them
    acc: float | None = None
    ent: float | None = None
    switch_rate: float | None = None
    clients_per_concept: float | None = None
    singleton_ratio: float | None = None
    reuse_rate: float | None = None
    consistency: float | None = None
    wmrr: float | None = None
    per_client: np.ndarray | None = None
    per_ts: np.ndarray | None = None

    if identity_capable:
        acc, per_client, per_ts = concept_re_id_accuracy(
            log.ground_truth, log.predicted,
        )
        n_concepts = int(log.ground_truth.max()) + 1
        ent = assignment_entropy(log.soft_assignments, log.predicted, n_concepts)
        switch_rate = assignment_switch_rate(log.predicted)
        clients_per_concept = avg_clients_per_concept(log.predicted)
        singleton_ratio = singleton_group_ratio(log.predicted)
        reuse_rate = memory_reuse_rate(log.ground_truth, log.predicted)
        consistency = routing_consistency(log.soft_assignments, log.predicted)
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

    # Final accuracy and AUC (optional)
    final_acc: float | None = None
    acc_auc: float | None = None
    if log.accuracy_curve is not None:
        final_acc = float(log.accuracy_curve[:, -1].mean())
        acc_auc = float(compute_accuracy_auc(log.accuracy_curve))

    return MetricsResult(
        concept_re_id_accuracy=acc,
        assignment_entropy=ent,
        assignment_switch_rate=switch_rate,
        avg_clients_per_concept=clients_per_concept,
        singleton_group_ratio=singleton_ratio,
        memory_reuse_rate=reuse_rate,
        routing_consistency=consistency,
        wrong_memory_reuse_rate=wmrr,
        worst_window_dip=dip,
        worst_window_recovery=recovery,
        budget_normalized_score=bns,
        per_client_re_id=per_client,
        per_timestep_re_id=per_ts,
        final_accuracy=final_acc,
        accuracy_auc=acc_auc,
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
    "assignment_switch_rate",
    "avg_clients_per_concept",
    "singleton_group_ratio",
    "memory_reuse_rate",
    "routing_consistency",
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
