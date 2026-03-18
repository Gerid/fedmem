from __future__ import annotations

"""Method capability registry for Phase 3 experiments.

Defines which methods support identity-level concept inference and
therefore produce meaningful identity metrics (concept_re_id_accuracy,
assignment_entropy, assignment_switch_rate, avg_clients_per_concept,
singleton_group_ratio, memory_reuse_rate, routing_consistency,
wrong_memory_reuse_rate).

Methods that do **not** perform identity inference (e.g. FedAvg, FedProto,
Flash, CompressedFedAvg, LocalOnly) should have their identity metrics
reported as ``None`` / ``"--"`` / ``NaN`` rather than zero.
"""

from dataclasses import dataclass

# --- Identity-capable methods ------------------------------------------------
# Only methods that actively infer *which* concept each client is on at each
# time step produce valid identity metrics.

IDENTITY_CAPABLE_METHODS: frozenset[str] = frozenset({
    "FedProTrack",
    "Oracle",
    "FedEM",
    "FedCCFA",
    "TrackedSummary",
    "FedDrift",
    "IFCA",
    "FeSEM",
    "FedRC",
})

# Methods whose ``predicted_concept_matrix`` is always all-zeros (or
# otherwise meaningless for identity-tracking purposes).
NON_IDENTITY_METHODS: frozenset[str] = frozenset({
    "FedAvg",
    "FedProto",
    "Flash",
    "CompressedFedAvg",
    "LocalOnly",
    "pFedMe",
    "APFL",
    "ATP",
    "CFL",
    "FLUX",
    "FLUX-prior",
})

# The three identity-specific metric field names on MetricsResult.
IDENTITY_METRIC_FIELDS: tuple[str, ...] = (
    "concept_re_id_accuracy",
    "assignment_entropy",
    "assignment_switch_rate",
    "avg_clients_per_concept",
    "singleton_group_ratio",
    "memory_reuse_rate",
    "routing_consistency",
    "wrong_memory_reuse_rate",
)


def identity_metrics_valid(method_name: str) -> bool:
    """Return True if *method_name* is expected to produce valid identity metrics.

    Parameters
    ----------
    method_name : str
        Human-readable method name (e.g. ``"FedAvg"``).

    Returns
    -------
    bool
    """
    return method_name in IDENTITY_CAPABLE_METHODS
