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

# Variant names used by real-data sweeps should map back to a canonical
# method family so identity-metric policy stays stable.
METHOD_ALIASES: dict[str, str] = {
    "FedProTrack-linear-split": "FedProTrack",
    "FedProTrack-feature-adapter": "FedProTrack",
    "IFCA-3": "IFCA",
    "IFCA-8": "IFCA",
    "FedEM-3": "FedEM",
    "FeSEM-3": "FeSEM",
    "FedRC-3": "FedRC",
}

FOCUSED_CIFAR_SWEEP_METHODS: tuple[str, ...] = (
    "FedProTrack-linear-split",
    "IFCA-3",
    "IFCA-8",
    "FedProto",
    "FedAvg",
    "LocalOnly",
    "Oracle",
)

FULL_CIFAR_SWEEP_METHODS: tuple[str, ...] = (
    "FedProTrack-linear-split",
    "FedAvg",
    "CompressedFedAvg",
    "FedProto",
    "pFedMe",
    "APFL",
    "FedEM-3",
    "IFCA-3",
    "IFCA-8",
    "CFL",
    "FeSEM-3",
    "FedRC-3",
    "FedCCFA",
    "FedDrift",
    "TrackedSummary",
    "Flash",
    "ATP",
    "FLUX",
    "FLUX-prior",
    "LocalOnly",
    "Oracle",
)

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


def canonical_method_name(method_name: str) -> str:
    """Return the canonical family name for *method_name*."""
    return METHOD_ALIASES.get(method_name, method_name)


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
    return canonical_method_name(method_name) in IDENTITY_CAPABLE_METHODS
