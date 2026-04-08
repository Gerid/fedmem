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

# --- Identity-capable methods ------------------------------------------------
# Only methods that actively infer *which* concept each client is on at each
# time step produce valid identity metrics.

IDENTITY_CAPABLE_METHODS: frozenset[str] = frozenset({
    "FedProTrack",
    "Oracle",
    "FedEM",
    "FedCCFA",
    "FedCCFA-Impl",
    "TrackedSummary",
    "FedDrift",
    "IFCA",
    "FeSEM",
    "FedRC",
    "HCFL",
    "FedGWC",
})

# Methods whose ``predicted_concept_matrix`` is always all-zeros (or
# otherwise meaningless for identity-tracking purposes).
NON_IDENTITY_METHODS: frozenset[str] = frozenset({
    "FedAvg",
    "FedProto",
    "FedProx",
    "Flash",
    "CompressedFedAvg",
    "LocalOnly",
    "pFedMe",
    "APFL",
    "ATP",
    "CFL",
    "FLUX",
    "FLUX-prior",
    "Ditto",
    "SCAFFOLD",
    "Adaptive-FedAvg",
})

# Methods that are currently aliases of another implementation in this repo.
# These should be de-duplicated in comparison tables when a distinct-method
# view is desired.
METHOD_ALIASES: dict[str, str] = {
    "FeSEM": "IFCA",
}

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
    return canonical_method_name(method_name) in IDENTITY_CAPABLE_METHODS


def canonical_method_name(method_name: str) -> str:
    """Return the canonical implementation name for a method label."""
    if method_name.startswith("FedProTrack-"):
        return "FedProTrack"
    return METHOD_ALIASES.get(method_name, method_name)


def dedupe_method_names(method_names: list[str]) -> list[str]:
    """Drop duplicate alias methods while preserving canonical methods."""
    requested_canonicals = {
        canonical_method_name(method_name) for method_name in method_names
    }
    deduped: list[str] = []
    seen: set[str] = set()
    for method_name in method_names:
        canonical = canonical_method_name(method_name)
        if (
            method_name in METHOD_ALIASES
            and canonical in requested_canonicals
            and canonical != method_name
        ):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(method_name)
    return deduped
