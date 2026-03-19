from __future__ import annotations

from .method_registry import (
    METHOD_ALIASES,
    IDENTITY_CAPABLE_METHODS,
    IDENTITY_METRIC_FIELDS,
    NON_IDENTITY_METHODS,
    canonical_method_name,
    dedupe_method_names,
    identity_metrics_valid,
)

__all__ = [
    "METHOD_ALIASES",
    "IDENTITY_CAPABLE_METHODS",
    "IDENTITY_METRIC_FIELDS",
    "NON_IDENTITY_METHODS",
    "canonical_method_name",
    "dedupe_method_names",
    "identity_metrics_valid",
]
