from __future__ import annotations

from .shrinkage import (
    ShrinkageEstimator,
    compute_effective_rank,
    compute_shrinkage_lambda,
)

__all__ = [
    "ShrinkageEstimator",
    "compute_effective_rank",
    "compute_shrinkage_lambda",
]
