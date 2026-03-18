from __future__ import annotations

from .two_phase_protocol import TwoPhaseConfig


def make_plan_c_config(**overrides) -> TwoPhaseConfig:
    """Return the repo-aligned Plan C preset with optional overrides."""
    params = {
        "omega": 2.0,
        "kappa": 0.7,
        "novelty_threshold": 0.25,
        "loss_novelty_threshold": 0.15,
        "sticky_dampening": 1.5,
        "sticky_posterior_gate": 0.35,
        "merge_threshold": 0.85,
        "merge_min_support": 2,
        "min_count": 5.0,
        "max_concepts": 6,
        "merge_every": 2,
        "shrink_every": 8,
        "key_mode": "multi_scale",
        "key_ema_decay": 0.6,
        "entropy_freeze_threshold": 0.75,
        "adaptive_addressing": True,
        "addressing_min_round_interval": 2,
        "addressing_drift_threshold": 0.02,
    }
    params.update(overrides)
    return TwoPhaseConfig(**params)
