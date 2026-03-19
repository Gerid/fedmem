from __future__ import annotations

from .two_phase_protocol import TwoPhaseConfig


FEDPROTRACK_VARIANTS: tuple[str, ...] = (
    "legacy",
    "plan_c_linear",
    "plan_c_feature_adapter",
)


def make_legacy_config(**overrides) -> TwoPhaseConfig:
    """Return the original main-branch FedProTrack configuration."""
    params = {
        "omega": 2.0,
        "kappa": 0.6,
        "novelty_threshold": 0.3,
        "loss_novelty_threshold": 0.05,
        "sticky_dampening": 1.0,
        "sticky_posterior_gate": 0.3,
        "model_loss_weight": 0.0,
        "post_spawn_merge": True,
        "merge_threshold": 0.98,
        "merge_min_support": 1,
        "min_count": 5.0,
        "max_concepts": 20,
        "merge_every": 2,
        "shrink_every": 5,
        "key_mode": "legacy_fingerprint",
        "key_ema_decay": 0.0,
        "key_style_weight": 0.25,
        "key_semantic_weight": 0.30,
        "key_prototype_weight": 0.45,
        "global_shared_aggregation": False,
        "entropy_freeze_threshold": None,
        "adaptive_addressing": False,
        "addressing_min_round_interval": 1,
        "addressing_drift_threshold": 0.0,
    }
    params.update(overrides)
    return TwoPhaseConfig(**params)


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


def make_variant_bundle(
    variant: str,
    *,
    config_overrides: dict[str, object] | None = None,
    runner_overrides: dict[str, object] | None = None,
) -> tuple[str, TwoPhaseConfig, dict[str, object]]:
    """Return method label, protocol config, and runner kwargs for a variant."""
    cfg_overrides = dict(config_overrides or {})
    run_overrides = dict(runner_overrides or {})

    if variant == "legacy":
        config = make_legacy_config(**cfg_overrides)
        runner_kwargs: dict[str, object] = {
            "soft_aggregation": False,
            "blend_alpha": 0.5,
            "skip_last_federation_round": False,
        }
        runner_kwargs.update(run_overrides)
        return "FedProTrack", config, runner_kwargs

    if variant == "plan_c_linear":
        config = make_plan_c_config(**cfg_overrides)
        runner_kwargs = {
            "soft_aggregation": True,
            "blend_alpha": 0.0,
            "skip_last_federation_round": True,
        }
        runner_kwargs.update(run_overrides)
        return "FedProTrack-PlanC", config, runner_kwargs

    if variant == "plan_c_feature_adapter":
        config = make_plan_c_config(**cfg_overrides)
        runner_kwargs = {
            "soft_aggregation": True,
            "blend_alpha": 0.0,
            "skip_last_federation_round": True,
            "model_type": "feature_adapter",
            "hidden_dim": 64,
            "adapter_dim": 16,
        }
        runner_kwargs.update(run_overrides)
        return "FedProTrack-Adapter", config, runner_kwargs

    raise ValueError(
        f"Unknown FedProTrack variant: {variant}. "
        f"Choose from {list(FEDPROTRACK_VARIANTS)}"
    )
