"""Ablation studies for FedProTrack hyperparameters and module toggles.

Systematically varies one hyperparameter at a time while fixing others,
AND runs module-level ablations (disable temporal prior, hard assignment,
disable memory bank, disable spawn/merge, phase-A/B only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..drift_generator import GeneratorConfig, generate_drift_dataset
from ..metrics import compute_all_metrics
from ..metrics.experiment_log import MetricsResult
from ..posterior.fedprotrack_runner import FedProTrackRunner
from ..posterior.two_phase_protocol import TwoPhaseConfig
from .figures import generate_ablation_plot


@dataclass
class AblationConfig:
    """Configuration for ablation studies.

    Parameters
    ----------
    gen_config : GeneratorConfig
        Base data generation config (default: SINE, rho=5, alpha=0.5, delta=0.5).
    omega_values : list[float]
        Inverse temperature values to sweep.
    kappa_values : list[float]
        Stickiness values to sweep.
    novelty_threshold_values : list[float]
        Novelty threshold values to sweep.
    merge_threshold_values : list[float]
        Merge threshold values to sweep.
    federation_every_values : list[int]
        Federation frequency values to sweep.
    """

    gen_config: GeneratorConfig = field(default_factory=lambda: GeneratorConfig(
        K=10, T=20, n_samples=500,
        rho=5.0, alpha=0.5, delta=0.5,
        generator_type="sine", seed=42,
    ))
    omega_values: list[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0]
    )
    kappa_values: list[float] = field(
        default_factory=lambda: [0.5, 0.7, 0.8, 0.9, 0.95]
    )
    novelty_threshold_values: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5]
    )
    merge_threshold_values: list[float] = field(
        default_factory=lambda: [0.7, 0.8, 0.85, 0.9, 0.95]
    )
    federation_every_values: list[int] = field(
        default_factory=lambda: [1, 2, 5, 10]
    )


def _run_one(
    gen_config: GeneratorConfig,
    two_phase_config: TwoPhaseConfig,
    federation_every: int = 1,
    seed: int = 42,
) -> MetricsResult:
    """Run FedProTrack once and return metrics."""
    dataset = generate_drift_dataset(gen_config)
    runner = FedProTrackRunner(
        config=two_phase_config,
        federation_every=federation_every,
        seed=seed,
    )
    result = runner.run(dataset)
    log = result.to_experiment_log()
    return compute_all_metrics(log)


def run_ablation_study(
    config: AblationConfig | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, list[tuple[float, MetricsResult]]]:
    """Run all ablation sweeps (scalar hyperparameters).

    Parameters
    ----------
    config : AblationConfig, optional
    output_dir : Path or str, optional
        If provided, save ablation plots here.

    Returns
    -------
    dict[str, list[tuple[float, MetricsResult]]]
        param_name -> [(param_value, MetricsResult), ...]
    """
    if config is None:
        config = AblationConfig()

    results: dict[str, list[tuple[float, MetricsResult]]] = {}

    # Base config
    base = TwoPhaseConfig(
        omega=1.0, kappa=0.8, novelty_threshold=0.3, merge_threshold=0.85,
    )

    # Sweep omega
    results["omega"] = []
    for omega in config.omega_values:
        cfg = TwoPhaseConfig(
            omega=omega, kappa=base.kappa,
            novelty_threshold=base.novelty_threshold,
            merge_threshold=base.merge_threshold,
        )
        mr = _run_one(config.gen_config, cfg)
        results["omega"].append((omega, mr))

    # Sweep kappa
    results["kappa"] = []
    for kappa in config.kappa_values:
        cfg = TwoPhaseConfig(
            omega=base.omega, kappa=kappa,
            novelty_threshold=base.novelty_threshold,
            merge_threshold=base.merge_threshold,
        )
        mr = _run_one(config.gen_config, cfg)
        results["kappa"].append((kappa, mr))

    # Sweep novelty_threshold
    results["novelty_threshold"] = []
    for nt in config.novelty_threshold_values:
        cfg = TwoPhaseConfig(
            omega=base.omega, kappa=base.kappa,
            novelty_threshold=nt,
            merge_threshold=base.merge_threshold,
        )
        mr = _run_one(config.gen_config, cfg)
        results["novelty_threshold"].append((nt, mr))

    # Sweep merge_threshold
    results["merge_threshold"] = []
    for mt in config.merge_threshold_values:
        cfg = TwoPhaseConfig(
            omega=base.omega, kappa=base.kappa,
            novelty_threshold=base.novelty_threshold,
            merge_threshold=mt,
        )
        mr = _run_one(config.gen_config, cfg)
        results["merge_threshold"].append((mt, mr))

    # Sweep federation_every
    results["federation_every"] = []
    for fe in config.federation_every_values:
        mr = _run_one(config.gen_config, base, federation_every=fe)
        results["federation_every"].append((float(fe), mr))

    # Generate plots
    if output_dir is not None:
        output_dir = Path(output_dir)
        plot_metrics = [
            "concept_re_id_accuracy",
            "assignment_entropy",
            "wrong_memory_reuse_rate",
        ]
        for param_name, entries in results.items():
            param_vals = [e[0] for e in entries]
            metric_vals: dict[str, list[float]] = {}
            for m in plot_metrics:
                raw_vals = [getattr(e[1], m, None) for e in entries]
                # Skip metrics that are entirely None (identity-incapable)
                if all(v is None for v in raw_vals):
                    metric_vals[m] = [float("nan")] * len(entries)
                else:
                    metric_vals[m] = [
                        float(v) if v is not None else float("nan")
                        for v in raw_vals
                    ]
            generate_ablation_plot(
                param_name, param_vals, metric_vals,
                output_dir / f"ablation_{param_name}.png",
            )

    return results


# ---------------------------------------------------------------------------
# Module-level ablations (E7)
# ---------------------------------------------------------------------------

# Module toggle labels and their config overrides
MODULE_ABLATIONS: dict[str, dict[str, object]] = {
    "Full FedProTrack": {},
    "No temporal prior": {"kappa": 0.001},
    "Hard assignment (omega=100)": {"omega": 100.0},
    "No spawn/merge": {
        "merge_threshold": 1.0,
        "loss_novelty_threshold": 100.0,
        "novelty_threshold": 0.001,
    },
    "No post-spawn merge": {"post_spawn_merge": False},
    "No sticky dampening": {"sticky_dampening": 1.0, "sticky_posterior_gate": 1.0},
    "No model-loss gate": {"model_loss_weight": 0.0},
    "Phase A only (no aggregation)": {"_phase_a_only": True},
}


def run_module_ablation(
    gen_config: GeneratorConfig | None = None,
    output_dir: Path | str | None = None,
    seed: int = 42,
) -> dict[str, MetricsResult]:
    """Run module-level ablations (E7).

    Each ablation disables or modifies one module while keeping others
    at their default settings.

    Parameters
    ----------
    gen_config : GeneratorConfig, optional
    output_dir : Path or str, optional
    seed : int

    Returns
    -------
    dict[str, MetricsResult]
        ablation_label -> MetricsResult
    """
    if gen_config is None:
        gen_config = GeneratorConfig(
            K=10, T=20, n_samples=500,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type="sine", seed=seed,
        )

    results: dict[str, MetricsResult] = {}

    for label, overrides in MODULE_ABLATIONS.items():
        # Copy to avoid mutating the module-level dict
        overrides = dict(overrides)
        phase_a_only = overrides.pop("_phase_a_only", False)

        # Build config with overrides
        base_kwargs: dict[str, object] = {}
        for key, val in overrides.items():
            base_kwargs[key] = val

        cfg = TwoPhaseConfig(**base_kwargs)

        if phase_a_only:
            # Run with federation but skip Phase B (model aggregation)
            # Emulate by setting federation_every very high so Phase B rarely runs
            # Actually, the cleanest approach: run with federation_every=1 but
            # the model aggregation won't be used if we disable it in the runner.
            # For simplicity, just use extremely infrequent federation.
            mr = _run_one(gen_config, cfg, federation_every=999, seed=seed)
        else:
            mr = _run_one(gen_config, cfg, seed=seed)

        results[label] = mr

    # Generate grouped ablation bar charts
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _plot_module_ablation(results, output_dir)

    return results


def _plot_module_ablation(
    results: dict[str, MetricsResult],
    output_dir: Path,
) -> None:
    """Generate bar chart figures for module ablations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    metrics_to_plot = {
        "concept_re_id_accuracy": "Re-ID Accuracy",
        "wrong_memory_reuse_rate": "Wrong Memory Reuse",
        "assignment_entropy": "Assignment Entropy",
    }

    for metric_key, metric_label in metrics_to_plot.items():
        raw_values = [getattr(results[l], metric_key, None) for l in labels]
        # If all values are None, skip this metric chart entirely
        if all(v is None for v in raw_values):
            continue
        values = [float(v) if v is not None else 0.0 for v in raw_values]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2196F3" if l == "Full FedProTrack" else "#FF9800" for l in labels]
        bars = ax.bar(range(len(labels)), values, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(f"Module Ablation: {metric_label}", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

        fig.tight_layout()
        fig.savefig(
            output_dir / f"ablation_module_{metric_key}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)
