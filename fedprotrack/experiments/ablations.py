"""Ablation studies for FedProTrack hyperparameters.

Systematically varies one hyperparameter at a time while fixing others,
measuring the effect on all paper metrics.
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
    """Run all ablation sweeps.

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
                metric_vals[m] = [
                    float(getattr(e[1], m, 0.0) or 0.0) for e in entries
                ]
            generate_ablation_plot(
                param_name, param_vals, metric_vals,
                output_dir / f"ablation_{param_name}.png",
            )

    return results
