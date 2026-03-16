"""Scalability experiments: vary K (clients) and T (timesteps).

Measures how FedProTrack and baselines scale with increasing number of
clients and time horizon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..drift_generator import GeneratorConfig, generate_drift_dataset
from ..experiment.baselines import run_fedavg_baseline, run_local_only
from ..experiment.runner import ExperimentConfig
from ..metrics import compute_all_metrics
from ..metrics.experiment_log import ExperimentLog, MetricsResult
from ..posterior.fedprotrack_runner import FedProTrackRunner
from ..posterior.two_phase_protocol import TwoPhaseConfig
from .figures import generate_scalability_plot
from .method_registry import identity_metrics_valid


@dataclass
class ScalabilityResult:
    """Result for one scalability data point.

    Parameters
    ----------
    method_name : str
    param_value : int
        Value of K or T.
    metrics : MetricsResult
    wall_time_s : float
        Wall-clock time in seconds.
    total_bytes : float
    """

    method_name: str
    param_value: int
    metrics: MetricsResult
    wall_time_s: float
    total_bytes: float


def run_scalability_K(
    K_values: list[int] | None = None,
    T: int = 20,
    n_samples: int = 500,
    seed: int = 42,
    output_dir: Path | str | None = None,
) -> list[ScalabilityResult]:
    """Vary number of clients K.

    Parameters
    ----------
    K_values : list[int], optional
        Default [5, 10, 20, 50].
    T : int
    n_samples : int
    seed : int
    output_dir : Path or str, optional

    Returns
    -------
    list[ScalabilityResult]
    """
    if K_values is None:
        K_values = [5, 10, 20, 50]

    results: list[ScalabilityResult] = []

    for K in K_values:
        gen_cfg = GeneratorConfig(
            K=K, T=T, n_samples=n_samples,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type="sine", seed=seed,
        )
        dataset = generate_drift_dataset(gen_cfg)

        # FedProTrack
        t0 = time.time()
        runner = FedProTrackRunner(seed=seed)
        fpt_result = runner.run(dataset)
        fpt_time = time.time() - t0
        fpt_log = fpt_result.to_experiment_log()
        fpt_metrics = compute_all_metrics(
            fpt_log, identity_capable=identity_metrics_valid("FedProTrack"),
        )
        results.append(ScalabilityResult(
            "FedProTrack", K, fpt_metrics, fpt_time, fpt_result.total_bytes,
        ))

        # FedAvg
        exp_cfg = ExperimentConfig(generator_config=gen_cfg)
        t0 = time.time()
        fa_result = run_fedavg_baseline(exp_cfg, dataset=dataset)
        fa_time = time.time() - t0
        fa_log = ExperimentLog(
            ground_truth=dataset.concept_matrix,
            predicted=fa_result.predicted_concept_matrix,
            accuracy_curve=fa_result.accuracy_matrix,
            total_bytes=None,
            method_name="FedAvg",
        )
        fa_metrics = compute_all_metrics(
            fa_log, identity_capable=identity_metrics_valid("FedAvg"),
        )
        results.append(ScalabilityResult(
            "FedAvg", K, fa_metrics, fa_time, 0.0,
        ))

    # Plot
    if output_dir is not None:
        output_dir = Path(output_dir)
        # Mean accuracy plot -- use final_accuracy instead of identity metric
        method_accs: dict[str, list[float]] = {}
        for r in results:
            if r.method_name not in method_accs:
                method_accs[r.method_name] = []
            val = r.metrics.concept_re_id_accuracy
            method_accs[r.method_name].append(val if val is not None else float("nan"))

        fpt_accs = [r.metrics.concept_re_id_accuracy for r in results
                     if r.method_name == "FedProTrack" and r.metrics.concept_re_id_accuracy is not None]
        if fpt_accs:
            generate_scalability_plot(
                "K (clients)", K_values,
                {"FedProTrack": fpt_accs},
                output_dir / "scalability_K.png",
                y_label="Re-ID Accuracy",
            )

        # Bytes plot
        fpt_bytes = [r.total_bytes for r in results if r.method_name == "FedProTrack"]
        if fpt_bytes:
            generate_scalability_plot(
                "K (clients)", K_values,
                {"FedProTrack": fpt_bytes},
                output_dir / "scalability_K_bytes.png",
                y_label="Total Bytes",
            )

    return results


def run_scalability_T(
    T_values: list[int] | None = None,
    K: int = 10,
    n_samples: int = 500,
    seed: int = 42,
    output_dir: Path | str | None = None,
) -> list[ScalabilityResult]:
    """Vary number of timesteps T.

    Parameters
    ----------
    T_values : list[int], optional
        Default [10, 20, 50, 100].
    K : int
    n_samples : int
    seed : int
    output_dir : Path or str, optional

    Returns
    -------
    list[ScalabilityResult]
    """
    if T_values is None:
        T_values = [10, 20, 50, 100]

    results: list[ScalabilityResult] = []

    for T in T_values:
        gen_cfg = GeneratorConfig(
            K=K, T=T, n_samples=n_samples,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type="sine", seed=seed,
        )
        dataset = generate_drift_dataset(gen_cfg)

        # FedProTrack
        t0 = time.time()
        runner = FedProTrackRunner(seed=seed)
        fpt_result = runner.run(dataset)
        fpt_time = time.time() - t0
        fpt_log = fpt_result.to_experiment_log()
        fpt_metrics = compute_all_metrics(
            fpt_log, identity_capable=identity_metrics_valid("FedProTrack"),
        )
        results.append(ScalabilityResult(
            "FedProTrack", T, fpt_metrics, fpt_time, fpt_result.total_bytes,
        ))

    # Plot
    if output_dir is not None:
        output_dir = Path(output_dir)
        fpt_accs = [r.metrics.concept_re_id_accuracy for r in results
                     if r.metrics.concept_re_id_accuracy is not None]
        generate_scalability_plot(
            "T (timesteps)", T_values,
            {"FedProTrack": fpt_accs},
            output_dir / "scalability_T.png",
            y_label="Re-ID Accuracy",
        )

    return results
