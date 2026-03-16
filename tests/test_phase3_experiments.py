"""Tests for Phase 3 experiment infrastructure."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.metrics.experiment_log import MetricsResult
from fedprotrack.baselines.runners import (
    MethodResult,
    run_fedproto_full,
    run_tracked_summary_full,
)
from fedprotrack.experiments.tables import generate_main_table, generate_per_axis_table
from fedprotrack.experiments.figures import (
    generate_accuracy_curves,
    generate_ablation_plot,
    generate_scalability_plot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_dataset():
    cfg = GeneratorConfig(
        K=3, T=5, n_samples=100,
        rho=5.0, alpha=0.5, delta=0.5,
        generator_type="sine", seed=42,
    )
    return generate_drift_dataset(cfg)


def _dummy_metrics(acc: float = 0.5) -> MetricsResult:
    return MetricsResult(
        concept_re_id_accuracy=acc,
        assignment_entropy=0.3,
        wrong_memory_reuse_rate=1.0 - acc,
        worst_window_dip=0.1,
        worst_window_recovery=2,
        budget_normalized_score=0.01,
        per_client_re_id=np.array([acc]),
        per_timestep_re_id=np.array([acc]),
    )


# ---------------------------------------------------------------------------
# MethodResult
# ---------------------------------------------------------------------------

class TestMethodResult:
    def test_to_experiment_log(self) -> None:
        mr = MethodResult(
            method_name="Test",
            accuracy_matrix=np.ones((3, 5)),
            predicted_concept_matrix=np.zeros((3, 5), dtype=np.int32),
            total_bytes=100.0,
        )
        gt = np.zeros((3, 5), dtype=np.int32)
        log = mr.to_experiment_log(gt)
        assert log.method_name == "Test"
        assert log.ground_truth.shape == (3, 5)
        assert log.total_bytes == 100.0


# ---------------------------------------------------------------------------
# Baseline full runners
# ---------------------------------------------------------------------------

class TestBaselineRunners:
    def test_fedproto_full(self) -> None:
        dataset = _small_dataset()
        result = run_fedproto_full(dataset)
        assert result.method_name == "FedProto"
        assert result.accuracy_matrix.shape == (3, 5)
        assert result.predicted_concept_matrix.shape == (3, 5)
        assert np.all(result.accuracy_matrix >= 0)
        assert np.all(result.accuracy_matrix <= 1)

    def test_tracked_summary_full(self) -> None:
        dataset = _small_dataset()
        result = run_tracked_summary_full(dataset)
        assert result.method_name == "TrackedSummary"
        assert result.accuracy_matrix.shape == (3, 5)
        assert result.predicted_concept_matrix.shape == (3, 5)

    def test_fedproto_federation_every(self) -> None:
        dataset = _small_dataset()
        r1 = run_fedproto_full(dataset, federation_every=1)
        r2 = run_fedproto_full(dataset, federation_every=5)
        # More frequent federation = more bytes
        assert r1.total_bytes >= r2.total_bytes


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

class TestTables:
    def test_main_table_generates_latex(self) -> None:
        all_results = {
            "MethodA": [_dummy_metrics(0.8), _dummy_metrics(0.7)],
            "MethodB": [_dummy_metrics(0.6), _dummy_metrics(0.5)],
        }
        latex = generate_main_table(all_results)
        assert r"\begin{table}" in latex
        assert "MethodA" in latex
        assert "MethodB" in latex
        assert r"\textbf" in latex  # best method should be bolded

    def test_main_table_writes_file(self, tmp_path: Path) -> None:
        all_results = {"M": [_dummy_metrics()]}
        path = tmp_path / "table.tex"
        generate_main_table(all_results, output_path=path)
        assert path.exists()
        content = path.read_text()
        assert r"\begin{table}" in content

    def test_per_axis_table(self) -> None:
        axis_results = {
            2.0: {"M": [_dummy_metrics(0.5)]},
            5.0: {"M": [_dummy_metrics(0.7)]},
        }
        latex = generate_per_axis_table(axis_results, "rho")
        assert "rho" in latex
        assert r"\begin{table}" in latex


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

class TestFigures:
    def test_accuracy_curves(self, tmp_path: Path) -> None:
        setting = {
            "MethodA": np.random.rand(3, 5),
            "MethodB": np.random.rand(3, 5),
        }
        path = tmp_path / "acc_curves.png"
        result = generate_accuracy_curves(setting, path)
        assert result.exists()

    def test_ablation_plot(self, tmp_path: Path) -> None:
        path = tmp_path / "ablation.png"
        generate_ablation_plot(
            "omega", [0.5, 1.0, 2.0],
            {"acc": [0.5, 0.7, 0.6]},
            path,
        )
        assert path.exists()

    def test_scalability_plot(self, tmp_path: Path) -> None:
        path = tmp_path / "scale.png"
        generate_scalability_plot(
            "K (clients)", [5, 10, 20],
            {"FedProTrack": [0.7, 0.6, 0.5]},
            path,
        )
        assert path.exists()
