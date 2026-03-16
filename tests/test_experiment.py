"""Tests for experiment runner and baselines."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.experiment.runner import ExperimentConfig, ExperimentRunner
from fedprotrack.experiment.baselines import (
    run_local_only,
    run_fedavg_baseline,
    run_oracle_baseline,
)


@pytest.fixture
def small_config():
    return ExperimentConfig(
        generator_config=GeneratorConfig(
            K=3, T=5, n_samples=100, rho=2.0, alpha=0.5, delta=0.5,
            generator_type="sine", seed=42,
        ),
        detector_name="ADWIN",
        similarity_threshold=0.5,
    )


@pytest.fixture
def small_dataset(small_config):
    return generate_drift_dataset(small_config.generator_config)


class TestExperimentRunner:
    def test_run_returns_result(self, small_config, small_dataset):
        runner = ExperimentRunner(small_config)
        result = runner.run(dataset=small_dataset)
        assert result.accuracy_matrix.shape == (3, 5)
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_result_has_concept_matrix(self, small_config, small_dataset):
        runner = ExperimentRunner(small_config)
        result = runner.run(dataset=small_dataset)
        assert result.predicted_concept_matrix.shape == (3, 5)
        assert result.true_concept_matrix.shape == (3, 5)

    def test_result_summary(self, small_config, small_dataset):
        runner = ExperimentRunner(small_config)
        result = runner.run(dataset=small_dataset)
        summary = result.summary
        assert "mean_accuracy" in summary
        assert "concept_tracking_accuracy" in summary

    def test_result_save(self, small_config, small_dataset, tmp_path):
        runner = ExperimentRunner(small_config)
        result = runner.run(dataset=small_dataset)
        result.save(tmp_path / "result.json")
        assert (tmp_path / "result.json").exists()


class TestBaselines:
    def test_local_only(self, small_config, small_dataset):
        result = run_local_only(small_config, small_dataset)
        assert result.method_name == "LocalOnly"
        assert result.accuracy_matrix.shape == (3, 5)

    def test_fedavg_baseline(self, small_config, small_dataset):
        result = run_fedavg_baseline(small_config, small_dataset)
        assert result.method_name == "FedAvg"
        assert result.accuracy_matrix.shape == (3, 5)

    def test_oracle_baseline(self, small_config, small_dataset):
        result = run_oracle_baseline(small_config, small_dataset)
        assert result.method_name == "Oracle"
        assert result.accuracy_matrix.shape == (3, 5)

    def test_oracle_has_best_tracking(self, small_config, small_dataset):
        result = run_oracle_baseline(small_config, small_dataset)
        assert result.concept_tracking_accuracy == 1.0
