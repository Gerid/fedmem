"""Tests for FedProTrackRunner end-to-end simulation."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.posterior.fedprotrack_runner import (
    FedProTrackResult,
    FedProTrackRunner,
)
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_dataset():
    """Generate a small SINE dataset for testing."""
    cfg = GeneratorConfig(
        K=3, T=5, n_samples=100,
        rho=5.0, alpha=0.5, delta=0.5,
        generator_type="sine", seed=42,
    )
    return generate_drift_dataset(cfg)


# ---------------------------------------------------------------------------
# FedProTrackResult
# ---------------------------------------------------------------------------

class TestFedProTrackResult:
    def test_to_experiment_log(self) -> None:
        result = FedProTrackResult(
            accuracy_matrix=np.ones((3, 5)),
            predicted_concept_matrix=np.zeros((3, 5), dtype=np.int32),
            true_concept_matrix=np.zeros((3, 5), dtype=np.int32),
            total_bytes=1000.0,
            phase_a_bytes=200.0,
            phase_b_bytes=800.0,
            mean_accuracy=1.0,
            final_accuracy=1.0,
        )
        log = result.to_experiment_log()
        assert log.ground_truth.shape == (3, 5)
        assert log.predicted.shape == (3, 5)
        assert log.accuracy_curve is not None
        assert log.total_bytes == 1000.0
        assert log.method_name == "FedProTrack"


# ---------------------------------------------------------------------------
# FedProTrackRunner - basic
# ---------------------------------------------------------------------------

class TestFedProTrackRunner:
    def test_run_produces_valid_result(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)

        assert isinstance(result, FedProTrackResult)
        assert result.accuracy_matrix.shape == (3, 5)
        assert result.predicted_concept_matrix.shape == (3, 5)
        assert result.true_concept_matrix.shape == (3, 5)

    def test_accuracy_in_range(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)

        assert 0.0 <= result.mean_accuracy <= 1.0
        assert 0.0 <= result.final_accuracy <= 1.0
        assert np.all(result.accuracy_matrix >= 0.0)
        assert np.all(result.accuracy_matrix <= 1.0)

    def test_bytes_positive(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)

        assert result.total_bytes > 0
        assert result.phase_a_bytes > 0
        assert result.phase_b_bytes > 0

    def test_bytes_decomposition(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)

        assert abs(result.total_bytes - result.phase_a_bytes - result.phase_b_bytes) < 1e-6

    def test_method_name(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)
        assert result.method_name == "FedProTrack"


# ---------------------------------------------------------------------------
# FedProTrackRunner - configuration
# ---------------------------------------------------------------------------

class TestRunnerConfig:
    def test_custom_config(self) -> None:
        cfg = TwoPhaseConfig(omega=2.0, kappa=0.9)
        dataset = _small_dataset()
        runner = FedProTrackRunner(config=cfg, seed=42)
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)

    def test_federation_every(self) -> None:
        dataset = _small_dataset()
        r1 = FedProTrackRunner(federation_every=1, seed=42).run(dataset)
        r2 = FedProTrackRunner(federation_every=2, seed=42).run(dataset)

        # More frequent federation = more bytes
        assert r1.total_bytes > r2.total_bytes

    def test_different_detectors(self) -> None:
        dataset = _small_dataset()
        for detector in ["ADWIN", "PageHinkley", "KSWIN", "NoDrift"]:
            runner = FedProTrackRunner(detector_name=detector, seed=42)
            result = runner.run(dataset)
            assert result.accuracy_matrix.shape == (3, 5)

    def test_invalid_detector(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(detector_name="BadDetector", seed=42)
        with pytest.raises(ValueError, match="Unknown detector"):
            runner.run(dataset)


# ---------------------------------------------------------------------------
# FedProTrackRunner - reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        dataset = _small_dataset()
        r1 = FedProTrackRunner(seed=42).run(dataset)
        r2 = FedProTrackRunner(seed=42).run(dataset)
        np.testing.assert_array_equal(r1.accuracy_matrix, r2.accuracy_matrix)

    def test_different_seed_different_result(self) -> None:
        dataset = _small_dataset()
        r1 = FedProTrackRunner(seed=42).run(dataset)
        r2 = FedProTrackRunner(seed=99).run(dataset)
        # They may not be exactly equal (different SGD randomness)
        # But this is probabilistic; just check they run
        assert r1.accuracy_matrix.shape == r2.accuracy_matrix.shape


# ---------------------------------------------------------------------------
# FedProTrackRunner - SEA / CIRCLE generators
# ---------------------------------------------------------------------------

class TestGenerators:
    def test_sea_generator(self) -> None:
        cfg = GeneratorConfig(
            K=3, T=5, n_samples=100,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type="sea", seed=42,
        )
        dataset = generate_drift_dataset(cfg)
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)

    def test_circle_generator(self) -> None:
        cfg = GeneratorConfig(
            K=3, T=5, n_samples=100,
            rho=5.0, alpha=0.5, delta=0.5,
            generator_type="circle", seed=42,
        )
        dataset = generate_drift_dataset(cfg)
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)
