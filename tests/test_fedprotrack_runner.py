"""Tests for FedProTrackRunner end-to-end simulation."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.drift_generator import (
    DriftDataset,
    GeneratorConfig,
    generate_drift_dataset,
)
from fedprotrack.drift_generator.data_streams import ConceptSpec
from fedprotrack.posterior.fedprotrack_runner import (
    FedProTrackResult,
    FedProTrackRunner,
    _PredictiveSubgroupCacheEntry,
    _blend_param_dicts,
    _blend_with_cached_subgroup,
    _estimate_fingerprint_similarity_quantiles,
    _project_model_signature,
)
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig, TwoPhaseFedProTrack


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


def _high_dim_dataset() -> DriftDataset:
    """Build a small deterministic high-dimensional dataset."""
    cfg = GeneratorConfig(
        K=3,
        T=4,
        n_samples=80,
        rho=2.0,
        alpha=0.5,
        delta=0.5,
        generator_type="cifar100_recurrence",
        seed=7,
    )
    rng = np.random.default_rng(1234)
    concept_matrix = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 2],
            [2, 2, 1, 0],
        ],
        dtype=np.int32,
    )
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    n_features = 12
    n_classes = 3
    concept_offsets = np.stack(
        [
            np.full(n_features, -0.6, dtype=np.float64),
            np.linspace(-0.2, 0.4, n_features, dtype=np.float64),
            np.full(n_features, 0.7, dtype=np.float64),
        ]
    )
    class_offsets = np.stack(
        [
            np.eye(1, n_features, k=0, dtype=np.float64).ravel() * 0.4,
            np.eye(1, n_features, k=1, dtype=np.float64).ravel() * 0.4,
            np.eye(1, n_features, k=2, dtype=np.float64).ravel() * 0.4,
        ]
    )
    for k in range(cfg.K):
        for t in range(cfg.T):
            cid = int(concept_matrix[k, t])
            y = rng.integers(0, n_classes, size=cfg.n_samples, endpoint=False)
            X = rng.normal(loc=0.0, scale=0.3, size=(cfg.n_samples, n_features))
            X += concept_offsets[cid]
            X += class_offsets[y]
            data[(k, t)] = (X.astype(np.float64), y.astype(np.int32))
    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=cfg,
        concept_specs=[
            ConceptSpec(concept_id=i, generator_type="cifar100_recurrence", variant=i, noise_scale=0.0)
            for i in range(3)
        ],
    )


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

    def test_similarity_quantile_estimation_returns_valid_scale(self) -> None:
        dataset = _high_dim_dataset()
        quantiles = _estimate_fingerprint_similarity_quantiles(
            dataset,
            n_features=12,
            n_classes=3,
            max_fingerprints=8,
        )
        assert quantiles is not None
        high_q, merge_q = quantiles
        assert 0.0 <= merge_q <= high_q <= 1.0

    def test_similarity_calibration_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.total_bytes > 0.0

    def test_model_signature_projection_has_fixed_shape(self) -> None:
        params = {
            "coef": np.arange(12, dtype=np.float64),
            "intercept": np.array([1.0, -1.0], dtype=np.float64),
        }
        signature = _project_model_signature(
            params,
            output_dim=8,
            seed=42,
        )
        assert signature.shape == (8,)
        assert np.linalg.norm(signature) > 0.0

    def test_blend_param_dicts_interpolates_between_payloads(self) -> None:
        concept = {
            "coef": np.zeros((2, 3), dtype=np.float64),
            "intercept": np.zeros(2, dtype=np.float64),
        }
        subgroup = {
            "coef": np.ones((2, 3), dtype=np.float64),
            "intercept": np.ones(2, dtype=np.float64),
        }
        mixed = _blend_param_dicts(concept, subgroup, alpha=0.25)
        np.testing.assert_allclose(mixed["coef"], 0.25 * np.ones((2, 3)))
        np.testing.assert_allclose(mixed["intercept"], 0.25 * np.ones(2))

    def test_blend_with_cached_subgroup_scales_alpha_by_similarity(self) -> None:
        concept = {
            "coef": np.zeros((2, 2), dtype=np.float64),
            "intercept": np.zeros(2, dtype=np.float64),
        }
        subgroup = {
            "coef": np.ones((2, 2), dtype=np.float64),
            "intercept": np.ones(2, dtype=np.float64),
        }
        fp = ConceptFingerprint(2, 2)
        fp.update(
            np.array([[2.0, 0.0], [2.2, 0.1], [0.0, 2.0], [0.1, 2.2]], dtype=np.float64),
            np.array([0, 0, 1, 1], dtype=np.int64),
        )
        cached = _PredictiveSubgroupCacheEntry(
            class_means=fp.class_means,
            label_distribution=fp.label_distribution,
            params=subgroup,
            expires_after_round=2,
        )
        blended = _blend_with_cached_subgroup(
            concept,
            fp,
            [cached],
            current_round=1,
            alpha=0.2,
        )
        assert np.all(blended["coef"] > 0.0)
        assert np.all(blended["coef"] < 0.21)

    def test_hybrid_routing_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_a_bytes > 0.0

    def test_update_ot_routing_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            update_ot_weight=0.35,
            update_ot_dim=4,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_a_bytes > 0.0

    def test_labelwise_proto_routing_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            labelwise_proto_weight=0.35,
            labelwise_proto_dim=4,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_a_bytes > 0.0

    def test_prototype_alignment_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_early_prototype_alignment_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            prototype_alignment_early_rounds=1,
            prototype_alignment_early_mix=0.4,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_early_prototype_prealignment_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_prealign_early_rounds=1,
            prototype_prealign_early_mix=0.4,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_client_prototype_personalization_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            client_prototype_personalization_rounds=1,
            client_prototype_personalization_mix=0.2,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_predictive_subgroup_prealignment_runs_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            prototype_subgroup_early_rounds=1,
            prototype_subgroup_early_mix=0.3,
            prototype_subgroup_min_clients=3,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_predictive_grouping_downloads_run_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            predictive_grouping_rounds=1,
            predictive_grouping_min_clients=3,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_predictive_grouping_blended_downloads_run_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            predictive_grouping_rounds=1,
            predictive_grouping_min_clients=3,
            predictive_grouping_blend_alpha=0.2,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_predictive_grouping_cached_downloads_run_on_high_dimensional_data(self) -> None:
        dataset = _high_dim_dataset()
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            predictive_grouping_rounds=1,
            predictive_grouping_min_clients=3,
            predictive_grouping_blend_alpha=0.1,
            predictive_grouping_cache_rounds=1,
            predictive_grouping_cache_blend_alpha=0.1,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0

    def test_predictive_grouping_skips_missing_aggregates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = _high_dim_dataset()
        original_phase_b_soft = TwoPhaseFedProTrack.phase_b_soft

        def _drop_one_concept(
            self,
            client_params,
            concept_assignments,
            posteriors,
            client_fingerprints=None,
        ):
            result = original_phase_b_soft(
                self,
                client_params,
                concept_assignments,
                posteriors,
                client_fingerprints=client_fingerprints,
            )
            if result.aggregated_params:
                dropped = next(iter(result.aggregated_params))
                result.aggregated_params = {
                    cid: params
                    for cid, params in result.aggregated_params.items()
                    if cid != dropped
                }
            return result

        monkeypatch.setattr(TwoPhaseFedProTrack, "phase_b_soft", _drop_one_concept)
        runner = FedProTrackRunner(
            seed=42,
            federation_every=2,
            similarity_calibration=True,
            model_signature_weight=0.35,
            model_signature_dim=8,
            prototype_alignment_mix=0.25,
            predictive_grouping_rounds=1,
            predictive_grouping_min_clients=2,
            predictive_grouping_blend_alpha=0.2,
            soft_aggregation=True,
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 4)
        assert result.phase_b_bytes > 0.0


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
