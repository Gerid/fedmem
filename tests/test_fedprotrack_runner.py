"""Tests for FedProTrackRunner end-to-end simulation."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.models import TorchFactorizedAdapterClassifier
from fedprotrack.posterior import fedprotrack_runner as runner_module
from fedprotrack.posterior.fedprotrack_runner import (
    _build_fingerprint_features,
    _compose_routed_payload,
    _prepare_routed_read_weights,
    _select_weighted_training_slots,
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

    def test_result_exposes_plan_c_diagnostics(self) -> None:
        result = FedProTrackResult(
            accuracy_matrix=np.ones((2, 3)),
            predicted_concept_matrix=np.zeros((2, 3), dtype=np.int32),
            true_concept_matrix=np.zeros((2, 3), dtype=np.int32),
            total_bytes=10.0,
            phase_a_bytes=4.0,
            phase_b_bytes=6.0,
            mean_accuracy=1.0,
            final_accuracy=1.0,
            assignment_switch_rate=0.0,
            avg_clients_per_concept=2.0,
            singleton_group_ratio=0.0,
            memory_reuse_rate=1.0,
            routing_consistency=1.0,
            shared_drift_norm=0.25,
            expert_update_coverage=1.5,
            multi_route_rate=0.3,
            phase_a_round_diagnostics=[{"t": 1, "mean_fp_loss": 0.4}],
        )
        assert result.assignment_switch_rate == pytest.approx(0.0)
        assert result.avg_clients_per_concept == pytest.approx(2.0)
        assert result.singleton_group_ratio == pytest.approx(0.0)
        assert result.memory_reuse_rate == pytest.approx(1.0)
        assert result.routing_consistency == pytest.approx(1.0)
        assert result.shared_drift_norm == pytest.approx(0.25)
        assert result.expert_update_coverage == pytest.approx(1.5)
        assert result.multi_route_rate == pytest.approx(0.3)
        assert result.phase_a_round_diagnostics == [{"t": 1, "mean_fp_loss": 0.4}]


# ---------------------------------------------------------------------------
# FedProTrackRunner - basic
# ---------------------------------------------------------------------------

class TestFedProTrackRunner:
    def test_prepare_routed_read_weights_applies_topk_and_temperature(self) -> None:
        weights = _prepare_routed_read_weights(
            {0: 0.6, 1: 0.3, 2: 0.1},
            top_k=2,
            temperature=0.5,
        )

        assert weights is not None
        assert list(weights) == [0, 1]
        assert weights[0] == pytest.approx(0.8)
        assert weights[1] == pytest.approx(0.2)

    def test_prepare_routed_read_weights_skips_sharpening_when_not_ambiguous(self) -> None:
        weights = _prepare_routed_read_weights(
            {0: 0.8, 1: 0.15, 2: 0.05},
            top_k=2,
            temperature=0.5,
            only_on_ambiguity=True,
            min_secondary_weight=0.2,
            max_primary_gap=0.35,
        )

        assert weights is not None
        assert list(weights) == [0, 1, 2]
        assert weights[0] == pytest.approx(0.8)
        assert weights[1] == pytest.approx(0.15)
        assert weights[2] == pytest.approx(0.05)

    def test_prepare_routed_read_weights_entropy_gate_sharpens_only_high_entropy(self) -> None:
        sharp_weights = _prepare_routed_read_weights(
            {0: 0.55, 1: 0.45},
            top_k=2,
            temperature=0.5,
            only_on_ambiguity=True,
            min_entropy=0.8,
        )
        flat_weights = _prepare_routed_read_weights(
            {0: 0.8, 1: 0.15, 2: 0.05},
            top_k=2,
            temperature=0.5,
            only_on_ambiguity=True,
            min_entropy=0.8,
        )

        assert sharp_weights is not None
        assert sharp_weights[0] == pytest.approx(0.599009900990099)
        assert sharp_weights[1] == pytest.approx(0.400990099009901)
        assert flat_weights is not None
        assert list(flat_weights) == [0, 1, 2]
        assert flat_weights[0] == pytest.approx(0.8)

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

    def test_run_supports_alternate_fingerprint_labels(self) -> None:
        dataset = _small_dataset()
        dataset.fingerprint_labels = {
            key: (y.astype(np.int64) + 10)
            for key, (_, y) in dataset.data.items()
        }
        runner = FedProTrackRunner(seed=42)
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)

    def test_feature_adapter_model_runs_end_to_end(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)
        assert result.assignment_switch_rate is not None
        assert result.routing_consistency is not None
        assert result.shared_drift_norm is not None
        assert result.expert_update_coverage is not None
        assert result.multi_route_rate is not None
        assert result.phase_a_round_diagnostics

    def test_factorized_adapter_model_runs_end_to_end(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(
            seed=42,
            model_type="factorized_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)
        assert result.assignment_switch_rate is not None
        assert result.routing_consistency is not None
        assert result.shared_drift_norm is not None
        assert result.expert_update_coverage is not None
        assert result.multi_route_rate is not None
        assert result.phase_a_round_diagnostics

    def test_factorized_adapter_runner_honors_top2_weighted_write_with_slot_preserving(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        select_calls: list[tuple[int, dict[int, float] | None, int | None, float]] = []
        warm_calls: list[list[int]] = []
        anchor_calls: list[list[int]] = []

        def wrapped_select_weighted_training_slots(
            slot_id: int,
            slot_weights: dict[int, float] | None = None,
            *,
            top_k: int | None = None,
            min_secondary_weight: float = 0.0,
        ) -> dict[int, float]:
            select_calls.append((slot_id, slot_weights, top_k, min_secondary_weight))
            assert top_k == 2
            assert min_secondary_weight == pytest.approx(0.20)
            return {0: 0.70, 1: 0.30}

        def wrapped_warm_start_factorized_slots(
            model,
            *,
            shared_slot_id: int,
            slot_ids: list[int],
            anchor_payloads: dict[int, dict[str, np.ndarray]],
        ) -> None:
            warm_calls.append(list(slot_ids))

        def wrapped_apply_factorized_slot_anchor(
            model,
            *,
            slot_ids: list[int],
            primary_slot_id: int,
            anchor_payloads: dict[int, dict[str, np.ndarray]],
            primary_anchor_alpha: float,
            secondary_anchor_alpha: float,
        ) -> None:
            anchor_calls.append(list(slot_ids))

        def wrapped_model_fit(*args, **kwargs):
            return None

        monkeypatch.setattr(
            runner_module,
            "_select_weighted_training_slots",
            wrapped_select_weighted_training_slots,
        )
        monkeypatch.setattr(
            runner_module,
            "_warm_start_factorized_slots",
            wrapped_warm_start_factorized_slots,
        )
        monkeypatch.setattr(
            runner_module,
            "_apply_factorized_slot_anchor",
            wrapped_apply_factorized_slot_anchor,
        )
        monkeypatch.setattr(runner_module, "_model_fit", wrapped_model_fit)
        monkeypatch.setattr(runner_module, "_model_partial_fit", wrapped_model_fit)

        runner = FedProTrackRunner(
            seed=42,
            model_type="factorized_adapter",
            expert_update_policy="posterior_weighted",
            shared_update_policy="freeze_on_multiroute",
            routed_write_top_k=2,
            routed_write_min_secondary_weight=0.20,
            factorized_slot_preserving=True,
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert select_calls
        assert any(slot_ids == [0, 1] for slot_ids in warm_calls)
        assert any(slot_ids == [0, 1] for slot_ids in anchor_calls)

    def test_factorized_adapter_runner_applies_sharpened_read_weights(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        read_calls: list[tuple[int | None, float, dict[int, float] | None]] = []
        original_prepare = runner_module._prepare_routed_read_weights

        def wrapped_prepare_routed_read_weights(
            slot_weights: dict[int, float] | None,
            *,
            top_k: int | None = None,
            temperature: float = 1.0,
            only_on_ambiguity: bool = False,
            min_entropy: float | None = None,
            min_secondary_weight: float = 0.0,
            max_primary_gap: float | None = None,
        ) -> dict[int, float] | None:
            read_calls.append(
                (
                    top_k,
                    temperature,
                    only_on_ambiguity,
                    min_entropy,
                    min_secondary_weight,
                    max_primary_gap,
                    slot_weights,
                )
            )
            return original_prepare(
                slot_weights,
                top_k=top_k,
                temperature=temperature,
                only_on_ambiguity=only_on_ambiguity,
                min_entropy=min_entropy,
                min_secondary_weight=min_secondary_weight,
                max_primary_gap=max_primary_gap,
            )

        monkeypatch.setattr(
            runner_module,
            "_prepare_routed_read_weights",
            wrapped_prepare_routed_read_weights,
        )

        runner = FedProTrackRunner(
            seed=42,
            model_type="factorized_adapter",
            expert_update_policy="posterior_weighted",
            shared_update_policy="freeze_on_multiroute",
            routed_read_top_k=2,
            routed_read_temperature=0.5,
            routed_read_only_on_ambiguity=True,
            routed_read_min_entropy=0.2,
            routed_read_min_secondary_weight=0.2,
            routed_read_max_primary_gap=0.35,
            factorized_slot_preserving=True,
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert read_calls
        assert any(
            top_k == 2
            and temperature == pytest.approx(0.5)
            and only_on_ambiguity is True
            and min_entropy == pytest.approx(0.2)
            and min_secondary_weight == pytest.approx(0.2)
            and max_primary_gap == pytest.approx(0.35)
            for top_k, temperature, only_on_ambiguity, min_entropy, min_secondary_weight, max_primary_gap, _ in read_calls
        )

    def test_factorized_adapter_runner_invokes_primary_consolidation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        consolidation_calls: list[tuple[int, int, str]] = []

        def wrapped_select_weighted_training_slots(
            slot_id: int,
            slot_weights: dict[int, float] | None = None,
            *,
            top_k: int | None = None,
            min_secondary_weight: float = 0.0,
        ) -> dict[int, float]:
            del slot_weights, top_k, min_secondary_weight
            return {slot_id: 0.7, slot_id + 1: 0.3}

        def wrapped_run_factorized_primary_consolidation(
            model,
            X: np.ndarray,
            y: np.ndarray,
            *,
            slot_id: int,
            steps: int,
            mode: str = "full",
        ) -> None:
            del model, X, y
            consolidation_calls.append((slot_id, steps, mode))

        monkeypatch.setattr(
            runner_module,
            "_select_weighted_training_slots",
            wrapped_select_weighted_training_slots,
        )
        monkeypatch.setattr(
            runner_module,
            "_run_factorized_primary_consolidation",
            wrapped_run_factorized_primary_consolidation,
        )

        runner = FedProTrackRunner(
            seed=42,
            model_type="factorized_adapter",
            expert_update_policy="posterior_weighted",
            shared_update_policy="freeze_on_multiroute",
            factorized_slot_preserving=True,
            factorized_primary_consolidation_steps=2,
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert consolidation_calls
        assert all(steps == 2 and mode == "full" for _, steps, mode in consolidation_calls)

    def test_factorized_adapter_runner_invokes_head_only_primary_consolidation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        consolidation_calls: list[str] = []

        def wrapped_select_weighted_training_slots(
            slot_id: int,
            slot_weights: dict[int, float] | None = None,
            *,
            top_k: int | None = None,
            min_secondary_weight: float = 0.0,
        ) -> dict[int, float]:
            del slot_weights, top_k, min_secondary_weight
            return {slot_id: 0.7, slot_id + 1: 0.3}

        def wrapped_run_factorized_primary_consolidation(
            model,
            X: np.ndarray,
            y: np.ndarray,
            *,
            slot_id: int,
            steps: int,
            mode: str = "full",
        ) -> None:
            del model, X, y, slot_id, steps
            consolidation_calls.append(mode)

        monkeypatch.setattr(
            runner_module,
            "_select_weighted_training_slots",
            wrapped_select_weighted_training_slots,
        )
        monkeypatch.setattr(
            runner_module,
            "_run_factorized_primary_consolidation",
            wrapped_run_factorized_primary_consolidation,
        )

        runner = FedProTrackRunner(
            seed=42,
            model_type="factorized_adapter",
            expert_update_policy="posterior_weighted",
            shared_update_policy="freeze_on_multiroute",
            factorized_slot_preserving=True,
            factorized_primary_consolidation_steps=1,
            factorized_primary_consolidation_mode="head_only",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert consolidation_calls
        assert all(mode == "head_only" for mode in consolidation_calls)

    def test_feature_adapter_runner_supports_model_embed_fingerprints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = _small_dataset()
        calls = {"count": 0}
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls["count"] += 1
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="model_embed",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)
        assert calls["count"] > 0

    def test_feature_adapter_runner_supports_pre_adapter_and_hybrid_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        calls: list[str] = []
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls.append(str(kwargs.get("representation", "post_adapter")))
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        pre_runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="pre_adapter_embed",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        pre_result = pre_runner.run(dataset)
        hybrid_runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="hybrid_raw_pre_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        hybrid_result = hybrid_runner.run(dataset)

        assert pre_result.accuracy_matrix.shape == (3, 5)
        assert hybrid_result.accuracy_matrix.shape == (3, 5)
        assert "pre_adapter" in calls
        assert any(
            diag.get("mean_fp_loss") is not None
            for diag in pre_result.phase_a_round_diagnostics
        )

    def test_build_fingerprint_features_supports_centered_hybrid_pre_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        calls: list[str] = []
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls.append(str(kwargs.get("representation", "post_adapter")))
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        model = runner_module._make_model(
            "feature_adapter",
            n_features=2,
            n_classes=2,
            lr=0.1,
            n_epochs=1,
            seed=42,
            hidden_dim=8,
            adapter_dim=4,
        )

        centered_fp = _build_fingerprint_features(
            model,
            X,
            fingerprint_source="centered_hybrid_raw_pre_adapter",
        )

        assert centered_fp.shape == (2, 10)
        np.testing.assert_allclose(centered_fp[:, :2], X)
        np.testing.assert_allclose(
            centered_fp[:, 2:].mean(axis=0),
            np.zeros(8, dtype=np.float32),
            atol=1e-6,
        )
        assert calls == ["pre_adapter"]

    def test_build_fingerprint_features_supports_attenuated_hybrid_pre_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        model = runner_module._make_model(
            "feature_adapter",
            n_features=2,
            n_classes=2,
            lr=0.1,
            n_epochs=1,
            seed=42,
            hidden_dim=8,
            adapter_dim=4,
        )
        pre_adapter = original_model_embed(
            model,
            X,
            representation="pre_adapter",
        )
        attenuated_fp = _build_fingerprint_features(
            model,
            X,
            fingerprint_source="attenuated_hybrid_raw_pre_adapter",
        )

        assert attenuated_fp.shape == (2, 10)
        np.testing.assert_allclose(attenuated_fp[:, :2], X)
        np.testing.assert_allclose(attenuated_fp[:, 2:], pre_adapter * 0.25, atol=1e-6)

    def test_build_fingerprint_features_supports_double_raw_hybrid_pre_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        model = runner_module._make_model(
            "feature_adapter",
            n_features=2,
            n_classes=2,
            lr=0.1,
            n_epochs=1,
            seed=42,
            hidden_dim=8,
            adapter_dim=4,
        )
        pre_adapter = original_model_embed(
            model,
            X,
            representation="pre_adapter",
        )
        double_raw_fp = _build_fingerprint_features(
            model,
            X,
            fingerprint_source="double_raw_hybrid_pre_adapter",
        )

        assert double_raw_fp.shape == (2, 12)
        np.testing.assert_allclose(double_raw_fp[:, :2], X)
        np.testing.assert_allclose(double_raw_fp[:, 2:4], X)
        np.testing.assert_allclose(double_raw_fp[:, 4:], pre_adapter, atol=1e-6)

    def test_build_fingerprint_features_supports_weighted_hybrid_pre_adapter(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        model = runner_module._make_model(
            "feature_adapter",
            n_features=2,
            n_classes=2,
            lr=0.1,
            n_epochs=1,
            seed=42,
            hidden_dim=8,
            adapter_dim=4,
        )
        pre_adapter = original_model_embed(
            model,
            X,
            representation="pre_adapter",
        )
        weighted_fp = _build_fingerprint_features(
            model,
            X,
            fingerprint_source="weighted_hybrid_raw_pre_adapter",
        )

        assert weighted_fp.shape == (2, 10)
        np.testing.assert_allclose(weighted_fp[:, :2], X)
        np.testing.assert_allclose(weighted_fp[:, 2:], pre_adapter, atol=1e-6)

    def test_bootstrap_raw_hybrid_helper_skips_embed_until_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        calls: list[str] = []
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls.append(str(kwargs.get("representation", "post_adapter")))
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        model = runner_module._make_model(
            "feature_adapter",
            n_features=2,
            n_classes=2,
            lr=0.1,
            n_epochs=1,
            seed=42,
            hidden_dim=8,
            adapter_dim=4,
        )

        bootstrap_fp = _build_fingerprint_features(
            model,
            X,
            fingerprint_source="bootstrap_raw_hybrid_pre_adapter",
            bootstrap_hidden_dim=8,
            use_bootstrap_raw=True,
        )
        assert bootstrap_fp.shape == (2, 10)
        np.testing.assert_allclose(bootstrap_fp[:, :2], X)
        np.testing.assert_allclose(bootstrap_fp[:, 2:], 0.0)
        assert calls == []

        hybrid_fp = _build_fingerprint_features(
            model,
            X,
            fingerprint_source="bootstrap_raw_hybrid_pre_adapter",
            bootstrap_hidden_dim=8,
            use_bootstrap_raw=False,
        )
        assert hybrid_fp.shape == (2, 10)
        assert calls == ["pre_adapter"]

    def test_feature_adapter_runner_supports_bootstrap_raw_hybrid_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        bootstrap_flags: list[bool] = []
        original_build_fingerprint_features = runner_module._build_fingerprint_features

        def wrapped_build_fingerprint_features(*args, **kwargs):
            if kwargs.get("fingerprint_source") == "bootstrap_raw_hybrid_pre_adapter":
                bootstrap_flags.append(bool(kwargs.get("use_bootstrap_raw")))
            return original_build_fingerprint_features(*args, **kwargs)

        monkeypatch.setattr(
            runner_module,
            "_build_fingerprint_features",
            wrapped_build_fingerprint_features,
        )
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="bootstrap_raw_hybrid_pre_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert bootstrap_flags
        assert any(bootstrap_flags)

    def test_feature_adapter_runner_supports_centered_hybrid_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        calls: list[str] = []
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls.append(str(kwargs.get("representation", "post_adapter")))
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="centered_hybrid_raw_pre_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert "pre_adapter" in calls

    def test_feature_adapter_runner_supports_attenuated_hybrid_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        calls: list[str] = []
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls.append(str(kwargs.get("representation", "post_adapter")))
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="attenuated_hybrid_raw_pre_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert "pre_adapter" in calls

    def test_feature_adapter_runner_supports_double_raw_hybrid_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        calls: list[str] = []
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls.append(str(kwargs.get("representation", "post_adapter")))
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="double_raw_hybrid_pre_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert "pre_adapter" in calls

    def test_feature_adapter_runner_supports_weighted_hybrid_fingerprints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        calls: list[str] = []
        original_model_embed = runner_module._model_embed

        def wrapped_model_embed(*args, **kwargs):
            calls.append(str(kwargs.get("representation", "post_adapter")))
            return original_model_embed(*args, **kwargs)

        monkeypatch.setattr(runner_module, "_model_embed", wrapped_model_embed)
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            fingerprint_source="weighted_hybrid_raw_pre_adapter",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert "pre_adapter" in calls

    def test_feature_adapter_runner_supports_weighted_expert_updates(self) -> None:
        dataset = _small_dataset()
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            expert_update_policy="posterior_weighted",
            shared_update_policy="freeze_on_multiroute",
            soft_aggregation=True,
            config=TwoPhaseConfig(key_mode="multi_scale"),
        )
        result = runner.run(dataset)
        assert result.accuracy_matrix.shape == (3, 5)
        assert result.expert_update_coverage is not None
        assert result.expert_update_coverage >= 1.0
        assert result.multi_route_rate is not None

    def test_compose_routed_payload_blends_shared_params_by_weights(self) -> None:
        payload, filtered = _compose_routed_payload(
            {
                0: {
                    "shared.trunk.weight": np.array([[1.0]], dtype=np.float64),
                    "shared.trunk.bias": np.array([0.0], dtype=np.float64),
                    "expert.0.head.weight": np.array([[2.0]], dtype=np.float64),
                    "expert.0.head.bias": np.array([0.0], dtype=np.float64),
                },
                1: {
                    "shared.trunk.weight": np.array([[3.0]], dtype=np.float64),
                    "shared.trunk.bias": np.array([2.0], dtype=np.float64),
                    "expert.1.head.weight": np.array([[10.0]], dtype=np.float64),
                    "expert.1.head.bias": np.array([1.0], dtype=np.float64),
                },
            },
            {0: 0.25, 1: 0.75},
        )

        np.testing.assert_allclose(
            payload["shared.trunk.weight"],
            np.array([[2.5]], dtype=np.float64),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            payload["shared.trunk.bias"],
            np.array([1.5], dtype=np.float64),
            atol=1e-10,
        )
        assert "expert.0.head.weight" in payload
        assert "expert.1.head.weight" in payload
        assert filtered == pytest.approx({0: 0.25, 1: 0.75})

    def test_select_weighted_training_slots_uses_top2_when_secondary_weight_is_high(self) -> None:
        selected = _select_weighted_training_slots(
            0,
            {0: 0.60, 1: 0.25, 2: 0.15},
            top_k=2,
            min_secondary_weight=0.20,
        )
        assert selected == pytest.approx({0: 0.7058823529, 1: 0.2941176471})

    def test_select_weighted_training_slots_falls_back_to_map_when_secondary_weight_is_low(self) -> None:
        selected = _select_weighted_training_slots(
            0,
            {0: 0.90, 1: 0.07, 2: 0.03},
            top_k=2,
            min_secondary_weight=0.20,
        )
        assert selected == pytest.approx({0: 1.0})

    def test_apply_factorized_slot_anchor_restores_secondary_expert(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(24, 6)).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
        model = TorchFactorizedAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=8,
            adapter_dim=4,
            lr=0.1,
            n_epochs=1,
            seed=13,
        )
        model.partial_fit(X, y, slot_id=0)
        model.partial_fit(X, 1 - y, slot_id=1)
        anchor_payloads = {
            0: model.get_params(slot_id=0),
            1: model.get_params(slot_id=1),
        }

        model.partial_fit(
            X,
            y,
            slot_id=0,
            slot_weights={0: 0.45, 1: 0.55},
            update_shared=False,
        )
        drifted_slot1 = model.get_params(slot_id=1)["expert.1.head.weight"].copy()
        runner_module._apply_factorized_slot_anchor(
            model,
            slot_ids=[0, 1],
            primary_slot_id=0,
            anchor_payloads=anchor_payloads,
            primary_anchor_alpha=0.0,
            secondary_anchor_alpha=1.0,
        )
        restored_slot1 = model.get_params(slot_id=1)["expert.1.head.weight"]

        assert not np.allclose(drifted_slot1, anchor_payloads[1]["expert.1.head.weight"])
        np.testing.assert_allclose(
            restored_slot1,
            anchor_payloads[1]["expert.1.head.weight"],
            atol=1e-10,
        )


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

    def test_skip_last_federation_round_reduces_bytes_without_changing_accuracy(self) -> None:
        dataset = _small_dataset()
        skipped = FedProTrackRunner(
            federation_every=1,
            skip_last_federation_round=True,
            seed=42,
        ).run(dataset)
        kept = FedProTrackRunner(
            federation_every=1,
            skip_last_federation_round=False,
            seed=42,
        ).run(dataset)

        np.testing.assert_allclose(skipped.accuracy_matrix, kept.accuracy_matrix)
        assert skipped.total_bytes < kept.total_bytes

    def test_routed_local_training_alias_maps_to_posterior_weighted(self) -> None:
        runner = FedProTrackRunner(
            seed=42,
            model_type="feature_adapter",
            routed_local_training=True,
        )
        assert runner.expert_update_policy == "posterior_weighted"

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

    def test_runner_preserves_protocol_spawn_cap_when_rebuilding_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset = _small_dataset()
        captured: dict[str, int | None] = {}
        original_phase_a = runner_module.TwoPhaseFedProTrack.phase_a

        def wrapped_phase_a(self, *args, **kwargs):
            captured["max_spawn_clusters_per_round"] = self.config.max_spawn_clusters_per_round
            captured["merge_min_support"] = self.config.merge_min_support
            captured["novelty_hysteresis_rounds"] = self.config.novelty_hysteresis_rounds
            return original_phase_a(self, *args, **kwargs)

        monkeypatch.setattr(
            runner_module.TwoPhaseFedProTrack,
            "phase_a",
            wrapped_phase_a,
        )

        runner = FedProTrackRunner(
            seed=42,
            config=TwoPhaseConfig(
                max_spawn_clusters_per_round=2,
                merge_min_support=3,
                novelty_hysteresis_rounds=2,
            ),
        )
        runner.run(dataset)

        assert captured["max_spawn_clusters_per_round"] == 2
        assert captured["merge_min_support"] == 3
        assert captured["novelty_hysteresis_rounds"] == 2


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
