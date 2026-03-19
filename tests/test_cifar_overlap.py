from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from fedprotrack.drift_generator import GeneratorConfig
from fedprotrack.experiments.cifar_overlap import (
    build_overlap_concept_classes,
    build_phase_concept_matrix,
    make_result_metadata,
    run_fpt,
    summarize_root_cause,
)
from fedprotrack.experiments.tables import export_summary_csv
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog


def _make_identity_log() -> ExperimentLog:
    gt = np.array(
        [[0, 0, 1, 1],
         [1, 1, 0, 0]],
        dtype=np.int32,
    )
    pred = gt.copy()
    soft = np.array(
        [[[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
         [[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]],
        dtype=np.float64,
    )
    acc = np.full_like(gt, 0.75, dtype=np.float64)
    return ExperimentLog(
        ground_truth=gt,
        predicted=pred,
        soft_assignments=soft,
        accuracy_curve=acc,
        total_bytes=1000.0,
        method_name="FedProTrack",
    )


class TestDiagnosticMetricIntegration:
    def test_compute_all_metrics_populates_diagnostics(self) -> None:
        result = compute_all_metrics(_make_identity_log(), identity_capable=True)
        assert result.assignment_switch_rate is not None
        assert result.avg_clients_per_concept is not None
        assert result.singleton_group_ratio is not None
        assert result.memory_reuse_rate is not None
        assert result.routing_consistency is not None

    def test_non_identity_metrics_leave_diagnostics_empty(self) -> None:
        result = compute_all_metrics(_make_identity_log(), identity_capable=False)
        assert result.assignment_switch_rate is None
        assert result.avg_clients_per_concept is None
        assert result.singleton_group_ratio is None
        assert result.memory_reuse_rate is None
        assert result.routing_consistency is None

    def test_export_summary_csv_includes_new_metric_columns(self, tmp_path) -> None:
        result = compute_all_metrics(_make_identity_log(), identity_capable=True)
        out = tmp_path / "summary.csv"
        export_summary_csv({"FedProTrack": [result]}, out)
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert "assignment_switch_rate" in header
        assert "singleton_group_ratio" in header
        assert "routing_consistency" in header


class TestOverlapConceptClasses:
    def test_zero_overlap_is_disjoint(self) -> None:
        concept_classes = build_overlap_concept_classes(0.0)
        seen: set[int] = set()
        for cls_subset in concept_classes.values():
            assert len(cls_subset) == 5
            assert seen.isdisjoint(cls_subset)
            seen.update(cls_subset)

    def test_positive_overlap_shares_adjacent_classes(self) -> None:
        concept_classes = build_overlap_concept_classes(0.4)
        overlap = set(concept_classes[0]) & set(concept_classes[1])
        assert len(overlap) == 2


class TestPhaseConceptMatrix:
    def test_builds_expected_phase_assignments(self) -> None:
        matrix = build_phase_concept_matrix(
            4,
            12,
            phase_concepts=[[0, 1], [2, 3], [0, 1]],
            seed=42,
            switch_prob=0.0,
        )
        assert matrix.shape == (4, 12)
        assert set(np.unique(matrix[:, :4])).issubset({0, 1})
        assert set(np.unique(matrix[:, 4:8])).issubset({2, 3})
        assert set(np.unique(matrix[:, 8:])).issubset({0, 1})


class TestCIFARGeneratorAliases:
    def test_generator_config_accepts_overlap_aliases(self) -> None:
        GeneratorConfig(generator_type="cifar100_overlap")
        GeneratorConfig(generator_type="cifar100_label_overlap")


class TestSubsetBenchmarkRunnerOverrides:
    def test_run_fpt_forwards_global_shared_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_make_plan_c_config(**overrides):
            captured["overrides"] = overrides
            return {"fake": True}

        class DummyRunner:
            def __init__(self, **kwargs):
                captured["runner_kwargs"] = kwargs

            def run(self, ds):
                del ds
                return SimpleNamespace(
                    accuracy_matrix=np.ones((1, 1), dtype=np.float64),
                    total_bytes=12.0,
                    spawned_concepts=1,
                    merged_concepts=0,
                    active_concepts=2,
                )

        monkeypatch.setattr(
            "fedprotrack.experiments.cifar_overlap.make_plan_c_config",
            fake_make_plan_c_config,
        )
        monkeypatch.setattr(
            "fedprotrack.posterior.fedprotrack_runner.FedProTrackRunner",
            DummyRunner,
        )

        ds = SimpleNamespace(concept_matrix=np.array([[0, 1]], dtype=np.int32))
        acc, total_bytes, spawned, merged, active = run_fpt(
            ds,
            fed_every=2,
            epochs=1,
            lr=0.1,
            seed=7,
            global_shared_aggregation=False,
            routed_local_training=True,
            fingerprint_source="hybrid_raw_pre_adapter",
            routed_write_top_k=2,
            routed_write_min_secondary_weight=0.2,
            routed_read_top_k=2,
            routed_read_temperature=0.5,
            routed_read_only_on_ambiguity=True,
            routed_read_min_entropy=0.2,
            routed_read_min_secondary_weight=0.2,
            routed_read_max_primary_gap=0.35,
            max_spawn_clusters_per_round=2,
            novelty_hysteresis_rounds=2,
            factorized_slot_preserving=True,
            factorized_primary_anchor_alpha=0.3,
            factorized_secondary_anchor_alpha=0.8,
            factorized_primary_consolidation_steps=2,
            factorized_primary_consolidation_mode="head_only",
        )

        assert acc.shape == (1, 1)
        assert total_bytes == 12.0
        assert spawned == 1
        assert merged == 0
        assert active == 2
        assert captured["overrides"]["global_shared_aggregation"] is False
        assert captured["runner_kwargs"]["routed_local_training"] is True
        assert captured["runner_kwargs"]["fingerprint_source"] == "hybrid_raw_pre_adapter"
        assert captured["runner_kwargs"]["routed_write_top_k"] == 2
        assert captured["runner_kwargs"]["routed_write_min_secondary_weight"] == pytest.approx(0.2)
        assert captured["runner_kwargs"]["routed_read_top_k"] == 2
        assert captured["runner_kwargs"]["routed_read_temperature"] == pytest.approx(0.5)
        assert captured["runner_kwargs"]["routed_read_only_on_ambiguity"] is True
        assert captured["runner_kwargs"]["routed_read_min_entropy"] == pytest.approx(0.2)
        assert captured["runner_kwargs"]["routed_read_min_secondary_weight"] == pytest.approx(0.2)
        assert captured["runner_kwargs"]["routed_read_max_primary_gap"] == pytest.approx(0.35)
        assert captured["overrides"]["max_spawn_clusters_per_round"] == 2
        assert captured["overrides"]["novelty_hysteresis_rounds"] == 2
        assert captured["runner_kwargs"]["factorized_slot_preserving"] is True
        assert captured["runner_kwargs"]["factorized_primary_anchor_alpha"] == pytest.approx(0.3)
        assert captured["runner_kwargs"]["factorized_secondary_anchor_alpha"] == pytest.approx(0.8)
        assert captured["runner_kwargs"]["factorized_primary_consolidation_steps"] == 2
        assert captured["runner_kwargs"]["factorized_primary_consolidation_mode"] == "head_only"



class TestResultTableHelpers:
    def test_make_result_metadata_populates_expected_columns(self) -> None:
        metadata = make_result_metadata(
            model_type="feature_adapter",
            fingerprint_source="raw_input",
            expert_update_policy="map_only",
            shared_update_policy="concept_local",
            global_shared_aggregation=False,
            routed_write_top_k=2,
            routed_write_min_secondary_weight=0.2,
            routed_read_top_k=2,
            routed_read_temperature=0.5,
            routed_read_only_on_ambiguity=True,
            routed_read_min_entropy=0.2,
            routed_read_min_secondary_weight=0.2,
            routed_read_max_primary_gap=0.35,
            max_spawn_clusters_per_round=2,
            novelty_hysteresis_rounds=2,
            factorized_slot_preserving=True,
            factorized_primary_anchor_alpha=0.25,
            factorized_secondary_anchor_alpha=0.75,
            factorized_primary_consolidation_steps=2,
            factorized_primary_consolidation_mode="head_only",
            spawned=3,
            merged=1,
            active=4,
            assignment_switch_rate=0.25,
            avg_clients_per_concept=1.5,
            singleton_group_ratio=0.2,
            memory_reuse_rate=0.4,
            routing_consistency=0.8,
            shared_drift_norm=0.15,
            expert_update_coverage=1.25,
            multi_route_rate=0.30,
        )

        assert metadata["model_type"] == "feature_adapter"
        assert metadata["fingerprint_source"] == "raw_input"
        assert metadata["shared_update_policy"] == "concept_local"
        assert metadata["routed_write_top_k"] == 2
        assert metadata["routed_write_min_secondary_weight"] == pytest.approx(0.2)
        assert metadata["routed_read_top_k"] == 2
        assert metadata["routed_read_temperature"] == pytest.approx(0.5)
        assert metadata["routed_read_only_on_ambiguity"] is True
        assert metadata["routed_read_min_entropy"] == pytest.approx(0.2)
        assert metadata["routed_read_min_secondary_weight"] == pytest.approx(0.2)
        assert metadata["routed_read_max_primary_gap"] == pytest.approx(0.35)
        assert metadata["max_spawn_clusters_per_round"] == 2
        assert metadata["novelty_hysteresis_rounds"] == 2
        assert metadata["factorized_slot_preserving"] is True
        assert metadata["factorized_primary_anchor_alpha"] == pytest.approx(0.25)
        assert metadata["factorized_secondary_anchor_alpha"] == pytest.approx(0.75)
        assert metadata["factorized_primary_consolidation_steps"] == 2
        assert metadata["factorized_primary_consolidation_mode"] == "head_only"
        assert metadata["spawned"] == 3
        assert metadata["routing_consistency"] == 0.8
        assert metadata["shared_drift_norm"] == pytest.approx(0.15)
        assert metadata["expert_update_coverage"] == pytest.approx(1.25)
        assert metadata["multi_route_rate"] == pytest.approx(0.30)

    @staticmethod
    def _row(
        *,
        seed: int,
        final: float,
        phase3: float,
        recovery_next_fed: float,
        bytes_: float,
        concept_re_id_accuracy: float,
        overlap_ratio: float = 0.0,
        assignment_switch_rate: float = 0.2,
        avg_clients_per_concept: float = 1.5,
        singleton_group_ratio: float = 0.2,
        memory_reuse_rate: float = 0.5,
        routing_consistency: float = 0.8,
        shared_drift_norm: float | None = None,
        expert_update_coverage: float | None = None,
        multi_route_rate: float | None = None,
        spawned: float | None = None,
        active: float | None = None,
    ) -> dict[str, object]:
        return {
            "seed": seed,
            "final": final,
            "phase3": phase3,
            "recovery_next_fed": recovery_next_fed,
            "bytes": bytes_,
            "concept_re_id_accuracy": concept_re_id_accuracy,
            "overlap_ratio": overlap_ratio,
            "assignment_switch_rate": assignment_switch_rate,
            "avg_clients_per_concept": avg_clients_per_concept,
            "singleton_group_ratio": singleton_group_ratio,
            "memory_reuse_rate": memory_reuse_rate,
            "routing_consistency": routing_consistency,
            "shared_drift_norm": shared_drift_norm,
            "expert_update_coverage": expert_update_coverage,
            "multi_route_rate": multi_route_rate,
            "spawned": spawned,
            "active": active,
        }

    def test_summarize_root_cause_detects_phase_a_failure(self) -> None:
        lines = summarize_root_cause(
            [self._row(seed=42, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50)],
            [self._row(seed=42, final=0.58, phase3=0.54, recovery_next_fed=0.55, bytes_=10.0, concept_re_id_accuracy=0.49)],
        )
        assert any("问题仍在 Phase A 表征" in line for line in lines)

    def test_summarize_root_cause_detects_downstream_gap(self) -> None:
        lines = summarize_root_cause(
            [self._row(seed=42, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50, shared_drift_norm=0.10, expert_update_coverage=1.0, multi_route_rate=0.0)],
            [self._row(seed=42, final=0.59, phase3=0.53, recovery_next_fed=0.54, bytes_=10.0, concept_re_id_accuracy=0.54, shared_drift_norm=0.25, expert_update_coverage=1.0, multi_route_rate=0.3)],
        )
        assert any("read/write mismatch + shared drift" in line for line in lines)

    def test_summarize_root_cause_detects_overlap_veto(self) -> None:
        lines = summarize_root_cause(
            [
                self._row(seed=42, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50, overlap_ratio=0.0, assignment_switch_rate=0.10, singleton_group_ratio=0.10),
                self._row(seed=42, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50, overlap_ratio=0.6, assignment_switch_rate=0.12, singleton_group_ratio=0.12),
            ],
            [
                self._row(seed=42, final=0.59, phase3=0.54, recovery_next_fed=0.55, bytes_=10.0, concept_re_id_accuracy=0.52, overlap_ratio=0.0, assignment_switch_rate=0.14, singleton_group_ratio=0.14),
                self._row(seed=42, final=0.57, phase3=0.53, recovery_next_fed=0.54, bytes_=10.0, concept_re_id_accuracy=0.53, overlap_ratio=0.6, assignment_switch_rate=0.22, singleton_group_ratio=0.24),
            ],
            overlap_compare=(0.0, 0.6),
        )
        assert any("overfits recurrence" in line or "overlap veto" in line for line in lines)

    def test_summarize_root_cause_detects_single_seed_instability(self) -> None:
        baseline_rows = [
            self._row(seed=42, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50, spawned=3, active=3),
            self._row(seed=123, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50, spawned=3, active=3),
            self._row(seed=456, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50, spawned=3, active=3),
        ]
        candidate_rows = [
            self._row(seed=42, final=0.70, phase3=0.56, recovery_next_fed=0.60, bytes_=10.0, concept_re_id_accuracy=0.58, spawned=2, active=2),
            self._row(seed=123, final=0.50, phase3=0.54, recovery_next_fed=0.52, bytes_=10.0, concept_re_id_accuracy=0.57, spawned=5, active=4),
            self._row(seed=456, final=0.50, phase3=0.54, recovery_next_fed=0.53, bytes_=10.0, concept_re_id_accuracy=0.57, spawned=6, active=5),
        ]
        lines = summarize_root_cause(baseline_rows, candidate_rows)
        assert any("routing instability" in line or "single-seed gain did not survive multi-seed averaging" in line for line in lines)

    def test_summarize_root_cause_detects_bytes_regression(self) -> None:
        lines = summarize_root_cause(
            [self._row(seed=42, final=0.60, phase3=0.55, recovery_next_fed=0.56, bytes_=10.0, concept_re_id_accuracy=0.50)],
            [self._row(seed=42, final=0.61, phase3=0.56, recovery_next_fed=0.57, bytes_=30.0, concept_re_id_accuracy=0.55)],
        )
        assert any("cost-inefficient improvement" in line for line in lines)
