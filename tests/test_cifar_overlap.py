from __future__ import annotations

import numpy as np

from fedprotrack.drift_generator import GeneratorConfig
from fedprotrack.experiments.cifar_overlap import (
    build_overlap_concept_classes,
    build_phase_concept_matrix,
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
