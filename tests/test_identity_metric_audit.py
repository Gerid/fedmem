from __future__ import annotations

"""Regression tests for the identity-metric audit.

Ensures that methods which do not perform concept identity inference
produce ``None`` for identity metrics, and that reporting/export/plotting
paths handle ``None`` correctly (displaying ``--`` or ``NaN`` instead of 0).
"""

import numpy as np
import pytest

from fedprotrack.experiments.method_registry import (
    IDENTITY_CAPABLE_METHODS,
    IDENTITY_METRIC_FIELDS,
    NON_IDENTITY_METHODS,
    canonical_method_name,
    identity_metrics_valid,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog, MetricsResult
from fedprotrack.experiments.tables import (
    _fmt,
    _get_metric,
    compute_rankings,
    compute_win_rates,
    export_summary_csv,
    generate_main_table,
)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------


class TestMethodRegistry:
    def test_identity_capable_methods_are_known(self) -> None:
        for m in IDENTITY_CAPABLE_METHODS:
            assert identity_metrics_valid(m) is True

    def test_non_identity_methods_are_known(self) -> None:
        for m in NON_IDENTITY_METHODS:
            assert identity_metrics_valid(m) is False

    def test_unknown_method_is_not_capable(self) -> None:
        assert identity_metrics_valid("SomeNewMethod") is False

    def test_variant_names_map_to_canonical_families(self) -> None:
        assert canonical_method_name("FedProTrack-linear-split") == "FedProTrack"
        assert canonical_method_name("IFCA-8") == "IFCA"
        assert identity_metrics_valid("FedProTrack-linear-split") is True
        assert identity_metrics_valid("FedAvg") is False

    def test_sets_are_disjoint(self) -> None:
        overlap = IDENTITY_CAPABLE_METHODS & NON_IDENTITY_METHODS
        assert len(overlap) == 0, f"Overlap: {overlap}"


# ---------------------------------------------------------------------------
# compute_all_metrics with identity_capable=False
# ---------------------------------------------------------------------------


def _make_log(method_name: str = "TestMethod") -> ExperimentLog:
    K, T = 3, 5
    gt = np.array([[0, 0, 1, 1, 0],
                    [1, 1, 0, 0, 1],
                    [0, 1, 1, 0, 0]], dtype=np.int32)
    pred = np.zeros((K, T), dtype=np.int32)
    acc = np.random.default_rng(42).random((K, T))
    return ExperimentLog(
        ground_truth=gt,
        predicted=pred,
        accuracy_curve=acc,
        total_bytes=1000.0,
        method_name=method_name,
    )


class TestComputeAllMetricsIdentityCapable:
    def test_identity_capable_true_produces_floats(self) -> None:
        log = _make_log()
        result = compute_all_metrics(log, identity_capable=True)
        assert isinstance(result.concept_re_id_accuracy, float)
        assert isinstance(result.assignment_entropy, float)
        assert isinstance(result.wrong_memory_reuse_rate, float)
        assert isinstance(result.per_client_re_id, np.ndarray)
        assert isinstance(result.per_timestep_re_id, np.ndarray)

    def test_identity_capable_false_produces_none(self) -> None:
        log = _make_log()
        result = compute_all_metrics(log, identity_capable=False)
        assert result.concept_re_id_accuracy is None
        assert result.assignment_entropy is None
        assert result.wrong_memory_reuse_rate is None
        assert result.per_client_re_id is None
        assert result.per_timestep_re_id is None

    def test_non_identity_still_has_accuracy_metrics(self) -> None:
        log = _make_log()
        result = compute_all_metrics(log, identity_capable=False)
        # Non-identity metrics should still be computed
        assert result.final_accuracy is not None
        assert result.accuracy_auc is not None
        assert result.worst_window_dip is not None
        assert result.budget_normalized_score is not None


# ---------------------------------------------------------------------------
# MetricsResult with None identity fields
# ---------------------------------------------------------------------------


def _none_identity_metrics() -> MetricsResult:
    return MetricsResult(
        concept_re_id_accuracy=None,
        assignment_entropy=None,
        wrong_memory_reuse_rate=None,
        worst_window_dip=0.1,
        worst_window_recovery=2,
        budget_normalized_score=0.01,
        per_client_re_id=None,
        per_timestep_re_id=None,
        final_accuracy=0.75,
        accuracy_auc=0.70,
    )


def _full_metrics(acc: float = 0.8) -> MetricsResult:
    return MetricsResult(
        concept_re_id_accuracy=acc,
        assignment_entropy=0.3,
        wrong_memory_reuse_rate=1.0 - acc,
        worst_window_dip=0.1,
        worst_window_recovery=2,
        budget_normalized_score=0.01,
        per_client_re_id=np.array([acc]),
        per_timestep_re_id=np.array([acc]),
        final_accuracy=0.75,
        accuracy_auc=0.70,
    )


class TestMetricsResultNoneIdentity:
    def test_to_dict_with_none_identity(self) -> None:
        mr = _none_identity_metrics()
        d = mr.to_dict()
        assert d["concept_re_id_accuracy"] is None
        assert d["assignment_entropy"] is None
        assert d["wrong_memory_reuse_rate"] is None
        assert d["per_client_re_id"] is None
        assert d["per_timestep_re_id"] is None
        # Non-identity metrics are present
        assert d["final_accuracy"] == 0.75

    def test_to_dict_with_full_identity(self) -> None:
        mr = _full_metrics()
        d = mr.to_dict()
        assert d["concept_re_id_accuracy"] == pytest.approx(0.8)
        assert d["per_client_re_id"] is not None


# ---------------------------------------------------------------------------
# Tables: _fmt and _get_metric with None
# ---------------------------------------------------------------------------


class TestTablesNoneHandling:
    def test_fmt_none_returns_dash(self) -> None:
        assert _fmt(None) == "--"

    def test_fmt_float_returns_formatted(self) -> None:
        assert _fmt(0.123) == "0.123"

    def test_get_metric_none_identity(self) -> None:
        mr = _none_identity_metrics()
        assert _get_metric(mr, "concept_re_id_accuracy") is None
        assert _get_metric(mr, "assignment_entropy") is None
        assert _get_metric(mr, "wrong_memory_reuse_rate") is None
        # Non-identity metrics should work
        assert _get_metric(mr, "final_accuracy") == pytest.approx(0.75)

    def test_rankings_with_none_identity(self) -> None:
        """Methods with None identity metrics get worst rank for that metric."""
        all_results = {
            "Capable": [_full_metrics(0.8)],
            "NonCapable": [_none_identity_metrics()],
        }
        rankings = compute_rankings(all_results, "concept_re_id_accuracy")
        # Capable should rank better (lower rank)
        assert rankings["Capable"] < rankings["NonCapable"]

    def test_win_rates_with_none_identity(self) -> None:
        all_results = {
            "Capable": [_full_metrics(0.8)],
            "NonCapable": [_none_identity_metrics()],
        }
        win_rates = compute_win_rates(all_results, "concept_re_id_accuracy")
        assert win_rates["Capable"] == 1.0
        assert win_rates["NonCapable"] == 0.0

    def test_main_table_with_none_identity_shows_dashes(self) -> None:
        all_results = {
            "FedProTrack": [_full_metrics(0.8)],
            "FedAvg": [_none_identity_metrics()],
        }
        latex = generate_main_table(all_results)
        # FedAvg row should contain "--" for identity metrics
        assert "--" in latex

    def test_export_csv_with_none_identity(self, tmp_path) -> None:
        all_results = {
            "FedProTrack": [_full_metrics(0.8)],
            "FedAvg": [_none_identity_metrics()],
        }
        csv_path = tmp_path / "summary.csv"
        export_summary_csv(all_results, csv_path)
        content = csv_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 methods
        # FedAvg should have empty cells for identity metrics
        fedavg_line = [l for l in lines if "FedAvg" in l][0]
        # concept_re_id_accuracy column should be empty
        parts = fedavg_line.split(",")
        # n_settings is col 1, then metrics start at col 2
        # concept_re_id_accuracy is first metric
        assert parts[2] == ""  # empty cell for None


# ---------------------------------------------------------------------------
# Visualization: bar chart NaN handling
# ---------------------------------------------------------------------------


class TestVisualizationNoneHandling:
    def test_plot_metric_comparison_with_none(self, tmp_path) -> None:
        """Bar chart should not crash when identity metrics are None."""
        import matplotlib
        matplotlib.use("Agg")

        from fedprotrack.metrics.visualization import plot_metric_comparison

        method_results = {
            "FedProTrack": _full_metrics(0.8),
            "FedAvg": _none_identity_metrics(),
        }
        path = tmp_path / "comparison.png"
        fig = plot_metric_comparison(method_results, output_path=path)
        assert path.exists()
        import matplotlib.pyplot as plt
        plt.close(fig)
