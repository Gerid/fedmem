"""Tests for the resume_diagnostics module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from fedprotrack.baselines.budget_sweep import BudgetPoint
from fedprotrack.metrics.experiment_log import MetricsResult

import resume_diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        final_accuracy=0.6,
        accuracy_auc=10.5,
    )


def _make_claim_check(notes: list[str] | None = None) -> dict:
    """Build a minimal claim_check dict."""
    if notes is None:
        notes = ["E1 gate failed: FedProTrack trails IFCA by 0.1115 re-ID."]
    return {
        "artifact_complete": True,
        "expected_n_settings": 375,
        "actual_n_settings_per_method": {"FedProTrack": 375, "IFCA": 375},
        "identity_methods": ["FedProTrack", "FedDrift", "IFCA"],
        "fedprotrack_minus_ifca_reid": -0.1115,
        "fedprotrack_has_non_dominated_budget_point": False,
        "main_table_columns_ok": True,
        "notes": notes,
    }


def _make_budget_records() -> list[dict]:
    """Minimal budget point records."""
    return [
        {
            "method_name": "FedProTrack",
            "federation_every": 1,
            "total_bytes": 12000.0,
            "accuracy_auc": 10.5,
        },
        {
            "method_name": "FedAvg-Full",
            "federation_every": 1,
            "total_bytes": 4560.0,
            "accuracy_auc": 10.3,
        },
    ]


def _make_alpha_diagnostics() -> dict:
    """Minimal alpha diagnostics."""
    return {
        "alpha=0.0": {
            "FedProTrack": {
                "mean_final_accuracy": 0.65,
                "mean_accuracy_auc": 12.0,
                "mean_worst_window_dip": 0.2,
                "mean_worst_window_recovery": 2.0,
            },
            "IFCA": {
                "mean_final_accuracy": 0.66,
                "mean_accuracy_auc": 12.1,
                "mean_worst_window_dip": 0.19,
                "mean_worst_window_recovery": 1.2,
            },
            "FedDrift": {
                "mean_final_accuracy": 0.66,
                "mean_accuracy_auc": 12.2,
                "mean_worst_window_dip": 0.22,
                "mean_worst_window_recovery": 1.0,
            },
        },
    }


def _setup_results_dir(tmp_path: Path, with_budget: bool = True,
                        with_alpha: bool = False,
                        notes: list[str] | None = None) -> Path:
    """Create a minimal results directory structure."""
    results_dir = tmp_path / "results"
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True)
    (results_dir / "figures" / "ablations").mkdir(parents=True)

    claim_check = _make_claim_check(notes=notes)
    (logs_dir / "claim_check.json").write_text(
        json.dumps(claim_check, indent=2), encoding="utf-8",
    )

    if with_budget:
        (logs_dir / "budget_points.json").write_text(
            json.dumps(_make_budget_records(), indent=2), encoding="utf-8",
        )

    if with_alpha:
        (logs_dir / "alpha_diagnostics.json").write_text(
            json.dumps(_make_alpha_diagnostics(), indent=2), encoding="utf-8",
        )

    return results_dir


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_fmt_metric_none(self) -> None:
        assert resume_diagnostics._fmt_metric(None) == "--"

    def test_fmt_metric_value(self) -> None:
        assert resume_diagnostics._fmt_metric(0.12345) == "0.123"

    def test_non_none_mean_all_none(self) -> None:
        assert resume_diagnostics._non_none_mean([None, None]) is None

    def test_non_none_mean_mixed(self) -> None:
        result = resume_diagnostics._non_none_mean([1.0, None, 3.0])
        assert result == pytest.approx(2.0)


class TestBudgetPointsSerialization:
    def test_roundtrip(self, tmp_path: Path) -> None:
        """Budget points survive a JSON roundtrip."""
        original = {
            "FedProTrack": [
                BudgetPoint("FedProTrack", 1, 12000.0, 10.5),
                BudgetPoint("FedProTrack", 5, 2400.0, 10.3),
            ],
            "IFCA": [
                BudgetPoint("IFCA", 1, 9120.0, 10.7),
            ],
        }

        path = tmp_path / "bp.json"
        records = resume_diagnostics.budget_points_to_records(original)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        loaded = resume_diagnostics.load_budget_points_from_json(path)

        assert set(loaded.keys()) == {"FedProTrack", "IFCA"}
        assert len(loaded["FedProTrack"]) == 2
        assert loaded["FedProTrack"][0].total_bytes == 12000.0
        assert loaded["IFCA"][0].accuracy_auc == pytest.approx(10.7)


class TestWriteDiagnosticSummary:
    def test_writes_markdown(self, tmp_path: Path) -> None:
        path = tmp_path / "diag.md"
        claim_check = _make_claim_check()
        budget_points = {
            "FedProTrack": [BudgetPoint("FedProTrack", 1, 12000.0, 10.5)],
        }
        alpha_diag = _make_alpha_diagnostics()
        module_abl = {
            "Full FedProTrack": _dummy_metrics(0.65),
            "No temporal prior": _dummy_metrics(0.40),
        }

        resume_diagnostics.write_diagnostic_summary(
            path, claim_check, budget_points, alpha_diag, module_abl,
        )

        text = path.read_text(encoding="utf-8")
        assert "# Gate Diagnostics" in text
        assert "E1 gate failed" in text
        assert "FedProTrack fe=1" in text
        assert "alpha=0.0" in text
        assert "Full FedProTrack" in text
        assert "No temporal prior" in text

    def test_no_notes_shows_no_failures(self, tmp_path: Path) -> None:
        path = tmp_path / "diag.md"
        claim_check = _make_claim_check(notes=[])
        resume_diagnostics.write_diagnostic_summary(
            path, claim_check, {}, {}, {},
        )
        text = path.read_text(encoding="utf-8")
        assert "No gate failures recorded." in text


# ---------------------------------------------------------------------------
# Integration tests: resume_diagnostics
# ---------------------------------------------------------------------------

class TestResumeDiagnostics:
    def test_missing_claim_check_raises(self, tmp_path: Path) -> None:
        """Error when claim_check.json is absent."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "logs").mkdir()

        with pytest.raises(FileNotFoundError, match="claim_check.json not found"):
            resume_diagnostics.resume_diagnostics(results_dir, seeds=[42])

    def test_empty_notes_raises(self, tmp_path: Path) -> None:
        """Error when notes is empty and force=False."""
        results_dir = _setup_results_dir(tmp_path, notes=[])

        with pytest.raises(ValueError, match="no gate failures"):
            resume_diagnostics.resume_diagnostics(results_dir, seeds=[42])

    def test_empty_notes_with_force(self, tmp_path: Path) -> None:
        """Force flag allows running even with empty notes."""
        results_dir = _setup_results_dir(tmp_path, notes=[], with_alpha=True)

        mock_module_results = {
            "Full FedProTrack": _dummy_metrics(0.65),
        }
        with patch.object(
            resume_diagnostics, "run_module_ablation",
            return_value=mock_module_results,
        ):
            generated = resume_diagnostics.resume_diagnostics(
                results_dir, seeds=[42], force=True,
            )

        assert "diagnostic_summary" in generated
        summary_path = results_dir / "logs" / "diagnostic_summary.md"
        assert summary_path.exists()

    def test_reuses_existing_budget_and_alpha(self, tmp_path: Path) -> None:
        """Existing budget_points.json and alpha_diagnostics.json are reused."""
        results_dir = _setup_results_dir(
            tmp_path, with_budget=True, with_alpha=True,
        )

        mock_module_results = {
            "Full FedProTrack": _dummy_metrics(0.65),
        }
        with patch.object(
            resume_diagnostics, "run_module_ablation",
            return_value=mock_module_results,
        ) as mock_abl:
            generated = resume_diagnostics.resume_diagnostics(
                results_dir, seeds=[42],
            )

        # Budget and alpha were reused, not regenerated
        assert "budget_points" not in generated
        assert "alpha_diagnostics" not in generated
        # But diagnostic summary was written
        assert "diagnostic_summary" in generated
        # Module ablation was called
        mock_abl.assert_called_once()

    def test_generates_missing_alpha(self, tmp_path: Path) -> None:
        """alpha_diagnostics.json is generated when missing."""
        results_dir = _setup_results_dir(
            tmp_path, with_budget=True, with_alpha=False,
        )

        mock_alpha = _make_alpha_diagnostics()
        mock_module_results = {
            "Full FedProTrack": _dummy_metrics(0.65),
        }

        with patch.object(
            resume_diagnostics, "run_alpha_diagnostics",
            return_value=mock_alpha,
        ) as mock_alpha_fn, patch.object(
            resume_diagnostics, "run_module_ablation",
            return_value=mock_module_results,
        ):
            generated = resume_diagnostics.resume_diagnostics(
                results_dir, seeds=[42, 123],
            )

        assert "alpha_diagnostics" in generated
        mock_alpha_fn.assert_called_once_with([42, 123])

        # Verify the file was written
        alpha_path = results_dir / "logs" / "alpha_diagnostics.json"
        assert alpha_path.exists()
        data = json.loads(alpha_path.read_text(encoding="utf-8"))
        assert "alpha=0.0" in data

    def test_generates_missing_budget(self, tmp_path: Path) -> None:
        """budget_points.json is generated when missing."""
        results_dir = _setup_results_dir(
            tmp_path, with_budget=False, with_alpha=True,
        )

        mock_budget = {
            "FedProTrack": [BudgetPoint("FedProTrack", 1, 12000.0, 10.5)],
        }
        mock_module_results = {
            "Full FedProTrack": _dummy_metrics(0.65),
        }

        with patch.object(
            resume_diagnostics, "collect_default_budget_points",
            return_value=mock_budget,
        ) as mock_budget_fn, patch.object(
            resume_diagnostics, "run_module_ablation",
            return_value=mock_module_results,
        ):
            generated = resume_diagnostics.resume_diagnostics(
                results_dir, seeds=[42],
            )

        assert "budget_points" in generated
        mock_budget_fn.assert_called_once()

    def test_force_regenerates_all(self, tmp_path: Path) -> None:
        """--force regenerates all diagnostics even if files exist."""
        results_dir = _setup_results_dir(
            tmp_path, with_budget=True, with_alpha=True,
        )

        mock_alpha = _make_alpha_diagnostics()
        mock_budget = {
            "FedProTrack": [BudgetPoint("FedProTrack", 1, 12000.0, 10.5)],
        }
        mock_module_results = {
            "Full FedProTrack": _dummy_metrics(0.65),
        }

        with patch.object(
            resume_diagnostics, "run_alpha_diagnostics",
            return_value=mock_alpha,
        ) as mock_alpha_fn, patch.object(
            resume_diagnostics, "collect_default_budget_points",
            return_value=mock_budget,
        ) as mock_budget_fn, patch.object(
            resume_diagnostics, "run_module_ablation",
            return_value=mock_module_results,
        ):
            generated = resume_diagnostics.resume_diagnostics(
                results_dir, seeds=[42], force=True,
            )

        assert "alpha_diagnostics" in generated
        assert "budget_points" in generated
        assert "diagnostic_summary" in generated
        mock_alpha_fn.assert_called_once()
        mock_budget_fn.assert_called_once()

    def test_no_overwrite_of_claim_check(self, tmp_path: Path) -> None:
        """claim_check.json is never modified by resume."""
        results_dir = _setup_results_dir(
            tmp_path, with_budget=True, with_alpha=True,
        )

        original_text = (results_dir / "logs" / "claim_check.json").read_text(
            encoding="utf-8",
        )

        mock_module_results = {
            "Full FedProTrack": _dummy_metrics(0.65),
        }
        with patch.object(
            resume_diagnostics, "run_module_ablation",
            return_value=mock_module_results,
        ):
            resume_diagnostics.resume_diagnostics(results_dir, seeds=[42])

        after_text = (results_dir / "logs" / "claim_check.json").read_text(
            encoding="utf-8",
        )
        assert original_text == after_text

    def test_diagnostic_summary_content(self, tmp_path: Path) -> None:
        """Verify diagnostic_summary.md has all expected sections."""
        results_dir = _setup_results_dir(
            tmp_path, with_budget=True, with_alpha=True,
        )

        mock_module_results = {
            "Full FedProTrack": _dummy_metrics(0.65),
            "No temporal prior": _dummy_metrics(0.40),
        }
        with patch.object(
            resume_diagnostics, "run_module_ablation",
            return_value=mock_module_results,
        ):
            resume_diagnostics.resume_diagnostics(results_dir, seeds=[42])

        text = (results_dir / "logs" / "diagnostic_summary.md").read_text(
            encoding="utf-8",
        )
        assert "## Gate Status" in text
        assert "## Default Budget Points" in text
        assert "## Alpha Sweep Diagnostics" in text
        assert "## Module Ablations" in text
