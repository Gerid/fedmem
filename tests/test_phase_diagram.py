"""Tests for fedprotrack.metrics.phase_diagram and visualization."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fedprotrack.drift_generator import GeneratorConfig
from fedprotrack.metrics.phase_diagram import (
    PhaseDiagramData,
    build_phase_diagram,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(rho: float, delta: float, alpha: float = 0.5) -> GeneratorConfig:
    """Return a minimal GeneratorConfig with given sweep parameters."""
    return GeneratorConfig(
        K=5,
        T=4,
        rho=rho,
        alpha=alpha,
        delta=delta,
        generator_type="sine",
        seed=42,
    )


def _make_metrics(
    concept_re_id_accuracy: float = 0.8,
    assignment_entropy: float = 1.2,
    wrong_memory_reuse_rate: float = 0.1,
    worst_window_dip: float | None = -0.05,
    worst_window_recovery: int | None = 2,
    budget_normalized_score: float | None = 0.75,
    n_clients: int = 5,
    n_timesteps: int = 4,
) -> SimpleNamespace:
    """Return a SimpleNamespace with the MetricsResult interface."""
    return SimpleNamespace(
        concept_re_id_accuracy=concept_re_id_accuracy,
        assignment_entropy=assignment_entropy,
        wrong_memory_reuse_rate=wrong_memory_reuse_rate,
        worst_window_dip=worst_window_dip,
        worst_window_recovery=worst_window_recovery,
        budget_normalized_score=budget_normalized_score,
        per_client_re_id=np.full(n_clients, concept_re_id_accuracy),
        per_timestep_re_id=np.full(n_timesteps, concept_re_id_accuracy),
    )


# ---------------------------------------------------------------------------
# test_build_2x2_grid
# ---------------------------------------------------------------------------


def test_build_2x2_grid():
    """Build a 2×2 phase diagram and verify shape and values are non-NaN."""
    rho_values = [2.0, 5.0]
    delta_values = [0.3, 0.7]

    pairs = [
        (_make_config(rho=r, delta=d), _make_metrics(concept_re_id_accuracy=r * d))
        for r in rho_values
        for d in delta_values
    ]

    diagram = build_phase_diagram(
        results=pairs,
        row_param="rho",
        col_param="delta",
        metric_name="concept_re_id_accuracy",
        method_name="test_method",
    )

    assert diagram.values.shape == (2, 2), (
        f"Expected shape (2, 2), got {diagram.values.shape}"
    )
    assert not np.any(np.isnan(diagram.values)), "No cell should be NaN in a complete grid"

    # Verify exact numeric values
    for ri, r in enumerate(sorted(rho_values)):
        for ci, d in enumerate(sorted(delta_values)):
            expected = r * d
            actual = diagram.values[ri, ci]
            assert abs(actual - expected) < 1e-9, (
                f"At rho={r}, delta={d}: expected {expected}, got {actual}"
            )

    # Axis bookkeeping
    assert diagram.row_param == "rho"
    assert diagram.col_param == "delta"
    assert diagram.row_values == sorted(rho_values)
    assert diagram.col_values == sorted(delta_values)
    assert diagram.method_name == "test_method"


# ---------------------------------------------------------------------------
# test_serialization_roundtrip
# ---------------------------------------------------------------------------


def test_serialization_roundtrip(tmp_path: Path):
    """Save PhaseDiagramData to .npz + .json and reload; values must match."""
    pairs = [
        (_make_config(rho=2.0, delta=0.3), _make_metrics(0.55)),
        (_make_config(rho=2.0, delta=0.7), _make_metrics(0.65)),
        (_make_config(rho=5.0, delta=0.3), _make_metrics(0.75)),
        (_make_config(rho=5.0, delta=0.7), _make_metrics(0.85)),
    ]

    original = build_phase_diagram(
        results=pairs,
        row_param="rho",
        col_param="delta",
        metric_name="concept_re_id_accuracy",
        method_name="roundtrip_method",
    )

    save_path = tmp_path / "diagram.npz"
    original.to_npz(save_path)
    loaded = PhaseDiagramData.from_npz(save_path)

    np.testing.assert_array_almost_equal(
        original.values,
        loaded.values,
        decimal=10,
        err_msg="Loaded values do not match original after roundtrip",
    )
    assert loaded.row_param == original.row_param
    assert loaded.col_param == original.col_param
    assert loaded.metric_name == original.metric_name
    assert loaded.method_name == original.method_name
    assert loaded.row_values == original.row_values
    assert loaded.col_values == original.col_values
    assert loaded.fixed_params == original.fixed_params


# ---------------------------------------------------------------------------
# test_missing_cell_is_nan
# ---------------------------------------------------------------------------


def test_missing_cell_is_nan():
    """Provide only 3 of 4 grid points; the missing cell should be NaN."""
    # Leave out rho=5, delta=0.7
    pairs = [
        (_make_config(rho=2.0, delta=0.3), _make_metrics(0.1)),
        (_make_config(rho=2.0, delta=0.7), _make_metrics(0.2)),
        (_make_config(rho=5.0, delta=0.3), _make_metrics(0.3)),
        # rho=5.0, delta=0.7 is intentionally missing
    ]

    diagram = build_phase_diagram(
        results=pairs,
        row_param="rho",
        col_param="delta",
        metric_name="concept_re_id_accuracy",
    )

    assert diagram.values.shape == (2, 2)

    # Locate the missing cell: rho=5 is row_idx 1, delta=0.7 is col_idx 1
    rho_idx = diagram.row_values.index(5.0)
    delta_idx = diagram.col_values.index(0.7)
    assert math.isnan(diagram.values[rho_idx, delta_idx]), (
        "Missing cell should be NaN"
    )

    # All other cells should be non-NaN
    present_cells = [
        (diagram.row_values.index(2.0), diagram.col_values.index(0.3)),
        (diagram.row_values.index(2.0), diagram.col_values.index(0.7)),
        (diagram.row_values.index(5.0), diagram.col_values.index(0.3)),
    ]
    for ri, ci in present_cells:
        assert not math.isnan(diagram.values[ri, ci]), (
            f"Cell ({ri},{ci}) should not be NaN"
        )


# ---------------------------------------------------------------------------
# test_metric_name_extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "metric_name,value",
    [
        ("concept_re_id_accuracy", 0.80),
        ("assignment_entropy", 1.50),
        ("wrong_memory_reuse_rate", 0.05),
        ("worst_window_dip", -0.10),
        ("worst_window_recovery", 3.0),
        ("budget_normalized_score", 0.90),
    ],
)
def test_metric_name_extraction(metric_name: str, value: float):
    """Each of the 6 supported metric names can be extracted correctly."""
    metrics_kwargs: dict = {
        "concept_re_id_accuracy": 0.80,
        "assignment_entropy": 1.50,
        "wrong_memory_reuse_rate": 0.05,
        "worst_window_dip": -0.10,
        "worst_window_recovery": 3,
        "budget_normalized_score": 0.90,
    }
    # Override only the one we're testing
    metrics_kwargs[metric_name] = value

    pairs = [
        (_make_config(rho=2.0, delta=0.5), _make_metrics(**metrics_kwargs)),
    ]

    diagram = build_phase_diagram(
        results=pairs,
        row_param="rho",
        col_param="delta",
        metric_name=metric_name,
    )

    assert diagram.values.shape == (1, 1)
    assert not math.isnan(diagram.values[0, 0]), (
        f"Value should not be NaN for metric '{metric_name}'"
    )
    assert abs(diagram.values[0, 0] - float(value)) < 1e-9, (
        f"Expected {value}, got {diagram.values[0, 0]} for '{metric_name}'"
    )


# ---------------------------------------------------------------------------
# test_rho_inf_handling
# ---------------------------------------------------------------------------


def test_rho_inf_handling():
    """Phase diagram handles rho=inf without errors and places it correctly."""
    pairs = [
        (_make_config(rho=2.0, delta=0.5), _make_metrics(0.60)),
        (_make_config(rho=5.0, delta=0.5), _make_metrics(0.70)),
        (_make_config(rho=float("inf"), delta=0.5), _make_metrics(0.80)),
    ]

    diagram = build_phase_diagram(
        results=pairs,
        row_param="rho",
        col_param="delta",
        metric_name="concept_re_id_accuracy",
        method_name="inf_test",
    )

    assert diagram.values.shape == (3, 1)

    # inf should appear in row_values and be the last entry (sorted to end)
    assert math.isinf(diagram.row_values[-1]), (
        f"Last row_value should be inf, got {diagram.row_values[-1]}"
    )

    # The inf row should have value 0.80
    inf_row_idx = len(diagram.row_values) - 1
    assert abs(diagram.values[inf_row_idx, 0] - 0.80) < 1e-9

    # No NaN cells
    assert not np.any(np.isnan(diagram.values)), "All cells should be filled"

    # Serialization round-trip with inf in row_values
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        save_path = Path(td) / "inf_diagram.npz"
        diagram.to_npz(save_path)
        loaded = PhaseDiagramData.from_npz(save_path)
        assert math.isinf(loaded.row_values[-1]), (
            "inf should survive serialization roundtrip"
        )
        np.testing.assert_array_almost_equal(diagram.values, loaded.values)


# ---------------------------------------------------------------------------
# test_unsupported_metric_raises
# ---------------------------------------------------------------------------


def test_unsupported_metric_raises():
    """build_phase_diagram raises ValueError for unsupported metric names."""
    pairs = [
        (_make_config(rho=2.0, delta=0.5), _make_metrics()),
    ]
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_phase_diagram(
            results=pairs,
            row_param="rho",
            col_param="delta",
            metric_name="nonexistent_metric",
        )


# ---------------------------------------------------------------------------
# test_none_metric_becomes_nan
# ---------------------------------------------------------------------------


def test_none_metric_becomes_nan():
    """Optional metrics that are None should be stored as NaN."""
    pairs = [
        (
            _make_config(rho=2.0, delta=0.5),
            _make_metrics(worst_window_dip=None, worst_window_recovery=None, budget_normalized_score=None),
        ),
    ]

    for metric_name in ("worst_window_dip", "worst_window_recovery", "budget_normalized_score"):
        diagram = build_phase_diagram(
            results=pairs,
            row_param="rho",
            col_param="delta",
            metric_name=metric_name,
        )
        assert math.isnan(diagram.values[0, 0]), (
            f"None value for '{metric_name}' should become NaN"
        )


# ---------------------------------------------------------------------------
# test_visualization_smoke (import and run without error)
# ---------------------------------------------------------------------------


def test_visualization_smoke():
    """plot_phase_diagram and plot_metric_comparison run without errors."""
    import matplotlib
    matplotlib.use("Agg")

    from fedprotrack.metrics.visualization import (
        plot_metric_comparison,
        plot_phase_diagram,
    )

    pairs = [
        (_make_config(rho=2.0, delta=0.3), _make_metrics(0.55)),
        (_make_config(rho=2.0, delta=0.7), _make_metrics(0.65)),
        (_make_config(rho=5.0, delta=0.3), _make_metrics(0.75)),
        (_make_config(rho=5.0, delta=0.7), _make_metrics(0.85)),
    ]

    diagram = build_phase_diagram(
        results=pairs,
        row_param="rho",
        col_param="delta",
        metric_name="concept_re_id_accuracy",
        method_name="smoke_method",
    )

    fig = plot_phase_diagram(diagram)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)

    method_results = {
        "MethodA": _make_metrics(0.80),
        "MethodB": _make_metrics(0.60),
    }
    fig2 = plot_metric_comparison(method_results)
    assert fig2 is not None
    plt.close(fig2)
