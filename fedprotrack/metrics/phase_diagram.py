"""Phase diagram aggregation for federated concept drift parameter sweeps.

Aggregates metric results across a 2D parameter grid (e.g. rho x delta)
into a PhaseDiagramData object that can be visualized as a heatmap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    from .experiment_log import MetricsResult
except ImportError:
    MetricsResult = None  # type: ignore[assignment,misc]

from ..drift_generator import GeneratorConfig


# ---------------------------------------------------------------------------
# Supported metric names
# ---------------------------------------------------------------------------

_SUPPORTED_METRICS = frozenset(
    [
        "concept_re_id_accuracy",
        "assignment_entropy",
        "wrong_memory_reuse_rate",
        "worst_window_dip",
        "worst_window_recovery",
        "budget_normalized_score",
    ]
)

# Config attributes that are not parameters (never treated as sweep axes)
_CONFIG_NON_PARAM_ATTRS = frozenset(
    ["generator_type", "seed", "output_dir", "K", "T", "n_samples"]
)

# All config attributes that can be sweep axes or fixed params
_CONFIG_PARAM_ATTRS = frozenset(["rho", "alpha", "delta", "K", "T"])


# ---------------------------------------------------------------------------
# Helper: extract metric value from a MetricsResult (or dict/namespace)
# ---------------------------------------------------------------------------


def _get_metric_value(result: Any, metric_name: str) -> float:
    """Extract a named metric from a MetricsResult or compatible object.

    Parameters
    ----------
    result : MetricsResult or SimpleNamespace or dict
        The result object to extract from.
    metric_name : str
        One of the supported metric names.

    Returns
    -------
    value : float
        Metric value, or NaN if not available (None or missing).
    """
    if metric_name not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. "
            f"Supported: {sorted(_SUPPORTED_METRICS)}"
        )

    if isinstance(result, dict):
        val = result.get(metric_name, None)
    else:
        val = getattr(result, metric_name, None)

    if val is None:
        return float("nan")
    return float(val)


# ---------------------------------------------------------------------------
# Helper: extract a config parameter by name
# ---------------------------------------------------------------------------


def _get_config_param(config: GeneratorConfig, param: str) -> Any:
    """Return the value of a named parameter from a GeneratorConfig.

    Parameters
    ----------
    config : GeneratorConfig
    param : str

    Returns
    -------
    value
    """
    return getattr(config, param)


# ---------------------------------------------------------------------------
# PhaseDiagramData
# ---------------------------------------------------------------------------


@dataclass
class PhaseDiagramData:
    """2D phase diagram data for a single metric over a parameter sweep.

    Parameters
    ----------
    row_param : str
        Name of the parameter mapped to rows (e.g. ``"rho"``).
    col_param : str
        Name of the parameter mapped to columns (e.g. ``"delta"``).
    row_values : list of float
        Sorted unique values of ``row_param``.
    col_values : list of float
        Sorted unique values of ``col_param``.
    metric_name : str
        Name of the metric stored (e.g. ``"concept_re_id_accuracy"``).
    values : np.ndarray
        Shape ``(n_rows, n_cols)``.  ``NaN`` for missing cells.
    fixed_params : dict
        Config attributes not in ``{row_param, col_param}`` that are
        constant across all results.
    method_name : str
        Human-readable method identifier.
    """

    row_param: str
    col_param: str
    row_values: list[float]
    col_values: list[float]
    metric_name: str
    values: np.ndarray  # shape (n_rows, n_cols)
    fixed_params: dict
    method_name: str = "unknown"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_npz(self, path: Path | str) -> None:
        """Save values array and metadata to an .npz file.

        A sidecar ``<stem>.json`` file stores the non-array metadata.

        Parameters
        ----------
        path : Path or str
            Destination path (e.g. ``"results/phase.npz"``).
        """
        path = Path(path)
        np.savez(path, values=self.values)

        # Metadata: row/col values need special handling for inf
        meta: dict[str, Any] = {
            "row_param": self.row_param,
            "col_param": self.col_param,
            "row_values": [
                "inf" if math.isinf(v) else v for v in self.row_values
            ],
            "col_values": [
                "inf" if math.isinf(v) else v for v in self.col_values
            ],
            "metric_name": self.metric_name,
            "fixed_params": self.fixed_params,
            "method_name": self.method_name,
        }
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def from_npz(cls, path: Path | str) -> PhaseDiagramData:
        """Load a PhaseDiagramData from an .npz file and sidecar .json.

        Parameters
        ----------
        path : Path or str
            Path to the ``.npz`` file.

        Returns
        -------
        PhaseDiagramData
        """
        path = Path(path)
        arrays = np.load(path)
        values: np.ndarray = arrays["values"]

        meta_path = path.with_suffix(".json")
        with open(meta_path) as fh:
            meta = json.load(fh)

        def _parse_float_list(lst: list) -> list[float]:
            return [float("inf") if v == "inf" else float(v) for v in lst]

        return cls(
            row_param=meta["row_param"],
            col_param=meta["col_param"],
            row_values=_parse_float_list(meta["row_values"]),
            col_values=_parse_float_list(meta["col_values"]),
            metric_name=meta["metric_name"],
            values=values,
            fixed_params=meta["fixed_params"],
            method_name=meta.get("method_name", "unknown"),
        )


# ---------------------------------------------------------------------------
# build_phase_diagram
# ---------------------------------------------------------------------------


def build_phase_diagram(
    results: list[tuple],
    row_param: str,
    col_param: str,
    metric_name: str,
    method_name: str = "unknown",
) -> PhaseDiagramData:
    """Aggregate experiment results into a 2D phase diagram.

    Parameters
    ----------
    results : list of (GeneratorConfig, MetricsResult) tuples
        All results to aggregate.
    row_param : str
        Config attribute name for the row axis (e.g. ``"rho"``).
    col_param : str
        Config attribute name for the column axis (e.g. ``"delta"``).
    metric_name : str
        Which metric to extract.  Must be one of the supported names.
    method_name : str
        Label for the method.

    Returns
    -------
    PhaseDiagramData
    """
    if metric_name not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. "
            f"Supported: {sorted(_SUPPORTED_METRICS)}"
        )

    if not results:
        raise ValueError("results list is empty.")

    # ------------------------------------------------------------------
    # 1. Collect unique sorted axis values
    # ------------------------------------------------------------------

    row_vals_raw: list = []
    col_vals_raw: list = []
    for config, _result in results:
        row_vals_raw.append(_get_config_param(config, row_param))
        col_vals_raw.append(_get_config_param(config, col_param))

    def _sort_key(v: Any) -> float:
        """Sort key that puts inf at the end."""
        if isinstance(v, float) and math.isinf(v):
            return float("inf")
        return float(v)

    row_values: list[float] = sorted(set(row_vals_raw), key=_sort_key)
    col_values: list[float] = sorted(set(col_vals_raw), key=_sort_key)

    row_index = {v: i for i, v in enumerate(row_values)}
    col_index = {v: i for i, v in enumerate(col_values)}

    # ------------------------------------------------------------------
    # 2. Build the values grid
    # ------------------------------------------------------------------

    n_rows = len(row_values)
    n_cols = len(col_values)
    values = np.full((n_rows, n_cols), float("nan"), dtype=np.float64)

    for config, result in results:
        rv = _get_config_param(config, row_param)
        cv = _get_config_param(config, col_param)

        # Locate the correct row/col using math.isinf-safe lookup
        ri = _find_index(row_values, rv)
        ci = _find_index(col_values, cv)

        metric_val = _get_metric_value(result, metric_name)
        values[ri, ci] = metric_val

    # ------------------------------------------------------------------
    # 3. Build fixed_params from config attributes that are not sweep axes
    # ------------------------------------------------------------------

    fixed_params = _build_fixed_params(results, {row_param, col_param})

    return PhaseDiagramData(
        row_param=row_param,
        col_param=col_param,
        row_values=row_values,
        col_values=col_values,
        metric_name=metric_name,
        values=values,
        fixed_params=fixed_params,
        method_name=method_name,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_index(values: list[float], target: Any) -> int:
    """Find the index of *target* in *values*, handling inf correctly.

    Parameters
    ----------
    values : list of float
    target : float

    Returns
    -------
    index : int
    """
    for i, v in enumerate(values):
        if _float_equal(v, target):
            return i
    raise KeyError(f"Value {target!r} not found in {values}")


def _float_equal(a: Any, b: Any) -> bool:
    """Compare two floats with special handling for inf.

    Parameters
    ----------
    a, b : float

    Returns
    -------
    bool
    """
    try:
        if math.isinf(a) and math.isinf(b):
            return True
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def _build_fixed_params(
    results: list[tuple],
    sweep_axes: set[str],
) -> dict:
    """Build fixed_params from config attributes not in sweep_axes.

    For each attribute in ``_CONFIG_PARAM_ATTRS`` that is not a sweep axis,
    check whether all configs have the same value.  If so, include it.

    Parameters
    ----------
    results : list of (GeneratorConfig, ...) tuples
    sweep_axes : set of str
        Names of the axes being swept.

    Returns
    -------
    fixed_params : dict
    """
    fixed: dict[str, Any] = {}
    candidate_attrs = _CONFIG_PARAM_ATTRS | _CONFIG_NON_PARAM_ATTRS
    for attr in sorted(candidate_attrs - sweep_axes):
        vals = []
        for config, _ in results:
            try:
                v = getattr(config, attr)
            except AttributeError:
                continue
            vals.append(v)
        if not vals:
            continue
        # Check all values are the same (inf-safe)
        first = vals[0]
        all_same = all(_float_equal(v, first) for v in vals[1:])
        if all_same:
            # Store inf as the string "inf" for JSON-serializability
            fixed[attr] = "inf" if isinstance(first, float) and math.isinf(first) else first
    return fixed


# ---------------------------------------------------------------------------
# MetricsResult reconstruction from dict (lightweight)
# ---------------------------------------------------------------------------


def _metrics_result_from_dict(d: dict) -> SimpleNamespace:
    """Reconstruct a MetricsResult-compatible namespace from a plain dict.

    Converts ``per_client_re_id`` and ``per_timestep_re_id`` back to
    ``np.ndarray``.

    Parameters
    ----------
    d : dict
        As produced by ``MetricsResult.to_dict()``.

    Returns
    -------
    SimpleNamespace
        Has the same attribute interface as MetricsResult.
    """
    obj = SimpleNamespace(**d)
    for arr_key in ("per_client_re_id", "per_timestep_re_id"):
        raw = getattr(obj, arr_key, None)
        if raw is not None and not isinstance(raw, np.ndarray):
            setattr(obj, arr_key, np.array(raw, dtype=np.float64))
        elif raw is None:
            setattr(obj, arr_key, np.array([], dtype=np.float64))
    return obj


# ---------------------------------------------------------------------------
# load_results_from_dir
# ---------------------------------------------------------------------------


def load_results_from_dir(
    results_dir: Path | str,
    method_name: str = "unknown",
) -> list[tuple]:
    """Scan a directory tree for experiment results and load them.

    Each qualifying subdirectory must contain both ``config.json`` and
    ``metrics.json``.

    Parameters
    ----------
    results_dir : Path or str
        Root directory to scan.
    method_name : str
        Label to attach to results (unused internally, returned for
        downstream use).

    Returns
    -------
    pairs : list of (GeneratorConfig, SimpleNamespace)
        Each pair is ``(config, metrics_result)`` where ``metrics_result``
        is a SimpleNamespace with the same interface as MetricsResult.
    """
    results_dir = Path(results_dir)
    pairs: list[tuple] = []

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        config_path = subdir / "config.json"
        metrics_path = subdir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue

        try:
            config = GeneratorConfig.from_json(config_path)
        except Exception:
            continue

        try:
            with open(metrics_path) as fh:
                metrics_dict = json.load(fh)
        except Exception:
            continue

        # Prefer the real MetricsResult.from_dict if available
        if MetricsResult is not None and hasattr(MetricsResult, "from_dict"):
            try:
                metrics_result = MetricsResult.from_dict(metrics_dict)
            except Exception:
                metrics_result = _metrics_result_from_dict(metrics_dict)
        else:
            metrics_result = _metrics_result_from_dict(metrics_dict)

        pairs.append((config, metrics_result))

    return pairs
