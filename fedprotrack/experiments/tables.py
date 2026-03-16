"""LaTeX table generation for Phase 3 experiment results.

Produces publication-ready LaTeX tables comparing methods across metrics,
parameter axes, and generators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..metrics.experiment_log import MetricsResult

# Metrics displayed in tables (order matches paper)
METRIC_NAMES = [
    "concept_re_id_accuracy",
    "assignment_entropy",
    "wrong_memory_reuse_rate",
    "budget_normalized_score",
]

METRIC_LABELS = {
    "concept_re_id_accuracy": "Re-ID Acc",
    "assignment_entropy": "Assign Ent",
    "wrong_memory_reuse_rate": "Wrong Mem",
    "budget_normalized_score": "Budget Score",
}


def _get_metric(result: MetricsResult, metric: str) -> float | None:
    """Extract a metric value from MetricsResult."""
    val = getattr(result, metric, None)
    if val is None:
        return None
    return float(val)


def _fmt(val: float | None) -> str:
    """Format a float for LaTeX."""
    if val is None:
        return "--"
    return f"{val:.3f}"


def _fmt_pm(values: list[float]) -> str:
    """Format mean ± std for LaTeX."""
    if not values:
        return "--"
    mean = np.mean(values)
    if len(values) == 1:
        return f"{mean:.3f}"
    std = np.std(values)
    return f"{mean:.3f}$\\pm${std:.3f}"


def generate_main_table(
    all_results: dict[str, list[MetricsResult]],
    output_path: Path | str | None = None,
    metrics: list[str] | None = None,
) -> str:
    """Generate the main comparison table: methods × metrics.

    Parameters
    ----------
    all_results : dict[str, list[MetricsResult]]
        method_name -> list of MetricsResult (one per setting/seed).
    output_path : Path or str, optional
        If provided, write the LaTeX string to this file.
    metrics : list[str], optional
        Which metrics to include. Defaults to METRIC_NAMES.

    Returns
    -------
    str
        LaTeX table source.
    """
    if metrics is None:
        metrics = METRIC_NAMES

    method_names = list(all_results.keys())
    n_metrics = len(metrics)

    # Header
    cols = "l" + "c" * n_metrics
    header_cells = " & ".join(
        METRIC_LABELS.get(m, m.replace("_", " ")) for m in metrics
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main comparison across all settings (mean $\pm$ std).}",
        r"\label{tab:main}",
        f"\\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        f"Method & {header_cells} \\\\",
        r"\midrule",
    ]

    # Find best method per metric (highest re-id, lowest entropy/wrong-mem, highest budget)
    higher_is_better = {
        "concept_re_id_accuracy": True,
        "assignment_entropy": False,
        "wrong_memory_reuse_rate": False,
        "budget_normalized_score": True,
    }

    # Compute means per method per metric
    method_means: dict[str, dict[str, float]] = {}
    for method_name in method_names:
        method_means[method_name] = {}
        for metric in metrics:
            vals = [
                _get_metric(r, metric)
                for r in all_results[method_name]
                if _get_metric(r, metric) is not None
            ]
            method_means[method_name][metric] = np.mean(vals) if vals else float("nan")

    # Find best per metric
    best_per_metric: dict[str, str] = {}
    for metric in metrics:
        hib = higher_is_better.get(metric, True)
        best_val = float("-inf") if hib else float("inf")
        best_method = ""
        for method_name in method_names:
            v = method_means[method_name][metric]
            if np.isnan(v):
                continue
            if (hib and v > best_val) or (not hib and v < best_val):
                best_val = v
                best_method = method_name
        best_per_metric[metric] = best_method

    # Rows
    for method_name in method_names:
        results = all_results[method_name]
        cells = []
        for metric in metrics:
            vals = [
                _get_metric(r, metric)
                for r in results
                if _get_metric(r, metric) is not None
            ]
            text = _fmt_pm(vals)
            if best_per_metric.get(metric) == method_name:
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        row = " & ".join([method_name.replace("_", r"\_")] + cells)
        lines.append(f"{row} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex, encoding="utf-8")

    return latex


def generate_per_axis_table(
    axis_results: dict[float, dict[str, list[MetricsResult]]],
    axis_name: str,
    output_path: Path | str | None = None,
    metric: str = "concept_re_id_accuracy",
) -> str:
    """Generate a breakdown table: axis values × methods.

    Parameters
    ----------
    axis_results : dict[float, dict[str, list[MetricsResult]]]
        axis_value -> method_name -> list of MetricsResult.
    axis_name : str
        Name of the axis (e.g. "rho", "alpha", "delta").
    output_path : Path or str, optional
    metric : str
        Which metric to display.

    Returns
    -------
    str
        LaTeX table source.
    """
    axis_values = sorted(axis_results.keys())
    method_names = list(next(iter(axis_results.values())).keys())

    cols = "l" + "c" * len(axis_values)
    header_cells = " & ".join(f"${axis_name}={v}$" for v in axis_values)

    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{metric_label} by ${axis_name}$.}}",
        f"\\label{{tab:{axis_name}_{metric}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        f"Method & {header_cells} \\\\",
        r"\midrule",
    ]

    for method_name in method_names:
        cells = []
        for av in axis_values:
            results = axis_results[av].get(method_name, [])
            vals = [
                _get_metric(r, metric)
                for r in results
                if _get_metric(r, metric) is not None
            ]
            cells.append(_fmt_pm(vals))
        row = " & ".join([method_name.replace("_", r"\_")] + cells)
        lines.append(f"{row} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex, encoding="utf-8")

    return latex
