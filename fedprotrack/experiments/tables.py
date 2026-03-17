"""LaTeX table generation for Phase 3 experiment results.

Produces publication-ready LaTeX tables comparing methods across metrics,
parameter axes, and generators.  Includes ranking helpers (mean rank, win rate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..metrics.experiment_log import MetricsResult

# Metrics displayed in tables (order matches paper)
METRIC_NAMES = [
    "final_accuracy",
    "accuracy_auc",
    "concept_re_id_accuracy",
    "wrong_memory_reuse_rate",
    "budget_normalized_score",
]

METRIC_LABELS = {
    "concept_re_id_accuracy": "Re-ID Acc",
    "assignment_entropy": "Assign Ent",
    "wrong_memory_reuse_rate": "Wrong Mem",
    "budget_normalized_score": "Budget Score",
    "final_accuracy": "Final Acc",
    "accuracy_auc": "AUC(acc)",
    "worst_window_dip": "Worst Dip",
    "worst_window_recovery": "Recovery",
    "mean_rank": "Mean Rank",
    "win_rate": "Win Rate",
}

# Direction: True = higher is better, False = lower is better
HIGHER_IS_BETTER = {
    "concept_re_id_accuracy": True,
    "assignment_entropy": False,
    "wrong_memory_reuse_rate": False,
    "budget_normalized_score": True,
    "final_accuracy": True,
    "accuracy_auc": True,
    "worst_window_dip": False,
    "worst_window_recovery": False,
    "mean_rank": False,
    "win_rate": True,
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
    """Format mean +/- std for LaTeX."""
    if not values:
        return "--"
    mean = np.mean(values)
    if len(values) == 1:
        return f"{mean:.3f}"
    std = np.std(values)
    return f"{mean:.3f}$\\pm${std:.3f}"


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------

def compute_rankings(
    all_results: dict[str, list[MetricsResult]],
    metric: str = "final_accuracy",
) -> dict[str, float]:
    """Compute mean rank of each method across settings.

    For each setting index *i*, methods are ranked by metric value (1 = best).
    Returns the average rank per method.

    Parameters
    ----------
    all_results : dict[str, list[MetricsResult]]
        method_name -> list of MetricsResult (same length, same order).
    metric : str

    Returns
    -------
    dict[str, float]
        method_name -> mean rank (lower is better).
    """
    methods = list(all_results.keys())
    n_settings = min(len(v) for v in all_results.values()) if all_results else 0
    if n_settings == 0:
        return {m: float("nan") for m in methods}

    hib = HIGHER_IS_BETTER.get(metric, True)
    rank_sums = {m: 0.0 for m in methods}

    for i in range(n_settings):
        vals = []
        for m in methods:
            v = _get_metric(all_results[m][i], metric)
            vals.append(v if v is not None else (float("-inf") if hib else float("inf")))
        # Rank: sort descending if higher-is-better, ascending otherwise
        order = sorted(range(len(methods)), key=lambda j: vals[j], reverse=hib)
        for rank, j in enumerate(order, 1):
            rank_sums[methods[j]] += rank

    return {m: rank_sums[m] / n_settings for m in methods}


def compute_win_rates(
    all_results: dict[str, list[MetricsResult]],
    metric: str = "final_accuracy",
) -> dict[str, float]:
    """Compute win rate of each method across settings.

    Parameters
    ----------
    all_results : dict[str, list[MetricsResult]]
    metric : str

    Returns
    -------
    dict[str, float]
        method_name -> win rate in [0, 1].
    """
    methods = list(all_results.keys())
    n_settings = min(len(v) for v in all_results.values()) if all_results else 0
    if n_settings == 0:
        return {m: 0.0 for m in methods}

    hib = HIGHER_IS_BETTER.get(metric, True)
    wins = {m: 0 for m in methods}

    for i in range(n_settings):
        vals = {}
        for m in methods:
            v = _get_metric(all_results[m][i], metric)
            vals[m] = v if v is not None else (float("-inf") if hib else float("inf"))
        if hib:
            best = max(vals.values())
        else:
            best = min(vals.values())
        for m in methods:
            if vals[m] == best:
                wins[m] += 1

    return {m: wins[m] / n_settings for m in methods}


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_main_table(
    all_results: dict[str, list[MetricsResult]],
    output_path: Path | str | None = None,
    metrics: list[str] | None = None,
) -> str:
    """Generate the main comparison table: methods x metrics + rank + win rate.

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

    # Compute rankings and win rates on the primary metric
    rankings = compute_rankings(all_results, "final_accuracy")
    win_rates = compute_win_rates(all_results, "final_accuracy")

    # Columns: metrics + mean rank + win rate
    display_cols = list(metrics) + ["mean_rank", "win_rate"]
    n_cols = len(display_cols)

    cols = "l" + "c" * n_cols
    header_cells = " & ".join(
        METRIC_LABELS.get(m, m.replace("_", " ")) for m in display_cols
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
        method_means[method_name]["mean_rank"] = rankings.get(method_name, float("nan"))
        method_means[method_name]["win_rate"] = win_rates.get(method_name, 0.0)

    # Find best per column
    best_per_col: dict[str, str] = {}
    for col in display_cols:
        hib = HIGHER_IS_BETTER.get(col, True)
        best_val = float("-inf") if hib else float("inf")
        best_method = ""
        for method_name in method_names:
            v = method_means[method_name][col]
            if np.isnan(v):
                continue
            if (hib and v > best_val) or (not hib and v < best_val):
                best_val = v
                best_method = method_name
        best_per_col[col] = best_method

    # Rows
    for method_name in method_names:
        results = all_results[method_name]
        cells = []
        for col in display_cols:
            if col == "mean_rank":
                text = f"{rankings.get(method_name, float('nan')):.2f}"
            elif col == "win_rate":
                text = f"{win_rates.get(method_name, 0.0):.2f}"
            else:
                vals = [
                    _get_metric(r, col)
                    for r in results
                    if _get_metric(r, col) is not None
                ]
                text = _fmt_pm(vals)
            if best_per_col.get(col) == method_name:
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
    """Generate a breakdown table: axis values x methods.

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


def generate_overhead_table(
    method_stats: dict[str, dict[str, float]],
    output_path: Path | str | None = None,
) -> str:
    """Generate the overhead/reproducibility appendix table.

    Parameters
    ----------
    method_stats : dict[str, dict[str, float]]
        method_name -> {"total_bytes", "phase_a_bytes", "phase_b_bytes",
        "wall_clock_s", "active_concepts", "spawned_concepts",
        "merged_concepts"}.
    output_path : Path or str, optional

    Returns
    -------
    str
        LaTeX table source.
    """
    overhead_cols = [
        "total_bytes", "phase_a_bytes", "phase_b_bytes",
        "wall_clock_s", "active_concepts",
        "spawned_concepts", "merged_concepts", "pruned_concepts",
    ]
    col_labels = {
        "total_bytes": "Total Bytes",
        "phase_a_bytes": "Phase A",
        "phase_b_bytes": "Phase B",
        "wall_clock_s": "Time (s)",
        "active_concepts": "\\# Active",
        "spawned_concepts": "\\# Spawned",
        "merged_concepts": "\\# Merged",
        "pruned_concepts": "\\# Pruned",
    }

    methods = list(method_stats.keys())
    n_cols = len(overhead_cols)
    cols_str = "l" + "r" * n_cols
    header = " & ".join(col_labels.get(c, c) for c in overhead_cols)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Communication overhead and system cost breakdown.}",
        r"\label{tab:overhead}",
        f"\\begin{{tabular}}{{{cols_str}}}",
        r"\toprule",
        f"Method & {header} \\\\",
        r"\midrule",
    ]

    for method in methods:
        stats = method_stats[method]
        cells = []
        for col in overhead_cols:
            val = stats.get(col)
            if val is None:
                cells.append("--")
            elif col == "wall_clock_s":
                cells.append(f"{val:.1f}")
            elif "bytes" in col:
                if val > 1e6:
                    cells.append(f"{val / 1e6:.1f}M")
                elif val > 1e3:
                    cells.append(f"{val / 1e3:.1f}K")
                else:
                    cells.append(f"{val:.0f}")
            else:
                cells.append(f"{val:.1f}")
        row = " & ".join([method.replace("_", r"\_")] + cells)
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


def export_summary_csv(
    all_results: dict[str, list[MetricsResult]],
    output_path: Path | str,
) -> None:
    """Export a CSV summary for post-processing.

    Parameters
    ----------
    all_results : dict[str, list[MetricsResult]]
    output_path : Path or str
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_metrics = [
        "concept_re_id_accuracy", "assignment_entropy",
        "wrong_memory_reuse_rate", "budget_normalized_score",
        "final_accuracy", "accuracy_auc",
        "worst_window_dip", "worst_window_recovery",
    ]

    rankings = compute_rankings(all_results, "final_accuracy")
    win_rates = compute_win_rates(all_results, "final_accuracy")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_settings"] + all_metrics + ["mean_rank", "win_rate"])
        for method_name, results in all_results.items():
            row = [method_name, len(results)]
            for metric in all_metrics:
                vals = [
                    _get_metric(r, metric)
                    for r in results
                    if _get_metric(r, metric) is not None
                ]
                row.append(f"{np.mean(vals):.4f}" if vals else "")
            row.append(f"{rankings.get(method_name, float('nan')):.3f}")
            row.append(f"{win_rates.get(method_name, 0.0):.3f}")
            writer.writerow(row)
