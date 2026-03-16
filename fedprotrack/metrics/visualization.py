"""Visualization utilities for phase diagrams and method comparisons.

Produces publication-quality matplotlib figures.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

from .phase_diagram import PhaseDiagramData


# ---------------------------------------------------------------------------
# plot_budget_frontier
# ---------------------------------------------------------------------------


def plot_budget_frontier(
    method_points: dict[str, list[Any]],
    output_path: Path | str | None = None,
    title: str = "Budget Frontier",
) -> matplotlib.figure.Figure:
    """Scatter + line plot of accuracy AUC vs communication bytes per method.

    Parameters
    ----------
    method_points : dict[str, list[BudgetPoint]]
        Maps method name to a list of ``BudgetPoint`` objects, each having
        ``total_bytes`` and ``accuracy_auc`` attributes.
    output_path : Path or str, optional
        If provided, save the figure here.
    title : str
        Figure title. Default ``"Budget Frontier"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c["color"] for c in prop_cycle]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for i, (method_name, points) in enumerate(method_points.items()):
        if not points:
            continue
        # Sort by total_bytes for a clean line plot
        sorted_pts = sorted(points, key=lambda p: p.total_bytes)
        xs = [p.total_bytes for p in sorted_pts]
        ys = [p.accuracy_auc for p in sorted_pts]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=1.8,
                markersize=7, label=method_name, zorder=3)

        # Annotate federation_every values if present
        for p in sorted_pts:
            if hasattr(p, "federation_every"):
                ax.annotate(
                    f"fe={p.federation_every}",
                    (p.total_bytes, p.accuracy_auc),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                )

    ax.set_xlabel("Total Communication (bytes)", fontsize=11)
    ax.set_ylabel("Accuracy AUC", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method", fontsize=9, title_fontsize=9)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# plot_phase_diagram
# ---------------------------------------------------------------------------


def plot_phase_diagram(
    data: PhaseDiagramData,
    output_path: Path | str | None = None,
    cmap: str = "RdYlGn",
    annotate: bool = True,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> matplotlib.figure.Figure:
    """Render a phase diagram as a heatmap.

    Parameters
    ----------
    data : PhaseDiagramData
        Aggregated phase diagram data.
    output_path : Path or str, optional
        If provided, save the figure here.
    cmap : str
        Matplotlib colormap name.  Default ``"RdYlGn"``.
    annotate : bool
        If True, write numeric values inside each cell.
    title : str, optional
        Figure title.  Auto-generated if not provided.
    vmin : float, optional
        Minimum value for colormap normalization.
    vmax : float, optional
        Maximum value for colormap normalization.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_rows = len(data.row_values)
    n_cols = len(data.col_values)

    # Scale figure to grid size (minimum 4 inches per side)
    fig_w = max(5.0, n_cols * 1.2 + 2.0)
    fig_h = max(4.0, n_rows * 1.2 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # --- build display array (NaN cells shown as gray) ---
    valid_mask = ~np.isnan(data.values)
    display = data.values.copy()

    # Determine color range from valid cells
    if vmin is None:
        vmin_eff = float(np.nanmin(data.values)) if valid_mask.any() else 0.0
    else:
        vmin_eff = vmin
    if vmax is None:
        vmax_eff = float(np.nanmax(data.values)) if valid_mask.any() else 1.0
    else:
        vmax_eff = vmax

    # Fill NaN with a sentinel below vmin for background rendering;
    # we'll overlay gray patches separately.
    sentinel = vmin_eff - 1.0
    display_filled = np.where(np.isnan(display), sentinel, display)

    # --- imshow ---
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_under("lightgray")  # color for sentinel / NaN cells

    im = ax.imshow(
        display_filled,
        cmap=colormap,
        interpolation="nearest",
        aspect="auto",
        vmin=vmin_eff,
        vmax=vmax_eff,
        origin="upper",
    )

    # --- Hatch NaN cells ---
    for r in range(n_rows):
        for c in range(n_cols):
            if np.isnan(data.values[r, c]):
                ax.add_patch(
                    plt.Rectangle(
                        (c - 0.5, r - 0.5),
                        1.0,
                        1.0,
                        fill=True,
                        facecolor="lightgray",
                        edgecolor="gray",
                        hatch="////",
                        linewidth=0.5,
                    )
                )

    # --- Cell annotations ---
    if annotate:
        color_range = vmax_eff - vmin_eff if vmax_eff != vmin_eff else 1.0
        for r in range(n_rows):
            for c in range(n_cols):
                cell_val = data.values[r, c]
                if np.isnan(cell_val):
                    ax.text(
                        c,
                        r,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="gray",
                    )
                else:
                    # Choose text color based on relative brightness
                    normalized = (cell_val - vmin_eff) / color_range
                    text_color = "white" if normalized < 0.45 or normalized > 0.85 else "black"
                    ax.text(
                        c,
                        r,
                        f"{cell_val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color,
                        fontweight="bold",
                    )

    # --- Tick labels ---
    def _fmt(v: float) -> str:
        if math.isinf(v):
            return "\u221e"  # Unicode infinity
        # Use 1 decimal place for floats, integer for whole numbers
        if v == int(v):
            return str(int(v))
        return f"{v:.1f}"

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([_fmt(v) for v in data.col_values])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_fmt(v) for v in data.row_values])

    ax.set_xlabel(data.col_param, fontsize=11)
    ax.set_ylabel(data.row_param, fontsize=11)

    # --- Colorbar ---
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(data.metric_name.replace("_", " "), fontsize=10)

    # --- Title ---
    if title is None:
        title = (
            f"{data.method_name}: {data.metric_name.replace('_', ' ')} "
            f"vs {data.row_param} \u00d7 {data.col_param}"
        )
    ax.set_title(title, fontsize=12, pad=10)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# plot_metric_comparison
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    method_results: dict[str, Any],  # method_name -> MetricsResult
    output_path: Path | str | None = None,
    metrics_to_show: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """Grouped bar chart comparing methods across multiple metrics.

    Parameters
    ----------
    method_results : dict mapping method_name -> MetricsResult
        Results for each method.
    output_path : Path or str, optional
        If provided, save figure here.
    metrics_to_show : list of str, optional
        Which metrics to display.  Defaults to
        ``["concept_re_id_accuracy", "wrong_memory_reuse_rate",
        "assignment_entropy"]``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if metrics_to_show is None:
        metrics_to_show = [
            "concept_re_id_accuracy",
            "wrong_memory_reuse_rate",
            "assignment_entropy",
        ]

    method_names = list(method_results.keys())
    n_methods = len(method_names)
    n_metrics = len(metrics_to_show)

    # Use the default matplotlib color cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c["color"] for c in prop_cycle]
    # Ensure enough colors
    while len(colors) < n_methods:
        colors = colors + colors

    fig, ax = plt.subplots(figsize=(max(7, n_metrics * 2.0 + 1.5), 5))

    x = np.arange(n_metrics)
    width = 0.8 / max(n_methods, 1)

    for i, method_name in enumerate(method_names):
        result = method_results[method_name]
        values: list[float] = []
        for metric in metrics_to_show:
            if isinstance(result, dict):
                raw = result.get(metric, None)
            else:
                raw = getattr(result, metric, None)
            values.append(float(raw) if raw is not None else 0.0)

        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=method_name,
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )

        # Value labels on bars
        for bar, val in zip(bars, values):
            bar_height = bar.get_height()
            if bar_height > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar_height + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=0,
                )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Method Comparison", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metrics_to_show],
        fontsize=9,
    )
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Method", fontsize=9, title_fontsize=9)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
