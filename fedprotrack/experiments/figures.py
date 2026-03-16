"""Publication-quality figure generation for Phase 3 experiments.

Delegates to existing visualization utilities and adds experiment-level
orchestration (phase diagrams, accuracy curves, concept matrix galleries).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..metrics.experiment_log import MetricsResult
from ..metrics.phase_diagram import PhaseDiagramData
from ..metrics.visualization import (
    plot_budget_frontier,
    plot_metric_comparison,
    plot_phase_diagram,
)


def generate_phase_diagrams(
    grid_results: dict[tuple[float, float], dict[str, MetricsResult]],
    row_param: str,
    col_param: str,
    output_dir: Path | str,
    methods: list[str] | None = None,
    metrics: list[str] | None = None,
) -> list[Path]:
    """Generate phase diagram heatmaps for each method × metric combination.

    Parameters
    ----------
    grid_results : dict[(row_val, col_val), dict[method_name, MetricsResult]]
        Results indexed by (row_param_value, col_param_value).
    row_param : str
        Name of the row axis (e.g. "rho").
    col_param : str
        Name of the column axis (e.g. "delta").
    output_dir : Path or str
    methods : list[str], optional
        Methods to include. Defaults to all.
    metrics : list[str], optional
        Metrics to plot. Defaults to common subset.

    Returns
    -------
    list[Path]
        Paths to saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics is None:
        metrics = [
            "concept_re_id_accuracy",
            "assignment_entropy",
            "wrong_memory_reuse_rate",
        ]

    # Extract unique row/col values
    row_values = sorted({k[0] for k in grid_results})
    col_values = sorted({k[1] for k in grid_results})

    # Discover methods from first entry
    sample = next(iter(grid_results.values()))
    if methods is None:
        methods = list(sample.keys())

    saved: list[Path] = []

    for method_name in methods:
        for metric_name in metrics:
            values = np.full((len(row_values), len(col_values)), np.nan)
            for ri, rv in enumerate(row_values):
                for ci, cv in enumerate(col_values):
                    key = (rv, cv)
                    if key in grid_results and method_name in grid_results[key]:
                        mr = grid_results[key][method_name]
                        val = getattr(mr, metric_name, None)
                        if val is not None:
                            values[ri, ci] = float(val)

            data = PhaseDiagramData(
                row_param=row_param,
                col_param=col_param,
                row_values=row_values,
                col_values=col_values,
                metric_name=metric_name,
                values=values,
                fixed_params={},
                method_name=method_name,
            )

            fname = f"phase_{method_name}_{metric_name}_{row_param}_vs_{col_param}.png"
            out_path = output_dir / fname
            plot_phase_diagram(data, output_path=out_path)
            plt.close("all")
            saved.append(out_path)

    return saved


def generate_accuracy_curves(
    setting_results: dict[str, np.ndarray],
    output_path: Path | str,
    title: str = "Accuracy Over Time",
) -> Path:
    """Plot mean accuracy over time for multiple methods.

    Parameters
    ----------
    setting_results : dict[str, np.ndarray]
        method_name -> accuracy_matrix of shape (K, T).
    output_path : Path or str
    title : str

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, acc_matrix in setting_results.items():
        mean_acc = acc_matrix.mean(axis=0)
        T = len(mean_acc)
        ax.plot(range(T), mean_acc, marker="o", markersize=3, label=method_name)

    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Mean Accuracy", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_ablation_plot(
    param_name: str,
    param_values: list[float],
    metric_values: dict[str, list[float]],
    output_path: Path | str,
) -> Path:
    """Plot ablation study results for one hyperparameter.

    Parameters
    ----------
    param_name : str
        Name of the ablated parameter (e.g. "omega").
    param_values : list[float]
        Parameter values swept.
    metric_values : dict[str, list[float]]
        metric_name -> list of metric values (same length as param_values).
    output_path : Path or str

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_metrics = len(metric_values)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metric_values.items()):
        ax.plot(param_values, values, "o-", linewidth=1.5, markersize=5)
        ax.set_xlabel(param_name, fontsize=10)
        ax.set_ylabel(metric_name.replace("_", " "), fontsize=10)
        ax.set_title(f"{metric_name.replace('_', ' ')}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Ablation: {param_name}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_scalability_plot(
    x_label: str,
    x_values: list[int],
    method_metrics: dict[str, list[float]],
    output_path: Path | str,
    y_label: str = "Mean Accuracy",
) -> Path:
    """Plot scalability results (K or T vs metric).

    Parameters
    ----------
    x_label : str
        Axis label (e.g. "K (clients)" or "T (timesteps)").
    x_values : list[int]
    method_metrics : dict[str, list[float]]
        method_name -> list of metric values at each x_value.
    output_path : Path or str
    y_label : str

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for method_name, values in method_metrics.items():
        ax.plot(x_values, values, "o-", linewidth=1.5, markersize=5, label=method_name)

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(f"Scalability: {x_label}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# E1: Axis-sweep line plots (re-ID vs delta, wrong-mem vs delta, etc.)
# ---------------------------------------------------------------------------

def generate_axis_sweep_plot(
    axis_values: list[float],
    method_curves: dict[str, list[float]],
    x_label: str,
    y_label: str,
    output_path: Path | str,
    title: str | None = None,
    error_bars: dict[str, list[float]] | None = None,
) -> Path:
    """Plot a metric vs an axis parameter for multiple methods.

    Parameters
    ----------
    axis_values : list[float]
        X-axis values (e.g. delta values).
    method_curves : dict[str, list[float]]
        method_name -> metric values at each axis value.
    x_label : str
    y_label : str
    output_path : Path or str
    title : str, optional
    error_bars : dict[str, list[float]], optional
        method_name -> std at each axis value.

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h"]
    for i, (method_name, values) in enumerate(method_curves.items()):
        marker = markers[i % len(markers)]
        if error_bars and method_name in error_bars:
            ax.errorbar(
                axis_values, values, yerr=error_bars[method_name],
                marker=marker, markersize=5, linewidth=1.5, capsize=3,
                label=method_name,
            )
        else:
            ax.plot(
                axis_values, values,
                marker=marker, markersize=5, linewidth=1.5,
                label=method_name,
            )

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# E3: Difference heatmaps (method A - method B)
# ---------------------------------------------------------------------------

def generate_difference_heatmap(
    grid_results: dict[tuple[float, float], dict[str, MetricsResult]],
    method_a: str,
    method_b: str,
    row_param: str,
    col_param: str,
    metric: str,
    output_path: Path | str,
    title: str | None = None,
) -> Path:
    """Generate a difference heatmap: method_a - method_b for a given metric.

    Parameters
    ----------
    grid_results : dict[(row_val, col_val), dict[method_name, MetricsResult]]
    method_a : str
        Method name (numerator).
    method_b : str
        Method name (denominator).
    row_param : str
    col_param : str
    metric : str
    output_path : Path or str
    title : str, optional

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_values = sorted({k[0] for k in grid_results})
    col_values = sorted({k[1] for k in grid_results})

    diff = np.full((len(row_values), len(col_values)), np.nan)
    for ri, rv in enumerate(row_values):
        for ci, cv in enumerate(col_values):
            key = (rv, cv)
            if key not in grid_results:
                continue
            cell = grid_results[key]
            if method_a in cell and method_b in cell:
                va = getattr(cell[method_a], metric, None)
                vb = getattr(cell[method_b], metric, None)
                if va is not None and vb is not None:
                    diff[ri, ci] = float(va) - float(vb)

    fig, ax = plt.subplots(figsize=(7, 5))
    vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff))) if not np.all(np.isnan(diff)) else 1.0
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_values)))
    ax.set_xticklabels([f"{v}" for v in col_values])
    ax.set_yticks(range(len(row_values)))
    ax.set_yticklabels([f"{v}" for v in row_values])
    ax.set_xlabel(col_param, fontsize=11)
    ax.set_ylabel(row_param, fontsize=11)

    # Annotate cells
    for ri in range(len(row_values)):
        for ci in range(len(col_values)):
            val = diff[ri, ci]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(ci, ri, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8, color=color)

    if title is None:
        metric_short = metric.replace("_", " ")
        title = f"{method_a} - {method_b}: {metric_short}"
    ax.set_title(title, fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# E2: Recovery-vs-axis plot and dip/recovery boxplot
# ---------------------------------------------------------------------------

def generate_dip_recovery_boxplot(
    method_dip_recovery: dict[str, list[tuple[float | None, int | None]]],
    output_path: Path | str,
    title: str = "Drift Dip and Recovery",
) -> Path:
    """Boxplot of worst-window dip and recovery across methods.

    Parameters
    ----------
    method_dip_recovery : dict[str, list[tuple[float|None, int|None]]]
        method_name -> list of (dip, recovery) per setting.
    output_path : Path or str
    title : str

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    methods = list(method_dip_recovery.keys())
    dip_data = []
    recovery_data = []
    for m in methods:
        dips = [d for d, r in method_dip_recovery[m] if d is not None]
        recoveries = [r for d, r in method_dip_recovery[m] if r is not None]
        dip_data.append(dips if dips else [0.0])
        recovery_data.append(recoveries if recoveries else [0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot(dip_data, labels=methods)
    ax1.set_ylabel("Worst Window Dip", fontsize=11)
    ax1.set_title("Dip", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    ax2.boxplot(recovery_data, labels=methods)
    ax2.set_ylabel("Recovery Steps", fontsize=11)
    ax2.set_title("Recovery", fontsize=12)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# E4: Budget x alpha best-method heatmap
# ---------------------------------------------------------------------------

def generate_budget_alpha_heatmap(
    alpha_values: list[float],
    budget_levels: list[str],
    best_method_grid: list[list[str]],
    output_path: Path | str,
    title: str = "Best Method by Budget and Alpha",
) -> Path:
    """Heatmap showing which method wins at each (budget_level, alpha) cell.

    Parameters
    ----------
    alpha_values : list[float]
    budget_levels : list[str]
        Labels for budget axis (e.g. ["fe=1", "fe=2", "fe=5", "fe=10"]).
    best_method_grid : list[list[str]]
        Shape (n_budget, n_alpha). Method name at each cell.
    output_path : Path or str
    title : str

    Returns
    -------
    Path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map method names to integers for coloring
    all_methods = sorted({m for row in best_method_grid for m in row})
    method_to_idx = {m: i for i, m in enumerate(all_methods)}

    grid = np.array([[method_to_idx[m] for m in row] for row in best_method_grid])

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.get_cmap("tab10", len(all_methods))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=-0.5, vmax=len(all_methods) - 0.5)

    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels([f"{a}" for a in alpha_values])
    ax.set_yticks(range(len(budget_levels)))
    ax.set_yticklabels(budget_levels)
    ax.set_xlabel("alpha", fontsize=11)
    ax.set_ylabel("Budget Level", fontsize=11)
    ax.set_title(title, fontsize=12)

    # Annotate cells with method names
    for bi in range(len(budget_levels)):
        for ai in range(len(alpha_values)):
            ax.text(ai, bi, best_method_grid[bi][ai], ha="center", va="center",
                    fontsize=7, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, ticks=range(len(all_methods)), shrink=0.8)
    cbar.ax.set_yticklabels(all_methods, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
