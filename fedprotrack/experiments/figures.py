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
