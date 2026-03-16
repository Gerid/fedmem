"""Result visualization for federated concept drift experiments.

Generates publication-quality figures comparing methods across
different experimental conditions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .runner import ExperimentResult


def plot_accuracy_over_time(
    results: list[ExperimentResult],
    title: str = "Prequential Accuracy Over Time",
    save_path: str | Path | None = None,
) -> Figure:
    """Plot mean accuracy across clients at each time step for multiple methods.

    Parameters
    ----------
    results : list of ExperimentResult
        Results from different methods to compare.
    title : str
        Figure title.
    save_path : str or Path, optional
        If provided, save figure to this path.

    Returns
    -------
    Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        mean_acc = r.accuracy_matrix.mean(axis=0)
        T = len(mean_acc)
        ax.plot(range(T), mean_acc, marker="o", markersize=3, label=r.method_name)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_concept_matrix_comparison(
    result: ExperimentResult,
    title: str = "Concept Matrix: True vs Predicted",
    save_path: str | Path | None = None,
) -> Figure:
    """Side-by-side heatmaps of true and predicted concept matrices.

    Parameters
    ----------
    result : ExperimentResult
    title : str
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(result.true_concept_matrix, aspect="auto", cmap="Set3")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Client")
    ax1.set_title("Ground Truth")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(result.predicted_concept_matrix, aspect="auto", cmap="Set3")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Client")
    ax2.set_title(f"Predicted ({result.method_name})")
    plt.colorbar(im2, ax=ax2)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_accuracy_heatmap(
    result: ExperimentResult,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Heatmap of per-client, per-step accuracy.

    Parameters
    ----------
    result : ExperimentResult
    title : str, optional
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    if title is None:
        title = f"Accuracy Heatmap — {result.method_name}"

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(result.accuracy_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Client")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Accuracy")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_method_comparison_bar(
    results: list[ExperimentResult],
    metrics: list[str] | None = None,
    title: str = "Method Comparison",
    save_path: str | Path | None = None,
) -> Figure:
    """Grouped bar chart comparing methods across multiple metrics.

    Parameters
    ----------
    results : list of ExperimentResult
    metrics : list of str, optional
        Which metrics to include. Defaults to main metrics.
    save_path : str or Path, optional

    Returns
    -------
    Figure
    """
    if metrics is None:
        metrics = [
            "mean_accuracy",
            "final_accuracy",
            "concept_tracking_accuracy",
        ]

    method_names = [r.method_name for r in results]
    n_methods = len(method_names)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    for i, r in enumerate(results):
        values = [r.summary.get(m, 0.0) for m in metrics]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=r.method_name)

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_all_figures(
    results: list[ExperimentResult],
    output_dir: str | Path,
    prefix: str = "",
) -> list[Path]:
    """Generate a complete set of figures for a group of results.

    Parameters
    ----------
    results : list of ExperimentResult
    output_dir : str or Path
    prefix : str
        Filename prefix.

    Returns
    -------
    paths : list of Path
        Paths to generated figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Accuracy over time
    p = output_dir / f"{prefix}accuracy_over_time.png"
    plot_accuracy_over_time(results, save_path=p)
    paths.append(p)
    plt.close()

    # 2. Concept matrix comparison for FedProTrack result
    for r in results:
        if r.method_name.lower() in ("fedprotrack", "concept_aware"):
            p = output_dir / f"{prefix}concept_matrix_{r.method_name}.png"
            plot_concept_matrix_comparison(r, save_path=p)
            paths.append(p)
            plt.close()

    # 3. Accuracy heatmap per method
    for r in results:
        p = output_dir / f"{prefix}accuracy_heatmap_{r.method_name}.png"
        plot_accuracy_heatmap(r, save_path=p)
        paths.append(p)
        plt.close()

    # 4. Bar chart comparison
    p = output_dir / f"{prefix}method_comparison.png"
    plot_method_comparison_bar(results, save_path=p)
    paths.append(p)
    plt.close()

    return paths
