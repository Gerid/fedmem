"""Phase 4 deep analysis: conditional breakdowns, stability, case studies,
ablation on anchor settings, hyperparameter robustness, statistical tests,
and E4 byte breakdown.

All analysis functions consume a pandas-like list-of-dicts (raw per-setting
results saved as CSV) and produce figures + LaTeX tables.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Type alias for raw rows (each row = one setting × one method)
# ---------------------------------------------------------------------------
RawRow = dict[str, str | float | None]


def load_raw_csv(path: Path | str) -> list[RawRow]:
    """Load a per-setting CSV into a list of dicts."""
    path = Path(path)
    rows: list[RawRow] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: RawRow = {}
            for k, v in r.items():
                if v == "" or v is None:
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


# ===================================================================
# 1. E5 Conditional Analysis
# ===================================================================

def _group_by(rows: list[RawRow], key: str) -> dict[str, list[RawRow]]:
    groups: dict[str, list[RawRow]] = {}
    for r in rows:
        k = str(r.get(key, ""))
        groups.setdefault(k, []).append(r)
    return groups


def _metric_stats(rows: list[RawRow], metric: str) -> tuple[float, float, int]:
    """Return (mean, std, count) for a metric from filtered rows."""
    vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def conditional_analysis_e5(
    rows: list[RawRow],
    output_dir: Path,
    methods: list[str] | None = None,
) -> dict:
    """Stratify E5 results by generator × {delta, alpha, rho}.

    Produces:
    - CSV tables with per-stratum mean±std for each method
    - Heatmaps showing FedProTrack advantage over IFCA/FedProto
    - Win-rate matrices

    Returns summary dict for programmatic use.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["FedProTrack", "IFCA", "FedProto", "FedAvg",
                    "TrackedSummary", "FedDrift"]

    axes = ["generator_type", "rho", "alpha", "delta"]
    metrics = ["concept_re_id_accuracy", "final_accuracy",
               "wrong_memory_reuse_rate", "assignment_entropy",
               "accuracy_auc"]

    summary: dict = {}

    # --- Per-axis conditional tables ---
    for axis in axes:
        _write_conditional_table(rows, axis, methods, metrics, output_dir)

    # --- Cross-tabulated analysis: generator × alpha for re-ID ---
    _write_cross_tab(rows, "generator_type", "alpha", methods,
                     "concept_re_id_accuracy", output_dir)

    # --- Win-rate heatmaps: FedProTrack vs IFCA by (rho, delta) per generator ---
    for gen in ["sine", "sea", "circle"]:
        gen_rows = [r for r in rows if r.get("generator_type") == gen]
        if not gen_rows:
            continue
        _plot_advantage_heatmap(
            gen_rows, "FedProTrack", "IFCA",
            "rho", "delta", "concept_re_id_accuracy",
            output_dir / f"advantage_fpt_vs_ifca_{gen}.png",
            title=f"FedProTrack - IFCA Re-ID ({gen.upper()})",
        )
        _plot_advantage_heatmap(
            gen_rows, "FedProTrack", "IFCA",
            "alpha", "delta", "concept_re_id_accuracy",
            output_dir / f"advantage_fpt_vs_ifca_{gen}_alpha_delta.png",
            title=f"FedProTrack - IFCA Re-ID ({gen.upper()}, alpha×delta)",
        )

    # --- Identify where FedProTrack wins/loses ---
    summary["win_conditions"] = _identify_win_conditions(rows, methods)

    # Save summary
    with open(output_dir / "conditional_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def _write_conditional_table(
    rows: list[RawRow],
    axis: str,
    methods: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    """Write a CSV table: axis_value × method → metric mean±std."""
    axis_vals = sorted(set(str(r.get(axis, "")) for r in rows))
    out_path = output_dir / f"conditional_{axis}.csv"

    header = ["axis_value", "method", "n"]
    for m in metrics:
        header.extend([f"{m}_mean", f"{m}_std"])

    csv_rows = []
    for av in axis_vals:
        for method in methods:
            filtered = [r for r in rows
                        if str(r.get(axis, "")) == av
                        and r.get("method") == method]
            if not filtered:
                continue
            row_out: dict[str, str | float] = {
                "axis_value": av, "method": method, "n": len(filtered),
            }
            for m in metrics:
                mean, std, _ = _metric_stats(filtered, m)
                row_out[f"{m}_mean"] = round(mean, 4) if not np.isnan(mean) else ""
                row_out[f"{m}_std"] = round(std, 4) if not np.isnan(std) else ""
            csv_rows.append(row_out)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)


def _write_cross_tab(
    rows: list[RawRow],
    axis1: str,
    axis2: str,
    methods: list[str],
    metric: str,
    output_dir: Path,
) -> None:
    """Write cross-tabulated CSV: axis1 × axis2 × method → metric."""
    v1s = sorted(set(str(r.get(axis1, "")) for r in rows))
    v2s = sorted(set(str(r.get(axis2, "")) for r in rows))

    out_path = output_dir / f"crosstab_{axis1}_{axis2}_{metric}.csv"
    header = [axis1, axis2, "method", "mean", "std", "n"]

    csv_rows = []
    for a1 in v1s:
        for a2 in v2s:
            for method in methods:
                filtered = [r for r in rows
                            if str(r.get(axis1, "")) == a1
                            and str(r.get(axis2, "")) == a2
                            and r.get("method") == method]
                if not filtered:
                    continue
                mean, std, n = _metric_stats(filtered, metric)
                csv_rows.append({
                    axis1: a1, axis2: a2, "method": method,
                    "mean": round(mean, 4) if not np.isnan(mean) else "",
                    "std": round(std, 4) if not np.isnan(std) else "",
                    "n": n,
                })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)


def _plot_advantage_heatmap(
    rows: list[RawRow],
    method_a: str,
    method_b: str,
    row_axis: str,
    col_axis: str,
    metric: str,
    out_path: Path,
    title: str = "",
) -> None:
    """Plot heatmap of method_a - method_b for a metric."""
    row_vals = sorted(set(float(r[row_axis]) for r in rows
                          if r.get(row_axis) is not None))
    col_vals = sorted(set(float(r[col_axis]) for r in rows
                          if r.get(col_axis) is not None))

    grid = np.full((len(row_vals), len(col_vals)), np.nan)

    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            a_vals = [float(r[metric]) for r in rows
                      if r.get("method") == method_a
                      and abs(float(r[row_axis]) - rv) < 1e-6
                      and abs(float(r[col_axis]) - cv) < 1e-6
                      and r.get(metric) is not None]
            b_vals = [float(r[metric]) for r in rows
                      if r.get("method") == method_b
                      and abs(float(r[row_axis]) - rv) < 1e-6
                      and abs(float(r[col_axis]) - cv) < 1e-6
                      and r.get(metric) is not None]
            if a_vals and b_vals:
                grid[i, j] = np.mean(a_vals) - np.mean(b_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.01)
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(col_vals)))
    ax.set_xticklabels([f"{v:.2g}" for v in col_vals])
    ax.set_yticks(range(len(row_vals)))
    ax.set_yticklabels([f"{v:.2g}" for v in row_vals])
    ax.set_xlabel(col_axis)
    ax.set_ylabel(row_axis)
    ax.set_title(title or f"{method_a} - {method_b}: {metric}")
    plt.colorbar(im, ax=ax, label=f"Delta {metric}")

    # Annotate cells
    for i in range(len(row_vals)):
        for j in range(len(col_vals)):
            if not np.isnan(grid[i, j]):
                color = "white" if abs(grid[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, f"{grid[i,j]:+.3f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _identify_win_conditions(
    rows: list[RawRow],
    methods: list[str],
) -> dict:
    """Identify where FedProTrack wins and loses vs each competitor."""
    result: dict = {}
    metric = "concept_re_id_accuracy"
    axes = ["generator_type", "rho", "alpha", "delta"]

    for competitor in ["IFCA", "FedProto", "FedAvg"]:
        wins: list[str] = []
        losses: list[str] = []

        for axis in axes:
            axis_vals = sorted(set(str(r.get(axis, "")) for r in rows))
            for av in axis_vals:
                fpt_rows = [r for r in rows
                            if str(r.get(axis, "")) == av
                            and r.get("method") == "FedProTrack"
                            and r.get(metric) is not None]
                comp_rows = [r for r in rows
                             if str(r.get(axis, "")) == av
                             and r.get("method") == competitor
                             and r.get(metric) is not None]

                if not fpt_rows or not comp_rows:
                    continue

                fpt_mean = np.mean([float(r[metric]) for r in fpt_rows])
                comp_mean = np.mean([float(r[metric]) for r in comp_rows])
                diff = fpt_mean - comp_mean

                if diff > 0.02:
                    wins.append(f"{axis}={av} (+{diff:.3f})")
                elif diff < -0.02:
                    losses.append(f"{axis}={av} ({diff:.3f})")

        result[competitor] = {"wins": wins, "losses": losses}

    return result


# ===================================================================
# 2. E6 Stability Analysis
# ===================================================================

def stability_analysis_e6(
    rows: list[RawRow],
    output_dir: Path,
    methods: list[str] | None = None,
) -> dict:
    """Compute stability metrics for E6: std, worst quartile, recovery.

    Produces:
    - Table: method × {mean, std, Q25, Q75, worst_quartile_mean}
    - Boxplot: per-method distribution of re-ID accuracy
    - Variance ratio plot across conditions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ["FedProTrack", "IFCA", "FedProto", "FedAvg"]

    stability_metrics = [
        "concept_re_id_accuracy", "final_accuracy",
        "worst_window_dip", "worst_window_recovery",
    ]

    table_rows = []
    boxplot_data: dict[str, list[float]] = {}

    for method in methods:
        m_rows = [r for r in rows if r.get("method") == method]
        if not m_rows:
            continue

        entry: dict[str, str | float] = {"method": method, "n": len(m_rows)}

        for metric in stability_metrics:
            vals = [float(r[metric]) for r in m_rows
                    if r.get(metric) is not None]
            if not vals:
                continue
            arr = np.array(vals)
            entry[f"{metric}_mean"] = round(float(np.mean(arr)), 4)
            entry[f"{metric}_std"] = round(float(np.std(arr)), 4)
            entry[f"{metric}_q25"] = round(float(np.percentile(arr, 25)), 4)
            entry[f"{metric}_q75"] = round(float(np.percentile(arr, 75)), 4)
            entry[f"{metric}_worst_q_mean"] = round(
                float(np.mean(arr[arr <= np.percentile(arr, 25)])), 4
            )

            if metric == "concept_re_id_accuracy":
                boxplot_data[method] = vals

        table_rows.append(entry)

    # Write CSV
    if table_rows:
        all_keys = list(table_rows[0].keys())
        for r in table_rows[1:]:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        with open(output_dir / "stability_table.csv", "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in table_rows:
                writer.writerow(r)

    # Boxplot
    if boxplot_data:
        _plot_stability_boxplot(boxplot_data, output_dir / "stability_boxplot.png")

    # Variance comparison across conditions
    _plot_variance_by_condition(rows, methods, output_dir)

    summary = {"methods": {m: e for m, e in zip(
        [r["method"] for r in table_rows], table_rows
    )}}

    with open(output_dir / "stability_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def _plot_stability_boxplot(
    data: dict[str, list[float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(data.keys())
    vals = [data[l] for l in labels]
    bp = ax.boxplot(vals, labels=labels, patch_artist=True, showmeans=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Concept Re-ID Accuracy")
    ax.set_title("E6 Stability: Distribution of Re-ID Accuracy")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_variance_by_condition(
    rows: list[RawRow],
    methods: list[str],
    output_dir: Path,
) -> None:
    """Plot per-method std of re-ID across delta values."""
    metric = "concept_re_id_accuracy"
    delta_vals = sorted(set(float(r["delta"]) for r in rows
                            if r.get("delta") is not None))
    if not delta_vals:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        stds = []
        for dv in delta_vals:
            vals = [float(r[metric]) for r in rows
                    if r.get("method") == method
                    and r.get(metric) is not None
                    and abs(float(r.get("delta", -1)) - dv) < 1e-6]
            stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
        ax.plot(delta_vals, stds, "o-", label=method)

    ax.set_xlabel("delta")
    ax.set_ylabel("Std of Re-ID Accuracy")
    ax.set_title("E6: Variability of Re-ID Accuracy vs Delta")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "variance_vs_delta.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 3. Case Studies: Trajectory Plots
# ===================================================================

def plot_case_study(
    accuracy_matrix: np.ndarray,
    predicted_concepts: np.ndarray,
    true_concepts: np.ndarray,
    assignment_entropy_per_t: np.ndarray | None,
    wrong_memory_per_t: np.ndarray | None,
    method_name: str,
    setting_label: str,
    out_path: Path,
) -> None:
    """Plot a 3-panel case study: concept trajectory, accuracy, entropy."""
    K, T = true_concepts.shape
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Panel 1: Concept assignment trajectory
    ax1 = axes[0]
    for k in range(K):
        ax1.plot(range(T), true_concepts[k], "s", color=f"C{k}",
                 markersize=4, alpha=0.4, label=f"GT k={k}" if k < 3 else None)
        ax1.plot(range(T), predicted_concepts[k], "x", color=f"C{k}",
                 markersize=4, alpha=0.7)
    ax1.set_ylabel("Concept ID")
    ax1.set_title(f"{method_name} | {setting_label}")
    if K <= 5:
        ax1.legend(fontsize=7, ncol=2)

    # Panel 2: Per-client accuracy
    ax2 = axes[1]
    mean_acc = accuracy_matrix.mean(axis=0)
    ax2.plot(range(T), mean_acc, "k-", linewidth=2, label="Mean")
    for k in range(min(K, 5)):
        ax2.plot(range(T), accuracy_matrix[k], "--", alpha=0.4, linewidth=1)
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Assignment entropy and wrong-memory
    ax3 = axes[2]
    if assignment_entropy_per_t is not None:
        ax3.plot(range(len(assignment_entropy_per_t)),
                 assignment_entropy_per_t, "b-", label="Assign. Entropy")
    if wrong_memory_per_t is not None:
        ax3b = ax3.twinx()
        ax3b.plot(range(len(wrong_memory_per_t)),
                  wrong_memory_per_t, "r--", label="Wrong Memory Rate")
        ax3b.set_ylabel("Wrong Memory Rate", color="r")
    ax3.set_xlabel("Time Step t")
    ax3.set_ylabel("Assignment Entropy", color="b")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 4. Component Ablation on Anchor Settings
# ===================================================================

@dataclass
class AnchorSetting:
    """A representative parameter setting for focused experiments."""
    generator_type: str
    rho: float
    alpha: float
    delta: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.generator_type}_r{self.rho}_a{self.alpha}_d{self.delta}"


# 18 anchor settings covering the full parameter space
DEFAULT_ANCHORS: list[AnchorSetting] = [
    # SINE: strong advantage region
    AnchorSetting("sine", 10.0, 0.0, 0.5, "sine_easy"),
    AnchorSetting("sine", 5.0, 0.25, 0.5, "sine_med_sync"),
    AnchorSetting("sine", 5.0, 0.5, 0.3, "sine_med_mild"),
    AnchorSetting("sine", 5.0, 0.5, 0.7, "sine_med_strong"),
    AnchorSetting("sine", 2.0, 0.75, 0.5, "sine_hard"),
    AnchorSetting("sine", 2.0, 1.0, 1.0, "sine_extreme"),
    # SEA: moderate advantage
    AnchorSetting("sea", 10.0, 0.0, 0.5, "sea_easy"),
    AnchorSetting("sea", 5.0, 0.5, 0.5, "sea_med"),
    AnchorSetting("sea", 5.0, 0.5, 1.0, "sea_med_strong"),
    AnchorSetting("sea", 2.0, 0.5, 0.5, "sea_hard"),
    AnchorSetting("sea", 2.0, 1.0, 0.5, "sea_extreme"),
    AnchorSetting("sea", 5.0, 0.75, 0.7, "sea_async"),
    # CIRCLE: FedProTrack may lose here
    AnchorSetting("circle", 10.0, 0.0, 0.5, "circle_easy"),
    AnchorSetting("circle", 5.0, 0.5, 0.5, "circle_med"),
    AnchorSetting("circle", 5.0, 0.5, 1.0, "circle_med_strong"),
    AnchorSetting("circle", 2.0, 0.5, 0.5, "circle_hard"),
    AnchorSetting("circle", 2.0, 1.0, 0.5, "circle_extreme"),
    AnchorSetting("circle", 5.0, 0.25, 0.3, "circle_sync_mild"),
]


# Component ablation variants
COMPONENT_ABLATIONS: dict[str, dict[str, object]] = {
    "Full": {},
    "Hard assignment": {"omega": 100.0},
    "No transition prior": {"kappa": 0.001},
    "No spawn/merge": {
        "merge_threshold": 1.0,
        "loss_novelty_threshold": 100.0,
        "novelty_threshold": 0.001,
    },
    "No sticky dampening": {"sticky_dampening": 1.0, "sticky_posterior_gate": 1.0},
    "No Phase A (model-only)": {"_phase_a_only": True},
    "No dynamic memory update": {"shrink_every": 99999, "merge_every": 99999},
}


def plot_ablation_table(
    results: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    """Generate ablation results table and bar chart.

    Parameters
    ----------
    results : dict[anchor_label, dict[ablation_variant, dict[metric, value]]]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate across anchors: mean metric per variant
    variants = list(COMPONENT_ABLATIONS.keys())
    metrics_to_show = [
        "concept_re_id_accuracy", "wrong_memory_reuse_rate",
        "final_accuracy", "assignment_entropy",
    ]

    agg: dict[str, dict[str, list[float]]] = {
        v: {m: [] for m in metrics_to_show} for v in variants
    }

    for anchor_label, variant_results in results.items():
        for variant, metric_dict in variant_results.items():
            if variant not in agg:
                continue
            for m in metrics_to_show:
                val = metric_dict.get(m)
                if val is not None and not np.isnan(val):
                    agg[variant][m].append(val)

    # Write CSV
    header = ["variant", "n_anchors"] + [
        f"{m}_mean" for m in metrics_to_show
    ] + [f"{m}_std" for m in metrics_to_show]

    csv_rows = []
    for v in variants:
        row: dict[str, str | float] = {"variant": v}
        n = 0
        for m in metrics_to_show:
            vals = agg[v][m]
            n = max(n, len(vals))
            row[f"{m}_mean"] = round(float(np.mean(vals)), 4) if vals else ""
            row[f"{m}_std"] = round(float(np.std(vals)), 4) if vals else ""
        row["n_anchors"] = n
        csv_rows.append(row)

    with open(output_dir / "ablation_anchor_table.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    # Bar chart
    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(5 * len(metrics_to_show), 5))
    if len(metrics_to_show) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics_to_show):
        means = [float(np.mean(agg[v][m])) if agg[v][m] else 0.0 for v in variants]
        stds = [float(np.std(agg[v][m])) if len(agg[v][m]) > 1 else 0.0
                for v in variants]
        colors = ["#2196F3" if v == "Full" else "#FF9800" for v in variants]
        ax.bar(range(len(variants)), means, yerr=stds, color=colors,
               capsize=3, alpha=0.8)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=40, ha="right", fontsize=7)
        ax.set_title(m.replace("_", " ").title(), fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Component Ablation (18 Anchor Settings)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_anchor_barchart.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# 5. Hyperparameter Robustness
# ===================================================================

def plot_hyperparam_robustness(
    results: dict[str, list[tuple[float, dict[str, float]]]],
    output_dir: Path,
) -> None:
    """Plot hyperparameter robustness: each param has a local sweep.

    Parameters
    ----------
    results : dict[param_name, list[(param_value, {metric: value})]]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = ["concept_re_id_accuracy", "final_accuracy",
                       "wrong_memory_reuse_rate"]

    for param_name, sweep in results.items():
        param_vals = [s[0] for s in sweep]
        fig, axes = plt.subplots(1, len(metrics_to_plot),
                                 figsize=(5 * len(metrics_to_plot), 4))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, m in zip(axes, metrics_to_plot):
            vals = [s[1].get(m, float("nan")) for s in sweep]
            ax.plot(param_vals, vals, "o-", linewidth=2, markersize=6)
            # Mark the default/optimal value
            ax.axvline(x=param_vals[len(param_vals)//2], color="red",
                       linestyle="--", alpha=0.5, label="default")
            ax.set_xlabel(param_name)
            ax.set_ylabel(m.replace("_", " ").title())
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

            # Compute plateau width (within 5% of max)
            arr = np.array([v for v in vals if not np.isnan(v)])
            if len(arr) > 0:
                max_val = np.max(arr)
                threshold = max_val * 0.95
                in_plateau = np.sum(arr >= threshold)
                ax.set_title(f"{m.split('_')[-1]}\nplateau: {in_plateau}/{len(arr)}")

        fig.suptitle(f"Robustness: {param_name}", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / f"robustness_{param_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Write summary CSV
    with open(output_dir / "robustness_summary.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["param", "value"] + metrics_to_plot)
        for param_name, sweep in results.items():
            for val, metric_dict in sweep:
                writer.writerow(
                    [param_name, val] + [
                        round(metric_dict.get(m, float("nan")), 4)
                        for m in metrics_to_plot
                    ]
                )


# ===================================================================
# 6. Statistical Significance
# ===================================================================

def statistical_significance(
    rows: list[RawRow],
    output_dir: Path,
    method_a: str = "FedProTrack",
    method_b: str = "IFCA",
    metric: str = "concept_re_id_accuracy",
    group_by: str | None = None,
) -> dict:
    """Paired statistical test between two methods.

    For each unique setting (same rho, alpha, delta, generator, seed),
    compute the paired difference and run a Wilcoxon signed-rank test.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build paired comparisons
    a_by_key: dict[str, float] = {}
    b_by_key: dict[str, float] = {}

    for r in rows:
        if r.get(metric) is None or r.get(metric) == "":
            continue
        key_parts = [
            str(r.get("generator_type", "")),
            str(r.get("rho", "")),
            str(r.get("alpha", "")),
            str(r.get("delta", "")),
            str(r.get("seed", "")),
        ]
        key = "|".join(key_parts)
        if r.get("method") == method_a:
            a_by_key[key] = float(r[metric])
        elif r.get("method") == method_b:
            b_by_key[key] = float(r[metric])

    common_keys = sorted(set(a_by_key.keys()) & set(b_by_key.keys()))
    if len(common_keys) < 5:
        return {"error": "Not enough paired observations", "n": len(common_keys)}

    a_vals = np.array([a_by_key[k] for k in common_keys])
    b_vals = np.array([b_by_key[k] for k in common_keys])
    diffs = a_vals - b_vals

    # Stats
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    n = len(diffs)
    se = std_diff / np.sqrt(n)
    ci_95 = (mean_diff - 1.96 * se, mean_diff + 1.96 * se)

    # Wilcoxon signed-rank test
    try:
        stat, p_value = sp_stats.wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        stat, p_value = float("nan"), float("nan")

    # Effect size (Cohen's d)
    cohens_d = mean_diff / std_diff if std_diff > 0 else float("inf")

    # Win rate
    win_rate = float(np.mean(diffs > 0))

    result = {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_pairs": n,
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "ci_95_lower": round(ci_95[0], 4),
        "ci_95_upper": round(ci_95[1], 4),
        "wilcoxon_stat": round(float(stat), 2) if not np.isnan(stat) else None,
        "p_value": round(float(p_value), 6) if not np.isnan(p_value) else None,
        "cohens_d": round(cohens_d, 4),
        "win_rate": round(win_rate, 4),
    }

    with open(output_dir / f"significance_{method_a}_vs_{method_b}_{metric}.json",
              "w") as f:
        json.dump(result, f, indent=2)

    # Plot paired difference distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diffs, bins=30, alpha=0.7, color="#2196F3", edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
    ax.axvline(x=mean_diff, color="green", linestyle="-", linewidth=2,
               label=f"mean={mean_diff:.4f}")
    ax.fill_betweenx([0, ax.get_ylim()[1] * 0.8], ci_95[0], ci_95[1],
                     alpha=0.2, color="green", label="95% CI")
    ax.set_xlabel(f"{method_a} - {method_b}: {metric}")
    ax.set_ylabel("Count")
    ax.set_title(f"Paired Differences (n={n}, p={p_value:.4g}, d={cohens_d:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"paired_diff_{method_a}_vs_{method_b}_{metric}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    return result


# ===================================================================
# 7. E4 Byte Breakdown
# ===================================================================

def e4_byte_breakdown(
    byte_stats: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot byte breakdown for E4 analysis.

    Parameters
    ----------
    byte_stats : dict[method, dict[component, bytes]]
        E.g. {"FedProTrack": {"phase_a": 1234, "phase_b": 5678, ...}}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(byte_stats.keys())
    components = set()
    for v in byte_stats.values():
        components.update(v.keys())
    components = sorted(components)

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    colors = plt.cm.tab10(np.linspace(0, 1, len(components)))

    for i, comp in enumerate(components):
        vals = [byte_stats[m].get(comp, 0) for m in methods]
        ax.bar(x, vals, bottom=bottom, label=comp, color=colors[i], alpha=0.8)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Bytes")
    ax.set_title("E4: Communication Cost Breakdown")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "e4_byte_breakdown.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save as CSV
    with open(output_dir / "e4_byte_breakdown.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + list(components) + ["total"])
        for m in methods:
            vals = [byte_stats[m].get(c, 0) for c in components]
            writer.writerow([m] + vals + [sum(vals)])
