from __future__ import annotations

"""Generate all paper figures from frozen artifacts."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
})

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = Path(__file__).resolve().parent


def load_crossover():
    with open(ROOT / "tmp/crossover_6method_seedavg/summary.json") as f:
        return json.load(f)


def load_cifar():
    with open(ROOT / "tmp/cifar100_bj_proxy_local/bj_proxy_results.json") as f:
        return json.load(f)


def load_transient():
    with open(ROOT / "tmp/transient_analysis/transient_results.json") as f:
        return json.load(f)


# ---- Figure 1: Phase diagram ----
def fig1_phase_diagram():
    data = load_crossover()
    rows = data["rows"]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.2))

    snr_vals = []
    c_vals = []
    colors = []
    markers = []

    for r in rows:
        snr = r["SNR_concept"]
        C = r["C"]
        snr_vals.append(snr)
        c_vals.append(C)
        if r["theory_correct"]:
            if r["empirical_oracle_wins"]:
                colors.append("#2196F3")  # blue = Oracle wins, theory correct
                markers.append("o")
            else:
                colors.append("#FF9800")  # orange = FedAvg wins, theory correct
                markers.append("s")
        else:
            colors.append("#F44336")  # red = mismatch
            markers.append("X")

    # Plot points
    for snr, c, color, marker in zip(snr_vals, c_vals, colors, markers):
        ax.scatter(c, snr, c=color, marker=marker, s=25, alpha=0.7,
                   edgecolors="k", linewidths=0.3, zorder=3)

    # Theory boundary: SNR = C - 1
    c_range = np.linspace(1.5, 9, 100)
    ax.plot(c_range, c_range - 1, "k--", linewidth=1.5, label="SNR = C$-$1 (theory)", zorder=2)

    ax.set_xlabel("Number of concepts $C$")
    ax.set_ylabel("SNR$_{\\mathrm{concept}}$")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 9)
    ax.set_ylim(0.3, 2000)
    ax.set_xticks([2, 4, 8])

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=6, markeredgecolor="k", markeredgewidth=0.3,
               label="Oracle wins (correct)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#FF9800",
               markersize=6, markeredgecolor="k", markeredgewidth=0.3,
               label="Global wins (correct)"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#F44336",
               markersize=7, markeredgecolor="k", markeredgewidth=0.3,
               label="Mismatch"),
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.5,
               label="SNR = $C-1$ (Thm 1)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
              framealpha=0.9)

    ax.set_title("Crossover Phase Diagram (108 configs, 3 seeds)")
    fig.savefig(FIG_DIR / "phase_diagram.pdf")
    plt.close(fig)
    print("  -> phase_diagram.pdf")


# ---- Figure 2: Best method bar chart ----
def fig2_best_method():
    data = load_crossover()
    counts = data["best_method_counts"]

    methods = ["Shrinkage", "Oracle", "APFL", "FedAvg", "IFCA", "CFL"]
    values = [counts.get(m, 0) for m in methods]
    colors_bar = ["#4CAF50", "#2196F3", "#9C27B0", "#FF9800", "#795548", "#607D8B"]

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    bars = ax.barh(methods, values, color=colors_bar, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Number of configs where method is best (out of 108)")
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val}", va="center", fontsize=8)

    ax.set_xlim(0, 80)
    ax.set_title("Best Method by MSE (seed-averaged)")
    fig.savefig(FIG_DIR / "best_method.pdf")
    plt.close(fig)
    print("  -> best_method.pdf")


# ---- Figure 3: SNR vs Oracle advantage scatter ----
def fig3_snr_advantage():
    data = load_crossover()
    rows = data["rows"]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))

    snr_list = []
    advantage_list = []
    color_list = []

    for r in rows:
        snr = r["SNR_concept"]
        adv = r["fedavg_mse"] - r["oracle_mse"]  # positive = Oracle better
        snr_list.append(snr)
        advantage_list.append(adv)
        color_list.append("#2196F3" if r["theory_correct"] else "#F44336")

    ax.scatter(snr_list, advantage_list, c=color_list, s=15, alpha=0.6,
               edgecolors="k", linewidths=0.2)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("SNR$_{\\mathrm{concept}}$")
    ax.set_ylabel("FedAvg MSE $-$ Oracle MSE")
    ax.set_title("Oracle Advantage vs SNR")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=5, label=f"Theory correct ({sum(1 for c in color_list if c == '#2196F3')})"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#F44336",
               markersize=5, label=f"Mismatch ({sum(1 for c in color_list if c == '#F44336')})"),
    ]
    ax.legend(handles=legend_elements, fontsize=7)

    fig.savefig(FIG_DIR / "snr_advantage.pdf")
    plt.close(fig)
    print("  -> snr_advantage.pdf")


# ---- Figure 4: Shrinkage advantage heatmap ----
def fig4_shrinkage_heatmap():
    data = load_crossover()
    rows = data["rows"]

    # Group by (K, C) — show shrinkage win rate
    from collections import defaultdict
    kc_wins = defaultdict(lambda: [0, 0])  # [wins, total]
    for r in rows:
        key = (r["K"], r["C"])
        kc_wins[key][1] += 1
        if r["best_method"] == "Shrinkage":
            kc_wins[key][0] += 1

    K_vals = sorted(set(r["K"] for r in rows))
    C_vals = sorted(set(r["C"] for r in rows))

    matrix = np.zeros((len(K_vals), len(C_vals)))
    for i, K in enumerate(K_vals):
        for j, C in enumerate(C_vals):
            wins, total = kc_wins.get((K, C), (0, 1))
            matrix[i, j] = wins / max(total, 1) * 100

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    im = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(C_vals)))
    ax.set_xticklabels(C_vals)
    ax.set_yticks(range(len(K_vals)))
    ax.set_yticklabels(K_vals)
    ax.set_xlabel("$C$ (concepts)")
    ax.set_ylabel("$K$ (clients)")

    for i in range(len(K_vals)):
        for j in range(len(C_vals)):
            ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if matrix[i,j] > 60 else "black")

    ax.set_title("Shrinkage Win Rate (%)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(FIG_DIR / "shrinkage_heatmap.pdf")
    plt.close(fig)
    print("  -> shrinkage_heatmap.pdf")


# ---- Figure 5: Regime boundary ----
def fig5_regime_boundary():
    data = load_transient()

    configs = [
        ("kc2_near_fast", "$K$=8, $\\delta$=0.5"),
        ("kc2_near_slow", "$K$=8, $\\delta$=0.5, $\\tau$=10"),
        ("kc1_near_fast", "$K$=4, $\\delta$=1.0"),
        ("kc2_above_fast", "$K$=8, $\\delta$=1.5"),
        ("kc5_above_fast", "$K$=20, $\\delta$=1.5"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))

    labels = []
    oracle_fracs = []
    fedavg_fracs = []

    for key, label in configs:
        r = data[key]
        n_o = len(r["oracle_dominates_at"])
        n_f = len(r["fedavg_dominates_at"])
        total = 40
        oracle_fracs.append(n_o / total * 100)
        fedavg_fracs.append(n_f / total * 100)
        p = r["params"]
        snr = p["K"] * (p["n"] // 2) * p["delta"] ** 2 / (p["sigma"] ** 2 * p["d"])
        labels.append(f"SNR={snr:.0f}")

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, oracle_fracs, width, label="Oracle dominates",
           color="#2196F3", edgecolor="k", linewidth=0.5)
    ax.bar(x + width / 2, fedavg_fracs, width, label="FedAvg dominates",
           color="#FF9800", edgecolor="k", linewidth=0.5)

    ax.set_ylabel("Fraction of rounds (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(50, color="k", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=7)
    ax.set_title("Regime Boundary: Oracle vs FedAvg Dominance")
    ax.set_ylim(0, 105)

    fig.savefig(FIG_DIR / "regime_boundary.pdf")
    plt.close(fig)
    print("  -> regime_boundary.pdf")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_phase_diagram()
    fig2_best_method()
    fig3_snr_advantage()
    fig4_shrinkage_heatmap()
    fig5_regime_boundary()
    print("Done!")
