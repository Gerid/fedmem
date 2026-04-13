from __future__ import annotations

"""Aggregate and visualize concept-count sensitivity experiments.

Reads RunPod result JSONs from both the C-sweep and misspecified-C
experiments, computes mean±std across seeds, and generates paper-ready
tables and figures.

Usage:
    python analyze_concept_count.py \
        --sweep-files runpod_results_csweep_rho*.json \
        --misspec-files runpod_results_misspec_nc*.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_rows(runpod_json: dict) -> list[dict]:
    """Extract result rows from a RunPod output JSON.

    RunPod results are keyed by seed, each containing the experiment
    output which includes a 'results' or 'rows' list, or the output
    may contain 'results.json' content directly.
    """
    rows: list[dict] = []
    for seed_key, seed_output in runpod_json.items():
        if not isinstance(seed_output, dict):
            continue
        # RunPod handler wraps experiment output; look for rows.
        candidates = (
            seed_output.get("rows")
            or seed_output.get("results", {}).get("rows")
        )
        if isinstance(candidates, list):
            rows.extend(candidates)
            continue
        # Fallback: results.json was saved and returned as nested dict.
        rj = seed_output.get("results.json")
        if isinstance(rj, dict) and "rows" in rj:
            rows.extend(rj["rows"])
    return rows


def _agg_stat(values: list[float]) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.array(values, dtype=np.float64)
    return (float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)


# ── C-sweep analysis ────────────────────────────────────────────────────────

def analyze_sweep(files: list[str], out_dir: Path) -> None:
    """Aggregate C-sweep results across seeds and generate outputs."""
    all_rows: list[dict] = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        all_rows.extend(_extract_rows(data))

    if not all_rows:
        print("  [sweep] No rows found — check file paths and format.")
        return

    # Group by (n_concepts, method).
    groups: dict[tuple[int, str], list[float]] = {}
    fpt_discovered: dict[tuple[int, int], list[int]] = {}  # (C, seed) → discovered_c
    for row in all_rows:
        c = row.get("n_concepts")
        method = row.get("method") or row.get("canonical_method")
        acc = row.get("final_accuracy")
        if c is None or method is None or acc is None:
            continue
        groups.setdefault((int(c), method), []).append(float(acc))
        disc = row.get("fpt_active_concepts")
        if disc is not None:
            seed = row.get("seed", 0)
            fpt_discovered.setdefault((int(c), int(seed)), []).append(int(disc))

    # Build summary table.
    methods_seen = sorted({m for (_, m) in groups})
    c_values = sorted({c for (c, _) in groups})

    table_rows: list[dict] = []
    for c in c_values:
        for m in methods_seen:
            vals = groups.get((c, m), [])
            mean, std = _agg_stat(vals)
            table_rows.append({
                "C": c, "method": m,
                "mean_acc": round(mean, 4), "std_acc": round(std, 4),
                "n_seeds": len(vals),
            })

    # Write CSV.
    import csv
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "c_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["C", "method", "mean_acc", "std_acc", "n_seeds"])
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"  [sweep] Table saved to {tables_dir / 'c_sensitivity.csv'}")

    # Write eigengap verification.
    if fpt_discovered:
        print("  [sweep] FPT eigengap verification:")
        for (c, seed), disc_list in sorted(fpt_discovered.items()):
            for d in disc_list:
                match = "✓" if d == c else "✗"
                print(f"    C={c}, seed={seed}: discovered={d} {match}")

    # Plot.
    if not HAS_MPL:
        print("  [sweep] matplotlib not available — skipping figure.")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Select key methods for the line plot.
    key_methods = [
        m for m in methods_seen
        if any(k in m for k in ["FedProTrack", "Oracle", "FedAvg", "IFCA", "CFL", "FedRC"])
        and "FPTTrain" not in m
    ]
    if not key_methods:
        key_methods = methods_seen[:8]

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in key_methods:
        means = [_agg_stat(groups.get((c, m), []))[0] for c in c_values]
        stds = [_agg_stat(groups.get((c, m), []))[1] for c in c_values]
        ax.errorbar(c_values, means, yerr=stds, label=m, marker="o", capsize=3)
    ax.set_xlabel("Number of Concepts (C)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Concept-Count Sensitivity (CIFAR-100, disjoint labels)")
    ax.set_xticks(c_values)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "c_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"  [sweep] Figure saved to {fig_dir / 'c_sensitivity.png'}")


# ── Misspecified-C analysis ──────────────────────────────────────────────────

def analyze_misspec(files: list[str], out_dir: Path) -> None:
    """Aggregate misspecified-C results and generate outputs."""
    all_rows: list[dict] = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        all_rows.extend(_extract_rows(data))

    if not all_rows:
        print("  [misspec] No rows found — check file paths and format.")
        return

    # Group by (given_n_clusters, method).
    groups: dict[tuple[int, str], list[float]] = {}
    for row in all_rows:
        nc = row.get("given_n_clusters")
        method = row.get("method") or row.get("canonical_method")
        acc = row.get("final_accuracy")
        if nc is None or method is None or acc is None:
            continue
        groups.setdefault((int(nc), method), []).append(float(acc))

    methods_seen = sorted({m for (_, m) in groups})
    nc_values = sorted({nc for (nc, _) in groups})

    # Separate cluster-count-free vs cluster-aware methods.
    free_methods = [m for m in methods_seen
                    if any(k in m for k in ["FedProTrack", "Oracle", "FedAvg", "CFL"])]
    aware_methods = [m for m in methods_seen if m not in free_methods]

    # Write CSV.
    import csv
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict] = []
    for nc in nc_values:
        for m in methods_seen:
            vals = groups.get((nc, m), [])
            mean, std = _agg_stat(vals)
            table_rows.append({
                "given_n_clusters": nc, "method": m,
                "mean_acc": round(mean, 4), "std_acc": round(std, 4),
                "n_seeds": len(vals),
            })

    with open(tables_dir / "misspecified_c.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["given_n_clusters", "method", "mean_acc", "std_acc", "n_seeds"],
        )
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"  [misspec] Table saved to {tables_dir / 'misspecified_c.csv'}")

    if not HAS_MPL:
        print("  [misspec] matplotlib not available — skipping figure.")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Cluster-aware: lines across n_clusters.
    for m in aware_methods:
        means = [_agg_stat(groups.get((nc, m), []))[0] for nc in nc_values]
        stds = [_agg_stat(groups.get((nc, m), []))[1] for nc in nc_values]
        ax.errorbar(nc_values, means, yerr=stds, label=m, marker="s", capsize=3)

    # Cluster-count-free: horizontal reference lines.
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(free_methods), 1)))
    for i, m in enumerate(free_methods):
        # Average across all nc (should be the same data, but average anyway).
        all_vals = []
        for nc in nc_values:
            all_vals.extend(groups.get((nc, m), []))
        if all_vals:
            mean = float(np.mean(all_vals))
            ax.axhline(y=mean, linestyle="--", color=colors[i], label=f"{m} (no C needed)")

    ax.axvline(x=4, linestyle=":", color="gray", alpha=0.5, label="True C=4")
    ax.set_xlabel("Given n_clusters")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Misspecified Cluster Count (CIFAR-100, true C=4)")
    ax.set_xticks(nc_values)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "misspecified_c.png", dpi=150)
    plt.close(fig)
    print(f"  [misspec] Figure saved to {fig_dir / 'misspecified_c.png'}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate concept-count sensitivity experiment results",
    )
    parser.add_argument("--sweep-files", nargs="*", default=[],
                        help="RunPod result JSONs from C-sweep experiment")
    parser.add_argument("--misspec-files", nargs="*", default=[],
                        help="RunPod result JSONs from misspecified-C experiment")
    parser.add_argument("--out-dir", default="tmp/concept_count_analysis",
                        help="Output directory for tables and figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep_files:
        print("=== C-Sweep Analysis ===")
        analyze_sweep(args.sweep_files, out_dir)

    if args.misspec_files:
        print("\n=== Misspecified-C Analysis ===")
        analyze_misspec(args.misspec_files, out_dir)

    if not args.sweep_files and not args.misspec_files:
        print("No input files provided. Use --sweep-files and/or --misspec-files.")


if __name__ == "__main__":
    main()
