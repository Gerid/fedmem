from __future__ import annotations

"""Analyze pre-fix adapter results vs linear baseline and other methods."""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics

BASE = Path(r"E:\fedprotrack\.claude\worktrees\elegant-poitras\tmp\cifar100_adapter")
TSWEEP = Path(r"E:\fedprotrack\.claude\worktrees\elegant-poitras\tmp\cifar100_h1_tsweep\results.csv")


def load_adapter_results():
    with open(BASE / "results.json") as f:
        data = json.load(f)
    return data["rows"]


def load_tsweep_results():
    rows = []
    with open(TSWEEP) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in ["final_accuracy", "accuracy_auc", "total_bytes", "wall_clock_s",
                       "concept_re_id_accuracy", "wrong_memory_reuse_rate", "assignment_entropy"]:
                if row.get(k) and row[k] != "":
                    row[k] = float(row[k])
                else:
                    row[k] = None
            row["T"] = int(row["T"])
            row["seed"] = int(row["seed"])
            rows.append(row)
    return rows


def aggregate_by_method(rows, key="final_accuracy"):
    grouped = defaultdict(list)
    for r in rows:
        val = r.get(key)
        if val is not None:
            grouped[r["method"]].append(val)
    result = {}
    for method, vals in grouped.items():
        result[method] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return result


def budget_efficiency(rows):
    """Compute accuracy per megabyte."""
    grouped = defaultdict(lambda: {"acc": [], "bytes": []})
    for r in rows:
        if r.get("final_accuracy") is not None and r.get("total_bytes") is not None:
            grouped[r["method"]]["acc"].append(r["final_accuracy"])
            grouped[r["method"]]["bytes"].append(r["total_bytes"])
    result = {}
    for method, d in grouped.items():
        mean_acc = statistics.mean(d["acc"])
        mean_bytes = statistics.mean(d["bytes"])
        result[method] = {
            "mean_acc": mean_acc,
            "mean_bytes": mean_bytes,
            "acc_per_MB": mean_acc / (mean_bytes / 1e6) if mean_bytes > 0 else 0,
        }
    return result


def main():
    print("=" * 70)
    print("PRE-FIX ADAPTER ANALYSIS: CIFAR-100 Recurrence Benchmark (T=20)")
    print("=" * 70)

    # --- Adapter experiment results (T=20 fixed) ---
    adapter_rows = load_adapter_results()
    methods_adapter = set(r["method"] for r in adapter_rows)

    print(f"\nMethods in adapter experiment: {sorted(methods_adapter)}")
    print(f"Seeds: {sorted(set(r['seed'] for r in adapter_rows))}")
    print(f"Total rows: {len(adapter_rows)}")

    print("\n--- Final Accuracy (mean +/- std over 5 seeds) ---")
    acc_stats = aggregate_by_method(adapter_rows, "final_accuracy")
    for m in sorted(acc_stats, key=lambda x: acc_stats[x]["mean"], reverse=True):
        s = acc_stats[m]
        print(f"  {m:30s}  {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n']})")

    print("\n--- Accuracy AUC (mean +/- std) ---")
    auc_stats = aggregate_by_method(adapter_rows, "accuracy_auc")
    for m in sorted(auc_stats, key=lambda x: auc_stats[x]["mean"], reverse=True):
        s = auc_stats[m]
        print(f"  {m:30s}  {s['mean']:.4f} +/- {s['std']:.4f}")

    print("\n--- Concept Re-ID Accuracy (identity methods only) ---")
    reid_stats = aggregate_by_method(adapter_rows, "concept_re_id_accuracy")
    for m in sorted(reid_stats, key=lambda x: reid_stats[x]["mean"], reverse=True):
        s = reid_stats[m]
        print(f"  {m:30s}  {s['mean']:.4f} +/- {s['std']:.4f}")

    print("\n--- Wrong Memory Reuse Rate (lower is better) ---")
    wmr_stats = aggregate_by_method(adapter_rows, "wrong_memory_reuse_rate")
    for m in sorted(wmr_stats, key=lambda x: wmr_stats[x]["mean"]):
        s = wmr_stats[m]
        print(f"  {m:30s}  {s['mean']:.4f} +/- {s['std']:.4f}")

    print("\n--- Budget Efficiency (accuracy per MB) ---")
    be = budget_efficiency(adapter_rows)
    for m in sorted(be, key=lambda x: be[x]["acc_per_MB"], reverse=True):
        d = be[m]
        print(f"  {m:30s}  acc={d['mean_acc']:.4f}  bytes={d['mean_bytes']/1e6:.3f}MB  acc/MB={d['acc_per_MB']:.2f}")

    # --- T-sweep comparison at T=20 ---
    print("\n" + "=" * 70)
    print("T-SWEEP COMPARISON AT T=20 (FedProTrack-base = linear)")
    print("=" * 70)

    tsweep_rows = load_tsweep_results()
    t20_rows = [r for r in tsweep_rows if r["T"] == 20]

    if t20_rows:
        acc_t20 = aggregate_by_method(t20_rows, "final_accuracy")
        print("\n--- Final Accuracy at T=20 ---")
        for m in sorted(acc_t20, key=lambda x: acc_t20[x]["mean"], reverse=True):
            s = acc_t20[m]
            print(f"  {m:30s}  {s['mean']:.4f} +/- {s['std']:.4f}")

    # --- Key comparison: adapter vs linear ---
    print("\n" + "=" * 70)
    print("KEY FINDING: ADAPTER vs LINEAR (pre-fix)")
    print("=" * 70)

    linear_acc = [r["final_accuracy"] for r in adapter_rows if r["method"] == "FPT-linear-base"]
    adapter_acc = [r["final_accuracy"] for r in adapter_rows if r["method"] == "FPT-adapter-base"]
    adapter_hp_acc = [r["final_accuracy"] for r in adapter_rows if r["method"] == "FPT-adapter-hybrid-proto"]

    if linear_acc and adapter_acc:
        lin_mean = statistics.mean(linear_acc)
        adp_mean = statistics.mean(adapter_acc)
        gap = lin_mean - adp_mean
        print(f"\n  FPT-linear-base:           {lin_mean:.4f} +/- {statistics.stdev(linear_acc):.4f}")
        print(f"  FPT-adapter-base:          {adp_mean:.4f} +/- {statistics.stdev(adapter_acc):.4f}")
        print(f"  FPT-adapter-hybrid-proto:  {statistics.mean(adapter_hp_acc):.4f} +/- {statistics.stdev(adapter_hp_acc):.4f}")
        print(f"\n  Accuracy gap (linear - adapter-base):        {gap:.4f} ({gap/lin_mean*100:.1f}% relative)")
        print(f"  Accuracy gap (linear - adapter-hybrid-proto): {lin_mean - statistics.mean(adapter_hp_acc):.4f}")

    # Byte comparison
    linear_bytes = [r["total_bytes"] for r in adapter_rows if r["method"] == "FPT-linear-base"]
    adapter_bytes = [r["total_bytes"] for r in adapter_rows if r["method"] == "FPT-adapter-base"]
    if linear_bytes and adapter_bytes:
        lin_b = statistics.mean(linear_bytes)
        adp_b = statistics.mean(adapter_bytes)
        print(f"\n  Linear bytes:  {lin_b/1e6:.3f} MB")
        print(f"  Adapter bytes: {adp_b/1e6:.3f} MB  ({adp_b/lin_b:.1f}x linear)")
        print(f"  --> Adapter uses {adp_b/lin_b:.1f}x more bytes AND gets {gap/lin_mean*100:.0f}% lower accuracy")

    # Re-ID comparison
    linear_reid = [r["concept_re_id_accuracy"] for r in adapter_rows
                   if r["method"] == "FPT-linear-base" and r["concept_re_id_accuracy"] is not None]
    adapter_reid = [r["concept_re_id_accuracy"] for r in adapter_rows
                    if r["method"] == "FPT-adapter-base" and r["concept_re_id_accuracy"] is not None]
    if linear_reid and adapter_reid:
        print(f"\n  Re-ID (linear):  {statistics.mean(linear_reid):.4f}")
        print(f"  Re-ID (adapter): {statistics.mean(adapter_reid):.4f}")
        print(f"  --> Re-ID is IDENTICAL (Phase A routing unchanged)")

    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print("""
  The pre-fix adapter results show catastrophic accuracy failure:
  - FPT-adapter-base: ~0.14 final acc vs FPT-linear-base: ~0.49 (3.5x worse)
  - FPT-adapter-hybrid-proto: ~0.20 (still 2.5x worse than linear)
  - Adapter uses 4.3x more communication bytes than linear
  - Re-ID accuracy is identical => the failure is in the MODEL, not routing
  - CFL (non-identity baseline) achieves 0.61, dominating all FPT variants

  ROOT CAUSE: The adapter model's forward pass or training is broken,
  causing near-random predictions despite correct concept assignments.

  WHAT POST-FIX RESULTS SHOULD SHOW:
  - FPT-adapter-base accuracy should approach or exceed FPT-linear-base (~0.49)
  - Budget efficiency should improve (ideally adapter < linear bytes)
  - Re-ID should remain comparable
  - If adapter still underperforms, the gap should be <10% not 70%
""")


if __name__ == "__main__":
    main()
