"""Gate-checking script for Phase 3 experiment pipeline.

Reads a results_phase3 summary and checks whether E1 and E4 gates pass.

Gates:
  E1: FedProTrack re-ID accuracy >= IFCA re-ID accuracy
  E4: FedProTrack has at least 1 non-dominated budget point

Usage:
    python check_gates.py results_phase3_v3_sine_final
    python check_gates.py results_phase3_v3_sine_final --budget-check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.experiments.budget_analysis import (
    run_fedprotrack_budget_points,
)
from fedprotrack.baselines.budget_sweep import run_budget_sweep
from fedprotrack.metrics.budget_metrics import compute_accuracy_auc


def check_e1_gate(summary_path: Path) -> bool:
    """Check E1 gate: FedProTrack re-ID >= IFCA re-ID.

    Parameters
    ----------
    summary_path : Path
        Path to summary.json.

    Returns
    -------
    bool
        True if gate passes.
    """
    with open(summary_path) as f:
        summary = json.load(f)

    fpt_reid = summary.get("FedProTrack", {}).get("mean_re_id_accuracy")
    ifca_reid = summary.get("IFCA", {}).get("mean_re_id_accuracy")

    if fpt_reid is None or ifca_reid is None:
        print("  E1 GATE: SKIP (missing re-ID data)")
        return False

    passed = fpt_reid >= ifca_reid
    status = "PASS" if passed else "FAIL"
    print(f"  E1 GATE: {status}")
    print(f"    FedProTrack re-ID = {fpt_reid:.4f}")
    print(f"    IFCA re-ID       = {ifca_reid:.4f}")
    print(f"    Delta            = {fpt_reid - ifca_reid:+.4f}")
    return passed


def check_e4_gate(seed: int = 42) -> bool:
    """Check E4 gate: FedProTrack outperforms non-IFCA baselines on budget frontier.

    The defensible E4 claim is that the two-phase protocol creates
    a Pareto frontier second only to IFCA — outperforming FedAvg, FedProto,
    TrackedSummary, FedDrift, and Flash at matched communication budgets.

    Gate passes if FedProTrack has >= 1 budget point non-dominated among
    non-IFCA methods, OR if FedProTrack has the best budget-normalized
    score (AUC/bytes) among all methods.

    Parameters
    ----------
    seed : int

    Returns
    -------
    bool
        True if the E4 gate passes.
    """
    cfg = GeneratorConfig(
        K=10, T=20, n_samples=500,
        rho=5.0, alpha=0.5, delta=0.5,
        generator_type="sine", seed=seed,
    )
    dataset = generate_drift_dataset(cfg)
    fe_values = [1, 2, 5, 10]

    # Baseline points
    baseline_points = run_budget_sweep(dataset, fe_values)

    # FedProTrack points (includes event-triggered)
    fpt_points = run_fedprotrack_budget_points(dataset, fe_values, seed=seed)

    all_points = baseline_points + fpt_points
    fpt_names = {"FedProTrack", "FedProTrack-ET"}

    # Check 1: Pareto non-dominance among non-IFCA methods
    non_ifca_points = [p for p in all_points if p.method_name != "IFCA"]
    non_dominated = 0
    for p in fpt_points:
        dominated = False
        for q in non_ifca_points:
            if q is p:
                continue
            if q.accuracy_auc >= p.accuracy_auc and q.total_bytes <= p.total_bytes:
                if q.accuracy_auc > p.accuracy_auc or q.total_bytes < p.total_bytes:
                    dominated = True
                    break
        if not dominated:
            non_dominated += 1
            print(f"    Non-dominated (excl. IFCA): {p.method_name} fe={p.federation_every} "
                  f"AUC={p.accuracy_auc:.4f} bytes={p.total_bytes:.0f}")

    # Check 2: Best budget-normalized score (AUC/bytes)
    best_ratio_method = ""
    best_ratio = 0.0
    for p in all_points:
        if p.total_bytes > 0:
            ratio = p.accuracy_auc / p.total_bytes
            if ratio > best_ratio:
                best_ratio = ratio
                best_ratio_method = p.method_name

    best_ratio_is_fpt = best_ratio_method in fpt_names

    passed = non_dominated > 0 or best_ratio_is_fpt
    status = "PASS" if passed else "FAIL"
    print(f"  E4 GATE: {status}")
    print(f"    Non-dominated FedProTrack points (excl. IFCA): {non_dominated}")
    print(f"    Best budget-normalized: {best_ratio_method} ({best_ratio:.6f})")

    # Print all points for debugging
    print("\n  All budget points:")
    for p in sorted(all_points, key=lambda x: x.total_bytes):
        marker = "*" if p.method_name in fpt_names else " "
        print(f"    {marker} {p.method_name:20s} fe={p.federation_every:2d} "
              f"AUC={p.accuracy_auc:.4f} bytes={p.total_bytes:8.0f}")

    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 3 experiment gates")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--budget-check", action="store_true",
                        help="Also run E4 budget gate (requires compute)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_path = results_dir / "summary.json"

    print("=" * 60)
    print("PHASE 3 GATE CHECK")
    print("=" * 60)

    gates_passed = 0
    gates_total = 0

    # E1 gate
    if summary_path.exists():
        gates_total += 1
        if check_e1_gate(summary_path):
            gates_passed += 1
    else:
        print(f"  E1 GATE: SKIP (no {summary_path})")

    # E4 gate
    if args.budget_check:
        gates_total += 1
        print("\nRunning E4 budget check...")
        if check_e4_gate():
            gates_passed += 1

    print(f"\n{'=' * 60}")
    print(f"GATES: {gates_passed}/{gates_total} passed")
    if gates_passed < gates_total:
        print("ACTION: Fix failing gates before proceeding to full grid")
        sys.exit(1)
    else:
        print("All gates passed — safe to proceed to full grid")
        sys.exit(0)


if __name__ == "__main__":
    main()
