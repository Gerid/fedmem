"""FMOW Budget-Matched Experiment.

Fair comparison: sweep federation_every ∈ {1, 2, 5, 10} to produce
accuracy vs total_bytes curves for each method.

Uses a representative setting: T=40, K=5, rho=5, alpha=0.5, delta=0.5.
3 seeds for variance estimates.

Methods: FedProTrack, FedProTrack-ET, IFCA, FedAvg-Full, FedProto,
         TrackedSummary, FedDrift.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fedprotrack.real_data.fmow import FMOWConfig, generate_fmow_dataset
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.baselines.budget_sweep import BudgetPoint, run_budget_sweep
from fedprotrack.metrics.budget_metrics import compute_accuracy_auc


OUTPUT_DIR = Path("results_fmow_budget")

# Fixed dataset params
T = 40
K = 5
RHO = 5.0
ALPHA = 0.5
DELTA = 0.5
N_SAMPLES = 100
N_CONCEPTS = 4
N_FEATURES = 16
N_CLASSES = 5
SEEDS = [42, 123, 777]

FEDERATION_EVERY_VALUES = [1, 2, 5, 10]


def run_fpt_budget_point(
    dataset, fe: int, seed: int, event_triggered: bool = False,
) -> BudgetPoint:
    """Run FedProTrack at a given federation_every, return BudgetPoint."""
    fpt_config = TwoPhaseConfig()
    fpt_runner = FedProTrackRunner(
        config=fpt_config,
        seed=seed,
        federation_every=fe,
        detector_name="ADWIN",
        event_triggered=event_triggered,
    )
    fpt_res = fpt_runner.run(dataset)
    auc = compute_accuracy_auc(fpt_res.accuracy_matrix)
    name = "FedProTrack-ET" if event_triggered else "FedProTrack"
    return BudgetPoint(
        method_name=name,
        federation_every=fe,
        total_bytes=fpt_res.total_bytes,
        accuracy_auc=auc,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FMOW Budget-Matched Experiment")
    print(f"T={T}, K={K}, rho={RHO}, alpha={ALPHA}, delta={DELTA}")
    print(f"federation_every = {FEDERATION_EVERY_VALUES}")
    print(f"seeds = {SEEDS}")
    print("=" * 70)

    # Collect all budget points across seeds
    all_points: list[dict] = []
    t_start = time.time()

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")

        cfg = FMOWConfig(
            K=K, T=T, n_samples=N_SAMPLES, n_concepts=N_CONCEPTS,
            n_features=N_FEATURES, n_classes=N_CLASSES,
            rho=RHO, alpha=ALPHA, delta=DELTA, seed=seed,
            data_root=".fmow_budget_cache",
            feature_cache_dir=".fmow_budget_features",
        )
        t0 = time.time()
        dataset = generate_fmow_dataset(cfg)
        print(f"  Dataset generated in {time.time()-t0:.1f}s")

        for fe in FEDERATION_EVERY_VALUES:
            print(f"\n  federation_every={fe}:")

            # FedProTrack
            t0 = time.time()
            fpt_bp = run_fpt_budget_point(dataset, fe, seed, event_triggered=False)
            print(f"    FedProTrack:    AUC={fpt_bp.accuracy_auc:.3f}  "
                  f"bytes={fpt_bp.total_bytes:,.0f}  ({time.time()-t0:.1f}s)")
            all_points.append({
                "method": fpt_bp.method_name, "fe": fe, "seed": seed,
                "bytes": fpt_bp.total_bytes, "auc": fpt_bp.accuracy_auc,
            })

            # FedProTrack-ET
            t0 = time.time()
            fpt_et_bp = run_fpt_budget_point(dataset, fe, seed, event_triggered=True)
            print(f"    FedProTrack-ET: AUC={fpt_et_bp.accuracy_auc:.3f}  "
                  f"bytes={fpt_et_bp.total_bytes:,.0f}  ({time.time()-t0:.1f}s)")
            all_points.append({
                "method": fpt_et_bp.method_name, "fe": fe, "seed": seed,
                "bytes": fpt_et_bp.total_bytes, "auc": fpt_et_bp.accuracy_auc,
            })

            # All baselines via run_budget_sweep (single fe value)
            t0 = time.time()
            baseline_points = run_budget_sweep(dataset, [fe])
            elapsed = time.time() - t0
            for bp in baseline_points:
                print(f"    {bp.method_name:<20s} AUC={bp.accuracy_auc:.3f}  "
                      f"bytes={bp.total_bytes:,.0f}")
                all_points.append({
                    "method": bp.method_name, "fe": fe, "seed": seed,
                    "bytes": bp.total_bytes, "auc": bp.accuracy_auc,
                })
            print(f"    (baselines: {elapsed:.1f}s)")

    t_total = time.time() - t_start

    # Save raw data
    raw_path = OUTPUT_DIR / "budget_points.json"
    with open(raw_path, "w") as f:
        json.dump(all_points, f, indent=2)
    print(f"\nRaw data saved to {raw_path}")

    # --- Aggregate table: method -> [(mean_bytes, mean_auc)] by fe ---
    print("\n" + "=" * 70)
    print("BUDGET-MATCHED RESULTS (mean across seeds)")
    print("=" * 70)

    methods_seen = sorted(set(p["method"] for p in all_points))

    # Group by (method, fe)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for p in all_points:
        grouped[(p["method"], p["fe"])].append(p)

    # Print table
    print(f"\n{'Method':<22s}", end="")
    for fe in FEDERATION_EVERY_VALUES:
        print(f" | fe={fe:>2d} (bytes / AUC)", end="")
    print()
    print("-" * (22 + len(FEDERATION_EVERY_VALUES) * 25))

    for method in methods_seen:
        print(f"{method:<22s}", end="")
        for fe in FEDERATION_EVERY_VALUES:
            pts = grouped.get((method, fe), [])
            if pts:
                mean_bytes = np.mean([p["bytes"] for p in pts])
                mean_auc = np.mean([p["auc"] for p in pts])
                print(f" | {mean_bytes:>8,.0f} / {mean_auc:.3f} ", end="")
            else:
                print(f" | {'N/A':>18s} ", end="")
        print()

    # --- Budget frontier: at similar byte counts, who has higher AUC? ---
    print("\n" + "=" * 70)
    print("BUDGET FRONTIER ANALYSIS")
    print("=" * 70)

    # For each method, compute mean (bytes, AUC) across seeds for each fe
    frontier: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for method in methods_seen:
        for fe in FEDERATION_EVERY_VALUES:
            pts = grouped.get((method, fe), [])
            if pts:
                mean_bytes = np.mean([p["bytes"] for p in pts])
                mean_auc = np.mean([p["auc"] for p in pts])
                frontier[method].append((mean_bytes, mean_auc))

    # Sort each method's frontier by bytes
    for method in frontier:
        frontier[method].sort(key=lambda x: x[0])

    print(f"\n{'Method':<22s} {'Min Bytes':>10s} {'Max Bytes':>10s} "
          f"{'AUC@min':>8s} {'AUC@max':>8s}")
    print("-" * 62)
    for method in sorted(frontier):
        pts = frontier[method]
        if pts:
            print(f"{method:<22s} {pts[0][0]:>10,.0f} {pts[-1][0]:>10,.0f} "
                  f"{pts[0][1]:>8.3f} {pts[-1][1]:>8.3f}")

    # --- Generate matplotlib plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            "FedProTrack": "#e41a1c",
            "FedProTrack-ET": "#ff7f00",
            "FedAvg-Full": "#377eb8",
            "FedProto": "#4daf4a",
            "IFCA": "#984ea3",
            "TrackedSummary": "#a65628",
            "FedDrift": "#f781bf",
            "Flash": "#999999",
            "CompressedFedAvg": "#66c2a5",
        }
        markers = {
            "FedProTrack": "o",
            "FedProTrack-ET": "s",
            "FedAvg-Full": "^",
            "FedProto": "D",
            "IFCA": "v",
            "TrackedSummary": "p",
            "FedDrift": "*",
            "Flash": "h",
            "CompressedFedAvg": "X",
        }

        for method in sorted(frontier):
            pts = frontier[method]
            bytes_arr = [p[0] for p in pts]
            auc_arr = [p[1] for p in pts]
            color = colors.get(method, "#333333")
            marker = markers.get(method, "o")
            ax.plot(bytes_arr, auc_arr, f"-{marker}",
                    color=color, label=method, markersize=8, linewidth=2)

        ax.set_xlabel("Total Communication Bytes", fontsize=12)
        ax.set_ylabel("Accuracy AUC", fontsize=12)
        ax.set_title(
            f"FMOW Budget Frontier (T={T}, K={K}, "
            f"ρ={RHO}, α={ALPHA}, δ={DELTA})",
            fontsize=13,
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        fig_path = OUTPUT_DIR / "budget_frontier.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nBudget frontier plot saved to {fig_path}")
    except Exception as e:
        print(f"\nWarning: could not generate plot: {e}")

    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")


if __name__ == "__main__":
    main()
