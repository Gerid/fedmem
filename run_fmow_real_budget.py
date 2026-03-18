"""FMOW Real Data Budget-Matched Experiment.

Uses pre-extracted ResNet18 features from real FMOW val images.
Sweeps federation_every to produce accuracy vs total_bytes curves.

T=40, K=5, 3 seeds, federation_every ∈ {1, 2, 5, 10}.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from fedprotrack.drift_generator.concept_matrix import generate_concept_matrix
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.data_streams import ConceptSpec
from fedprotrack.drift_generator.generator import DriftDataset
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.baselines.budget_sweep import BudgetPoint, run_budget_sweep
from fedprotrack.metrics.budget_metrics import compute_accuracy_auc
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog

OUTPUT_DIR = Path("results_fmow_real_budget")

# Grid
T = 40
K = 5
N_CONCEPTS = 4
N_FEATURES = 64
N_CLASSES = 10
N_SAMPLES = 200
RHO = 5.0
ALPHA = 0.5
DELTA = 0.5
SEEDS = [42, 123, 777]
FEDERATION_EVERY_VALUES = [1, 2, 5, 10]
FEATURE_CACHE_DIR = Path(".fmow_real_features")


def load_feature_pools() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load pre-extracted feature pools per concept."""
    cache_tag = f"nc{N_CLASSES}_nf{N_FEATURES}_nconcepts{N_CONCEPTS}"
    pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid in range(N_CONCEPTS):
        path = FEATURE_CACHE_DIR / f"concept_{cid}_{cache_tag}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Feature cache not found: {path}\n"
                "Run extract_fmow_features.py first!"
            )
        with np.load(path) as data:
            pools[cid] = (data["X"].astype(np.float32), data["y"].astype(np.int64))
    return pools


def build_dataset(
    pools: dict[int, tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> DriftDataset:
    """Build a DriftDataset from pre-extracted feature pools."""
    rng = np.random.RandomState(seed)

    gen_config = GeneratorConfig(
        K=K, T=T, n_samples=N_SAMPLES,
        rho=RHO, alpha=ALPHA, delta=DELTA,
        generator_type="fmow", seed=seed,
    )
    concept_matrix = generate_concept_matrix(
        K=K, T=T, n_concepts=N_CONCEPTS,
        alpha=ALPHA, seed=seed,
    )

    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            concept_id = int(concept_matrix[k, t])
            seed_kt = seed + 10000 + k * T + t
            kt_rng = np.random.RandomState(seed_kt)

            if concept_id in pools:
                X_pool, y_pool = pools[concept_id]
            else:
                fallback_id = min(pools.keys())
                X_pool, y_pool = pools[fallback_id]

            # Balanced sampling
            classes = np.unique(y_pool)
            per_class = N_SAMPLES // len(classes)
            remainder = N_SAMPLES % len(classes)
            chosen: list[np.ndarray] = []
            for offset, cls in enumerate(classes):
                cls_idx = np.flatnonzero(y_pool == cls)
                take = per_class + (1 if offset < remainder else 0)
                chosen.append(kt_rng.choice(cls_idx, size=take, replace=True))
            batch_idx = np.concatenate(chosen)
            kt_rng.shuffle(batch_idx)
            data[(k, t)] = (
                X_pool[batch_idx].astype(np.float32),
                y_pool[batch_idx].astype(np.int64),
            )

    unique_concepts = sorted(set(concept_matrix.flatten()))
    concept_specs = [
        ConceptSpec(
            concept_id=cid, generator_type="fmow",
            variant=cid, noise_scale=DELTA,
        )
        for cid in unique_concepts
    ]

    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=gen_config,
        concept_specs=concept_specs,
    )


def run_fpt_budget_point(
    dataset: DriftDataset, fe: int, seed: int, event_triggered: bool = False,
) -> BudgetPoint:
    """Run FedProTrack at a given federation_every."""
    fpt_config = TwoPhaseConfig()
    fpt_runner = FedProTrackRunner(
        config=fpt_config, seed=seed, federation_every=fe,
        detector_name="ADWIN", event_triggered=event_triggered,
    )
    fpt_res = fpt_runner.run(dataset)
    auc = compute_accuracy_auc(fpt_res.accuracy_matrix)
    name = "FedProTrack-ET" if event_triggered else "FedProTrack"
    return BudgetPoint(
        method_name=name, federation_every=fe,
        total_bytes=fpt_res.total_bytes, accuracy_auc=auc,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FMOW REAL DATA Budget-Matched Experiment")
    print(f"T={T}, K={K}, rho={RHO}, alpha={ALPHA}, delta={DELTA}")
    print(f"federation_every = {FEDERATION_EVERY_VALUES}")
    print(f"seeds = {SEEDS}")
    print("=" * 70)

    # Load pre-extracted features
    print("\nLoading feature pools...")
    pools = load_feature_pools()
    for cid, (X, y) in sorted(pools.items()):
        print(f"  Concept {cid}: X={X.shape}, y={y.shape}, "
              f"classes={np.unique(y)}")

    all_points: list[dict] = []
    t_start = time.time()

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"Seed {seed} ({seed_idx+1}/{len(SEEDS)})")
        print("=" * 70)

        dataset = build_dataset(pools, seed)
        gt = dataset.concept_matrix
        print(f"Concept matrix:\n{gt[:3, :10]}...")

        for fe in FEDERATION_EVERY_VALUES:
            print(f"\n  federation_every={fe}:")

            # FedProTrack
            t0 = time.time()
            fpt_bp = run_fpt_budget_point(dataset, fe, seed)
            print(f"    FedProTrack:    AUC={fpt_bp.accuracy_auc:.3f}  "
                  f"bytes={fpt_bp.total_bytes:,.0f}  ({time.time()-t0:.1f}s)")
            all_points.append({
                "method": fpt_bp.method_name, "fe": fe, "seed": seed,
                "bytes": fpt_bp.total_bytes, "auc": fpt_bp.accuracy_auc,
            })

            # FedProTrack-ET
            t0 = time.time()
            fpt_et = run_fpt_budget_point(dataset, fe, seed, event_triggered=True)
            print(f"    FedProTrack-ET: AUC={fpt_et.accuracy_auc:.3f}  "
                  f"bytes={fpt_et.total_bytes:,.0f}  ({time.time()-t0:.1f}s)")
            all_points.append({
                "method": fpt_et.method_name, "fe": fe, "seed": seed,
                "bytes": fpt_et.total_bytes, "auc": fpt_et.accuracy_auc,
            })

            # All baselines
            t0 = time.time()
            baseline_points = run_budget_sweep(dataset, [fe])
            for bp in baseline_points:
                print(f"    {bp.method_name:<20s} AUC={bp.accuracy_auc:.3f}  "
                      f"bytes={bp.total_bytes:,.0f}")
                all_points.append({
                    "method": bp.method_name, "fe": fe, "seed": seed,
                    "bytes": bp.total_bytes, "auc": bp.accuracy_auc,
                })
            print(f"    (baselines: {time.time()-t0:.1f}s)")

    t_total = time.time() - t_start

    # Save raw
    raw_path = OUTPUT_DIR / "budget_points.json"
    with open(raw_path, "w") as f:
        json.dump(all_points, f, indent=2)

    # --- Aggregate ---
    print("\n" + "=" * 70)
    print("BUDGET-MATCHED RESULTS (mean across seeds)")
    print("=" * 70)

    methods_seen = sorted(set(p["method"] for p in all_points))
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for p in all_points:
        grouped[(p["method"], p["fe"])].append(p)

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
                mb = np.mean([p["bytes"] for p in pts])
                ma = np.mean([p["auc"] for p in pts])
                print(f" | {mb:>8,.0f} / {ma:.3f} ", end="")
            else:
                print(f" | {'N/A':>18s} ", end="")
        print()

    # --- Budget frontier ---
    frontier: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for method in methods_seen:
        for fe in FEDERATION_EVERY_VALUES:
            pts = grouped.get((method, fe), [])
            if pts:
                mb = np.mean([p["bytes"] for p in pts])
                ma = np.mean([p["auc"] for p in pts])
                frontier[method].append((mb, ma))
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

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {
            "FedProTrack": "#e41a1c", "FedProTrack-ET": "#ff7f00",
            "FedAvg-Full": "#377eb8", "FedProto": "#4daf4a",
            "IFCA": "#984ea3", "TrackedSummary": "#a65628",
            "FedDrift": "#f781bf", "Flash": "#999999",
            "CompressedFedAvg": "#66c2a5",
        }
        markers = {
            "FedProTrack": "o", "FedProTrack-ET": "s",
            "FedAvg-Full": "^", "FedProto": "D", "IFCA": "v",
            "TrackedSummary": "p", "FedDrift": "*", "Flash": "h",
            "CompressedFedAvg": "X",
        }

        for method in sorted(frontier):
            pts = frontier[method]
            bts = [p[0] for p in pts]
            aucs = [p[1] for p in pts]
            ax.plot(bts, aucs, f"-{markers.get(method, 'o')}",
                    color=colors.get(method, "#333"),
                    label=method, markersize=8, linewidth=2)

        ax.set_xlabel("Total Communication Bytes", fontsize=12)
        ax.set_ylabel("Accuracy AUC", fontsize=12)
        ax.set_title("FMOW Real Data Budget Frontier\n"
                      f"(T={T}, K={K}, ρ={RHO}, α={ALPHA}, δ={DELTA})",
                      fontsize=13)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

        fig_path = OUTPUT_DIR / "budget_frontier_real.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved to {fig_path}")
    except Exception as e:
        print(f"\nWarning: plot failed: {e}")

    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")


if __name__ == "__main__":
    main()
