from __future__ import annotations

"""Recurrence-with-gap CIFAR-100 benchmark using shared subset helpers."""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fedprotrack.baselines.runners import run_ifca_full
from fedprotrack.experiments.cifar_overlap import (
    CIFARSubsetBenchmarkConfig,
    build_subset_dataset,
    run_fedavg,
    run_fpt,
    run_local,
    run_oracle,
)

os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

# 8 concepts: disjoint/recombined subsets with long-gap recurrence.
CONCEPT_CLASSES = {
    0: [0, 1, 2, 3, 4],
    1: [5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14],
    3: [15, 16, 17, 18, 19],
    4: [0, 1, 5, 6, 10],
    5: [11, 15, 16, 2, 3],
    6: [4, 7, 8, 17, 18],
    7: [9, 12, 13, 14, 19],
}
PHASE_CONCEPTS = [[0, 1, 2], [3, 4, 5], [0, 1, 2]]


def _next_federation_t(start_t: int, federation_every: int, horizon: int) -> int | None:
    """Return the first time index at or after ``start_t`` that federates."""
    t = start_t
    while t < horizon:
        if (t + 1) % federation_every == 0:
            return t
        t += 1
    return None


def main() -> None:
    results_dir = Path("results_cifar100_recurrence_gap")
    results_dir.mkdir(parents=True, exist_ok=True)

    K, T = 12, 30
    n_samples = 200
    n_features = 128
    epochs, lr = 5, 0.05
    federation_every = 2
    seeds = [42, 123, 456]
    phase3_start = 20
    recovery_eval_t = _next_federation_t(phase3_start, federation_every, T)

    print("=" * 65)
    print("Recurrence-with-Gap Experiment")
    print(f"K={K}, T={T}, 8 total concepts, 3 active/phase")
    print("Phase 1 (t=0-9): concepts 0,1,2")
    print("Phase 2 (t=10-19): concepts 3,4,5  (0,1,2 dormant)")
    print("Phase 3 (t=20-29): concepts 0,1,2 recur")
    print("=" * 65)

    all_rows: list[dict[str, float | int | str]] = []
    all_curves: dict[str, list[np.ndarray]] = {}

    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        ds = build_subset_dataset(
            CIFARSubsetBenchmarkConfig(
                K=K,
                T=T,
                n_samples=n_samples,
                n_features=n_features,
                seed=seed,
                generator_type="cifar100_recurrence",
            ),
            concept_classes=CONCEPT_CLASSES,
            phase_concepts=PHASE_CONCEPTS,
        )
        gt = ds.concept_matrix
        print(f"  Concept matrix sample (client 0): {gt[0].tolist()}")

        fpt_configs = [
            ("FPT", False, 0.15, 0.85, 10, True),
            ("FPT-tight", False, 0.5, 0.6, 8, True),
            ("FPT-tight+DR", True, 0.5, 0.6, 8, True),
            ("FPT-tight+DR+LS", True, 0.5, 0.6, 8, False),
            ("FPT-recur+DR+LS", True, 0.575, 0.6, 8, False),
        ]

        all_methods: list[tuple[str, tuple[bool, float, float, int, bool] | None]] = [
            ("LocalOnly", None),
            ("FedAvg f=2", None),
            ("IFCA-3 f=2", None),
            ("IFCA-8 f=2", None),
            ("Oracle f=2", None),
        ]
        for cfg_name, dr, lnt, mt, mc, global_shared in fpt_configs:
            all_methods.append((f"{cfg_name} f=2", (dr, lnt, mt, mc, global_shared)))

        for name, fpt_cfg in all_methods:
            extra = ""
            if name == "LocalOnly":
                acc_mat, tb = run_local(ds, epochs, lr, seed)
            elif name == "FedAvg f=2":
                acc_mat, tb = run_fedavg(ds, federation_every, epochs, lr, seed)
            elif name.startswith("IFCA"):
                n_clusters = 3 if "3" in name else 8
                ifca_res = run_ifca_full(
                    ds,
                    federation_every=federation_every,
                    n_clusters=n_clusters,
                    lr=lr,
                    n_epochs=epochs,
                )
                acc_mat, tb = ifca_res.accuracy_matrix, ifca_res.total_bytes
            elif name == "Oracle f=2":
                acc_mat, tb = run_oracle(ds, federation_every, epochs, lr, seed)
            else:
                assert fpt_cfg is not None
                dr, lnt, mt, mc, global_shared = fpt_cfg
                acc_mat, tb, sp, mg, ac = run_fpt(
                    ds,
                    federation_every,
                    epochs,
                    lr,
                    seed,
                    dormant_recall=dr,
                    loss_novelty_threshold=lnt,
                    merge_threshold=mt,
                    max_concepts=mc,
                    model_type="feature_adapter",
                    hidden_dim=64,
                    adapter_dim=16,
                    global_shared_aggregation=global_shared,
                )
                extra = (
                    f" spawn={sp} merge={mg} active={ac}"
                    f" local_shared={int(not global_shared)}"
                )

            final = float(acc_mat[:, -1].mean())
            phase1_acc = float(acc_mat[:, :10].mean())
            phase2_acc = float(acc_mat[:, 10:20].mean())
            phase3_acc = float(acc_mat[:, 20:].mean())
            recovery_t20 = float(acc_mat[:, phase3_start].mean()) if T > phase3_start else 0.0
            recovery_next_fed = (
                float(acc_mat[:, recovery_eval_t].mean())
                if recovery_eval_t is not None
                else 0.0
            )

            all_rows.append({
                "method": name,
                "seed": seed,
                "final": final,
                "phase1": phase1_acc,
                "phase2": phase2_acc,
                "phase3": phase3_acc,
                "recovery_t20": recovery_t20,
                "recovery_next_fed": recovery_next_fed,
                "bytes": tb,
            })
            all_curves.setdefault(name, []).append(acc_mat.mean(axis=0))

            print(
                f"  {name:18s} final={final:.4f} "
                f"P1={phase1_acc:.3f} P2={phase2_acc:.3f} P3={phase3_acc:.3f} "
                f"recov@20={recovery_t20:.3f} "
                f"recov@next-fed={recovery_next_fed:.3f} "
                f"bytes={tb:.0f}{extra}"
            )

    print("\n" + "=" * 65)
    print(
        f"{'Method':14s} {'Final':>8s} {'Phase1':>8s} {'Phase2':>8s} "
        f"{'Phase3':>8s} {'Recov@20':>9s} {'Recov@Fed':>10s} {'Bytes':>10s}"
    )
    print("-" * 65)

    method_names = [
        "LocalOnly",
        "FedAvg f=2",
        "IFCA-3 f=2",
        "IFCA-8 f=2",
        "FPT f=2",
        "FPT-tight f=2",
        "FPT-tight+DR f=2",
        "FPT-tight+DR+LS f=2",
        "FPT-recur+DR+LS f=2",
        "Oracle f=2",
    ]
    for method in method_names:
        rows = [r for r in all_rows if r["method"] == method]
        if not rows:
            continue
        mf = np.mean([float(r["final"]) for r in rows])
        p1 = np.mean([float(r["phase1"]) for r in rows])
        p2 = np.mean([float(r["phase2"]) for r in rows])
        p3 = np.mean([float(r["phase3"]) for r in rows])
        rc = np.mean([float(r["recovery_t20"]) for r in rows])
        rf = np.mean([float(r["recovery_next_fed"]) for r in rows])
        mb = np.mean([float(r["bytes"]) for r in rows])
        print(
            f"{method:14s} {mf:8.4f} {p1:8.4f} {p2:8.4f} "
            f"{p3:8.4f} {rc:9.4f} {rf:10.4f} {mb:10.0f}"
        )

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {
        "LocalOnly": "gray",
        "FedAvg f=2": "C2",
        "IFCA-3 f=2": "C1",
        "IFCA-8 f=2": "C3",
        "FPT f=2": "C0",
        "FPT-tight f=2": "C4",
        "FPT-tight+DR f=2": "C5",
        "FPT-tight+DR+LS f=2": "C7",
        "FPT-recur+DR+LS f=2": "C8",
        "Oracle f=2": "C6",
    }
    linestyles = {
        "Oracle f=2": "--",
        "FPT-tight+DR+LS f=2": "-.",
        "FPT-recur+DR+LS f=2": "-.",
    }

    for method in method_names:
        curves = all_curves.get(method, [])
        if not curves:
            continue
        mean_curve = np.mean(curves, axis=0)
        ax.plot(
            range(T),
            mean_curve,
            linewidth=2,
            color=colors.get(method, "black"),
            linestyle=linestyles.get(method, "-"),
            label=method,
            marker="o" if "Oracle" not in method else "",
        )

    ax.axvline(10, color="red", alpha=0.5, linestyle=":", linewidth=2)
    ax.axvline(20, color="red", alpha=0.5, linestyle=":", linewidth=2)
    ymax = ax.get_ylim()[1]
    ax.text(4, ymax * 0.95, "Phase 1\nconcepts 0,1,2", ha="center", fontsize=9, color="red")
    ax.text(14, ymax * 0.95, "Phase 2\nconcepts 3,4,5", ha="center", fontsize=9, color="red")
    ax.text(
        24,
        ymax * 0.95,
        "Phase 3\n0,1,2 recur",
        ha="center",
        fontsize=9,
        color="red",
        fontweight="bold",
    )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Accuracy (across clients)")
    ax.set_title("Recurrence-with-Gap: 8 concepts, 3 active/phase, K=12")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "recurrence_gap_curves.png", dpi=150)
    plt.close(fig)

    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, default=str)

    print(f"\nSaved to {results_dir}/")


if __name__ == "__main__":
    main()
