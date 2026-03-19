"""Budget-matched CIFAR-100 comparison.

FedProTrack sends lightweight fingerprints per round (Phase A) while IFCA
sends full models.  A fair comparison must match **total communication bytes**,
not federation frequency.  This script sweeps ``federation_every`` for each
method and plots accuracy vs total bytes so we can compare at iso-budget.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fedprotrack.baselines.runners import run_ifca_full
from fedprotrack.baselines.comm_tracker import model_bytes
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.models import TorchLinearClassifier
from fedprotrack.posterior import (
    FEDPROTRACK_VARIANTS,
    FedProTrackRunner,
    make_variant_bundle,
)
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def _run_fedavg(
    dataset, n_features: int, n_classes: int,
    lr: float, n_epochs: int, federation_every: int, seed: int,
) -> tuple[np.ndarray, float]:
    """FedAvg with configurable lr/epochs/federation_every and exact byte tracking."""
    K, T = dataset.config.K, dataset.config.T

    global_model = TorchLinearClassifier(
        n_features=n_features, n_classes=n_classes,
        lr=lr, n_epochs=n_epochs, seed=seed,
    )
    client_models = [
        TorchLinearClassifier(
            n_features=n_features, n_classes=n_classes,
            lr=lr, n_epochs=n_epochs, seed=seed + k,
        )
        for k in range(K)
    ]

    acc_matrix = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        # Evaluate then train locally
        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            y_pred = global_model.predict(X_test)
            acc_matrix[k, t] = float(np.mean(y_pred == y_test))

            if t > 0:
                client_models[k].set_params(global_model.get_params())
            if n_epochs > 1:
                client_models[k].fit(X_train, y_train)
            else:
                client_models[k].partial_fit(X_train, y_train)

        # Aggregate only at federation rounds
        if (t + 1) % federation_every == 0 and t < T - 1:
            client_params_list = [cm.get_params() for cm in client_models]

            # Byte accounting: each client uploads its model, server broadcasts global
            one_model_b = model_bytes(client_params_list[0], precision_bits=32)
            upload_b = K * one_model_b
            download_b = K * one_model_b
            total_bytes += upload_b + download_b

            # Simple average
            global_params: dict[str, np.ndarray] = {}
            for key in client_params_list[0]:
                stacked = np.stack([p[key] for p in client_params_list])
                global_params[key] = np.mean(stacked, axis=0)
            global_model.set_params(global_params)

    return acc_matrix, total_bytes


def run_one(method: str, dataset, ground_truth, cfg: dict) -> dict:
    import os
    os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

    K, T = dataset.config.K, dataset.config.T
    n_features = cfg["n_features"]
    n_classes = len(set(y_val for k in range(K) for t in range(T)
                       for y_val in dataset.data[(k, t)][1]))
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    fed_every = cfg["federation_every"]

    t0 = time.time()

    if method == cfg["fedprotrack_method_name"]:
        n_concepts = int(ground_truth.max()) + 1
        _, config, runner_kwargs = make_variant_bundle(
            cfg["fedprotrack_variant"],
            config_overrides={
                "max_concepts": max(6, n_concepts + 2),
                "shrink_every": 6,
            },
            runner_overrides={
                "federation_every": fed_every,
                "detector_name": "ADWIN",
                "lr": lr,
                "n_epochs": epochs,
            },
        )
        runner = FedProTrackRunner(
            config=config,
            seed=cfg["seed"],
            **runner_kwargs,
        )
        result = runner.run(dataset)
        log = result.to_experiment_log()
        metrics = compute_all_metrics(log, identity_capable=True)
        total_bytes = result.total_bytes
        accuracy_matrix = result.accuracy_matrix

    elif method == "IFCA":
        result = run_ifca_full(
            dataset,
            federation_every=fed_every,
            n_clusters=cfg.get("ifca_clusters", 4),
            lr=lr,
            n_epochs=epochs,
        )
        log = result.to_experiment_log(ground_truth)
        metrics = compute_all_metrics(log, identity_capable=True)
        total_bytes = result.total_bytes
        accuracy_matrix = result.accuracy_matrix

    elif method == "FedAvg":
        accuracy_matrix, total_bytes = _run_fedavg(
            dataset, n_features, n_classes,
            lr=lr, n_epochs=epochs,
            federation_every=fed_every, seed=cfg["seed"],
        )
        predicted_matrix = np.zeros((K, T), dtype=np.int32)
        log = ExperimentLog(
            ground_truth=ground_truth,
            predicted=predicted_matrix,
            accuracy_curve=accuracy_matrix,
            total_bytes=total_bytes if total_bytes > 0 else None,
            method_name="FedAvg",
        )
        metrics = compute_all_metrics(log, identity_capable=False)

    else:
        raise ValueError(method)

    elapsed = time.time() - t0
    return {
        "method": method,
        "seed": cfg["seed"],
        "federation_every": fed_every,
        "epochs": epochs,
        "lr": lr,
        "final_accuracy": metrics.final_accuracy,
        "accuracy_auc": metrics.accuracy_auc,
        "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
        "assignment_switch_rate": metrics.assignment_switch_rate,
        "avg_clients_per_concept": metrics.avg_clients_per_concept,
        "singleton_group_ratio": metrics.singleton_group_ratio,
        "memory_reuse_rate": metrics.memory_reuse_rate,
        "routing_consistency": metrics.routing_consistency,
        "total_bytes": total_bytes,
        "wall_clock_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-matched CIFAR-100")
    parser.add_argument("--results-dir", default="results_cifar100_budget")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--T", type=int, default=12)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--rho", type=float, default=2.5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--delta", type=float, default=0.85)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--samples-per-coarse-class", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ifca-clusters", type=int, default=4)
    # Matched local training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--fedprotrack-variant",
        choices=FEDPROTRACK_VARIANTS,
        default="legacy",
        help="FedProTrack preset to evaluate on the budget frontier.",
    )
    # Federation frequency sweep
    parser.add_argument("--fed-every-list", default="1,2,3,5",
                        help="Comma-separated federation_every values to sweep")
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    fed_every_list = [int(s.strip()) for s in args.fed_every_list.split(",") if s.strip()]
    fedprotrack_method_name, _, _ = make_variant_bundle(args.fedprotrack_variant)
    methods = [fedprotrack_method_name, "IFCA", "FedAvg"]

    dataset_cfg_dict = {
        "K": args.K, "T": args.T, "n_samples": args.n_samples,
        "rho": args.rho, "alpha": args.alpha, "delta": args.delta,
        "n_features": args.n_features,
        "samples_per_coarse_class": args.samples_per_coarse_class,
        "batch_size": args.batch_size, "n_workers": 0,
        "data_root": args.data_root, "feature_cache_dir": args.feature_cache_dir,
        "feature_seed": args.feature_seed,
    }

    print("=== Budget-Matched CIFAR-100 Comparison ===", flush=True)
    print(f"Matched training: epochs={args.epochs}, lr={args.lr}", flush=True)
    print(f"K={args.K}, T={args.T}, seeds={seeds}", flush=True)
    print(f"federation_every sweep: {fed_every_list}", flush=True)
    print(
        f"FedProTrack variant: {args.fedprotrack_variant} "
        f"({fedprotrack_method_name})",
        flush=True,
    )

    # Warm cache
    print("Warming feature cache...", flush=True)
    warm_cfg = CIFAR100RecurrenceConfig(**{**dataset_cfg_dict, "seed": seeds[0]})
    prepare_cifar100_recurrence_feature_cache(warm_cfg)

    all_rows: list[dict] = []
    for seed in seeds:
        ds_cfg = CIFAR100RecurrenceConfig(**{**dataset_cfg_dict, "seed": seed})
        dataset = generate_cifar100_recurrence_dataset(ds_cfg)
        ground_truth = dataset.concept_matrix

        for fed_every in fed_every_list:
            for method in methods:
                cfg = {
                    "seed": seed, "epochs": args.epochs, "lr": args.lr,
                    "n_features": args.n_features,
                    "federation_every": fed_every,
                    "ifca_clusters": args.ifca_clusters,
                    "fedprotrack_variant": args.fedprotrack_variant,
                    "fedprotrack_method_name": fedprotrack_method_name,
                }
                row = run_one(method, dataset, ground_truth, cfg)
                all_rows.append(row)
                reid_s = f" reid={row['concept_re_id_accuracy']:.3f}" if row["concept_re_id_accuracy"] is not None else ""
                print(
                    f"  {method:14s} fed_every={fed_every} seed={seed} "
                    f"acc={row['final_accuracy']:.4f} "
                    f"bytes={row['total_bytes']:.0f}{reid_s}",
                    flush=True,
                )

    # --- Aggregate and print table ---
    print("\n=== Aggregated Results ===", flush=True)
    print(f"{'Method':14s} {'fed_every':>9s} {'FinalAcc':>10s} {'AUC':>8s} "
          f"{'TotalBytes':>12s} {'Re-ID':>8s}", flush=True)
    print("-" * 66, flush=True)

    plot_data: dict[str, list[tuple[float, float, float]]] = {}  # method -> [(bytes, acc, reid)]

    for method in methods:
        for fed_every in fed_every_list:
            rows = [r for r in all_rows
                    if r["method"] == method and r["federation_every"] == fed_every]
            if not rows:
                continue
            mean_acc = float(np.mean([r["final_accuracy"] for r in rows]))
            std_acc = float(np.std([r["final_accuracy"] for r in rows]))
            mean_auc = float(np.mean([r["accuracy_auc"] for r in rows]))
            mean_bytes = float(np.mean([r["total_bytes"] for r in rows]))
            reid_vals = [r["concept_re_id_accuracy"] for r in rows
                         if r["concept_re_id_accuracy"] is not None]
            mean_reid = float(np.mean(reid_vals)) if reid_vals else float("nan")
            reid_s = f"{mean_reid:.4f}" if reid_vals else "--"

            print(
                f"{method:14s} {fed_every:>9d} "
                f"{mean_acc:.4f}±{std_acc:.3f} {mean_auc:8.3f} "
                f"{mean_bytes:12.0f} {reid_s:>8s}",
                flush=True,
            )

            plot_data.setdefault(method, []).append((mean_bytes, mean_acc, mean_reid))

    # --- Plot: accuracy vs bytes (budget frontier) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    markers = {fedprotrack_method_name: "o", "IFCA": "s", "FedAvg": "^"}
    colors = {fedprotrack_method_name: "C0", "IFCA": "C1", "FedAvg": "C2"}
    for method, pts in plot_data.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker=markers.get(method, "o"),
                color=colors.get(method, "gray"),
                linewidth=2, markersize=8, label=method)
        # Annotate federation_every values
        for i, fed_every in enumerate(sorted(fed_every_list)):
            ax.annotate(f"f={fed_every}", (xs[i], ys[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Total Communication Bytes")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("Budget Frontier: Accuracy vs Communication")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Re-ID vs bytes
    ax = axes[1]
    for method, pts in plot_data.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[2] for p in pts_sorted]
        if all(np.isnan(y) for y in ys):
            continue
        ax.plot(xs, ys, marker=markers.get(method, "o"),
                color=colors.get(method, "gray"),
                linewidth=2, markersize=8, label=method)
    ax.set_xlabel("Total Communication Bytes")
    ax.set_ylabel("Concept Re-ID Accuracy")
    ax.set_title("Budget Frontier: Concept Tracking vs Communication")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"CIFAR-100 Budget-Matched (epochs={args.epochs}, lr={args.lr})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(results_dir / "budget_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save raw
    with open(results_dir / "raw_results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    with open(results_dir / "config.json", "w") as f:
        json.dump({
            "dataset_config": dataset_cfg_dict,
            "matched_epochs": args.epochs, "matched_lr": args.lr,
            "seeds": seeds, "methods": methods,
            "fedprotrack_variant": args.fedprotrack_variant,
            "fed_every_list": fed_every_list,
        }, f, indent=2)

    print(f"\nSaved to {results_dir}/", flush=True)


if __name__ == "__main__":
    main()
