from __future__ import annotations

"""Fair CIFAR-100 comparison with matched n_epochs across ALL methods.

All methods use the SAME local training strength:
  - n_epochs = 5
  - lr = 0.1
  - federation_every = 2

This eliminates the confound where CFL defaulted to n_epochs=10 while
FedProTrack used n_epochs=1, giving CFL an unfair ~10% accuracy advantage.

Methods: FedProTrack (tuned), CFL, IFCA, FedAvg, Oracle.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.cfl import CFLClient, CFLServer
from fedprotrack.baselines.comm_tracker import model_bytes
from fedprotrack.baselines.runners import run_ifca_full
from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import identity_metrics_valid
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
N_EPOCHS = 5
LR = 0.1
FEDERATION_EVERY = 2
K = 10
T = 30
N_SAMPLES = 800
RHO = 7.5       # -> 4 concepts
ALPHA = 0.5
DELTA = 0.85
N_FEATURES = 128
SEEDS = [42, 123, 456]
RESULTS_DIR = Path("E:/fedprotrack/.claude/worktrees/elegant-poitras/tmp/matched_epochs_comparison")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def _make_log(
    method_name: str,
    result: object,
    ground_truth: np.ndarray,
) -> ExperimentLog:
    total_bytes = getattr(result, "total_bytes", None)
    if total_bytes is not None and float(total_bytes) <= 0.0:
        total_bytes = None
    if hasattr(result, "to_experiment_log"):
        try:
            return result.to_experiment_log(ground_truth)
        except TypeError:
            return result.to_experiment_log()
    return ExperimentLog(
        ground_truth=ground_truth,
        predicted=np.asarray(
            getattr(result, "predicted_concept_matrix"), dtype=np.int32,
        ),
        accuracy_curve=np.asarray(
            getattr(result, "accuracy_matrix"), dtype=np.float64,
        ),
        total_bytes=total_bytes,
        method_name=method_name,
    )


# ---------------------------------------------------------------------------
# CFL with explicit n_epochs / lr (run_cfl_full hardcodes defaults)
# ---------------------------------------------------------------------------

def run_cfl_matched(
    dataset,
    *,
    federation_every: int = FEDERATION_EVERY,
    eps_1: float = 0.4,
    eps_2: float = 1.6,
    warmup_rounds: int = 20,
    max_clusters: int = 8,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
) -> object:
    """CFL with explicitly matched n_epochs and lr."""
    gc = dataset.config
    K_local, T_local = gc.K, gc.T

    first_key = next(iter(dataset.data))
    X0, _ = dataset.data[first_key]
    n_features = X0.shape[1]
    all_labels: set[int] = set()
    for _, y in dataset.data.values():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    clients = [
        CFLClient(k, n_features, n_classes, lr=lr, n_epochs=n_epochs, seed=42 + k)
        for k in range(K_local)
    ]
    server = CFLServer(
        n_features=n_features,
        n_classes=n_classes,
        seed=42,
        eps_1=eps_1,
        eps_2=eps_2,
        warmup_rounds=warmup_rounds,
        max_clusters=max_clusters,
    )

    accuracy_matrix = np.zeros((K_local, T_local), dtype=np.float64)
    predicted_matrix = np.zeros((K_local, T_local), dtype=np.int32)
    total_bytes = 0.0

    for client in clients:
        params = server.broadcast(client.client_id)
        if params:
            client.set_model_params(params)
        client._cluster_id = server.client_cluster_map.get(client.client_id, 0)

    for t in range(T_local):
        for k in range(K_local):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = _accuracy(y, preds)

        for k in range(K_local):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % federation_every == 0 and t < T_local - 1:
            uploads = [c.get_upload() for c in clients]
            upload_b = sum(c.upload_bytes() for c in clients)
            server.aggregate(uploads, round_idx=t)
            download_b = 0.0
            for client in clients:
                client._cluster_id = server.client_cluster_map.get(client.client_id, 0)
                params = server.broadcast(client.client_id)
                if params:
                    download_b += model_bytes(params)
                    client.set_model_params(params)
            total_bytes += upload_b + download_b

        for k in range(K_local):
            predicted_matrix[k, t] = server.client_cluster_map.get(
                k, clients[k]._cluster_id,
            )

    return type("MethodResult", (), {
        "accuracy_matrix": accuracy_matrix,
        "predicted_concept_matrix": predicted_matrix,
        "total_bytes": total_bytes,
    })()


# ---------------------------------------------------------------------------
# Build all methods
# ---------------------------------------------------------------------------

def build_methods(dataset, exp_cfg, seed):
    """Build method callables with matched training strength."""
    n_concepts_true = int(dataset.concept_matrix.max()) + 1

    return {
        "FedProTrack": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.02,
                sticky_dampening=1.0,
                sticky_posterior_gate=0.35,
                merge_threshold=0.80,
                min_count=5.0,
                max_concepts=6,
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=FEDERATION_EVERY,
            detector_name="ADWIN",
            seed=seed,
            lr=LR,
            n_epochs=N_EPOCHS,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        ).run(dataset),

        "CFL": lambda: run_cfl_matched(
            dataset,
            federation_every=FEDERATION_EVERY,
            eps_1=0.4,
            eps_2=1.6,
            n_epochs=N_EPOCHS,
            lr=LR,
        ),

        "IFCA": lambda: run_ifca_full(
            dataset,
            federation_every=FEDERATION_EVERY,
            n_clusters=n_concepts_true,
            lr=LR,
            n_epochs=N_EPOCHS,
        ),

        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg,
            dataset=dataset,
            lr=LR,
            n_epochs=N_EPOCHS,
            seed=seed,
        ),

        "Oracle": lambda: run_oracle_baseline(
            exp_cfg,
            dataset=dataset,
            lr=LR,
            n_epochs=N_EPOCHS,
            seed=seed,
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MATCHED-EPOCHS CIFAR-100 COMPARISON")
    print(f"  n_epochs={N_EPOCHS}, lr={LR}, federation_every={FEDERATION_EVERY}")
    print(f"  K={K}, T={T}, n_samples={N_SAMPLES}, rho={RHO}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 70)

    all_rows: list[dict] = []
    t_start = time.time()

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED = {seed}")
        print(f"{'='*60}")

        cfg = CIFAR100RecurrenceConfig(
            K=K,
            T=T,
            n_samples=N_SAMPLES,
            rho=RHO,
            alpha=ALPHA,
            delta=DELTA,
            n_features=N_FEATURES,
            seed=seed,
            label_split="disjoint",
            min_group_size=2,
        )

        print("  Preparing feature cache...", flush=True)
        prepare_cifar100_recurrence_feature_cache(cfg)
        dataset = generate_cifar100_recurrence_dataset(cfg)

        n_concepts = int(dataset.concept_matrix.max()) + 1
        print(f"  True concepts: {n_concepts}", flush=True)

        # Print label distribution per concept
        for c_id in range(n_concepts):
            cells = list(zip(*np.where(dataset.concept_matrix == c_id)))
            if cells:
                k, t_idx = cells[0]
                _, y = dataset.data[(k, t_idx)]
                unique_labels = np.unique(y)
                print(f"    Concept {c_id} labels: {unique_labels.tolist()}", flush=True)

        exp_cfg = ExperimentConfig(
            generator_config=dataset.config,
            federation_every=FEDERATION_EVERY,
        )

        methods = build_methods(dataset, exp_cfg, seed)

        for method_name, run_fn in methods.items():
            print(f"  Running {method_name}...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = run_fn()
                log = _make_log(method_name, result, dataset.concept_matrix)
                metrics = compute_all_metrics(
                    log,
                    identity_capable=identity_metrics_valid(method_name),
                )
                elapsed = time.time() - t0

                fa = getattr(metrics, "final_accuracy", None) or 0.0
                reid = getattr(metrics, "concept_re_id_accuracy", None) or 0.0
                auc = getattr(metrics, "accuracy_auc", None) or 0.0
                ent = getattr(metrics, "assignment_entropy", None) or 0.0
                total_bytes = getattr(log, "total_bytes", None) or 0.0

                row = {
                    "seed": seed,
                    "method": method_name,
                    "final_acc": round(float(fa), 4),
                    "re_id": round(float(reid), 4),
                    "auc": round(float(auc), 4),
                    "entropy": round(float(ent), 4),
                    "total_bytes": int(total_bytes),
                    "elapsed_s": round(elapsed, 1),
                }
                all_rows.append(row)
                print(
                    f"acc={row['final_acc']:.3f}  "
                    f"re_id={row['re_id']:.3f}  "
                    f"auc={row['auc']:.3f}  "
                    f"bytes={row['total_bytes']}  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
            except Exception:
                print("FAILED", flush=True)
                traceback.print_exc()

    wall_clock = time.time() - t_start

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("  SUMMARY: Matched n_epochs={} lr={} federation_every={}".format(
        N_EPOCHS, LR, FEDERATION_EVERY,
    ))
    print("  Mean +/- std across seeds")
    print("=" * 80)
    header = (
        f"{'Method':<15} {'FinalAcc':<18} {'Re-ID':<18} "
        f"{'AUC':<18} {'Entropy':<12} {'Bytes':<14}"
    )
    print(header)
    print("-" * 80)

    method_names = list(dict.fromkeys(r["method"] for r in all_rows))
    summary_rows = []
    for method in method_names:
        mrows = [r for r in all_rows if r["method"] == method]
        accs = [r["final_acc"] for r in mrows]
        reids = [r["re_id"] for r in mrows]
        aucs = [r["auc"] for r in mrows]
        ents = [r["entropy"] for r in mrows]
        bytess = [r["total_bytes"] for r in mrows]

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_reid = float(np.mean(reids))
        std_reid = float(np.std(reids))
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        mean_ent = float(np.mean(ents))
        mean_bytes = float(np.mean(bytess))

        print(
            f"{method:<15} "
            f"{mean_acc:.4f}+/-{std_acc:.3f}   "
            f"{mean_reid:.4f}+/-{std_reid:.3f}   "
            f"{mean_auc:.4f}+/-{std_auc:.3f}   "
            f"{mean_ent:<12.4f} "
            f"{mean_bytes:<14.0f}"
        )
        summary_rows.append({
            "method": method,
            "mean_final_acc": round(mean_acc, 4),
            "std_final_acc": round(std_acc, 4),
            "mean_re_id": round(mean_reid, 4),
            "std_re_id": round(std_reid, 4),
            "mean_auc": round(mean_auc, 4),
            "std_auc": round(std_auc, 4),
            "mean_entropy": round(mean_ent, 4),
            "mean_total_bytes": int(mean_bytes),
        })

    # Rank by final accuracy
    summary_rows_sorted = sorted(summary_rows, key=lambda r: -r["mean_final_acc"])
    print("\n  RANKED BY FINAL ACCURACY:")
    for i, r in enumerate(summary_rows_sorted, 1):
        print(
            f"  {i}. {r['method']:<15} "
            f"acc={r['mean_final_acc']:.4f}+/-{r['std_final_acc']:.3f}  "
            f"re_id={r['mean_re_id']:.4f}"
        )

    print(f"\n  Wall-clock: {wall_clock:.0f}s")

    # ---- Save results ----
    summary = {
        "experiment": "matched_epochs_comparison",
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "federation_every": FEDERATION_EVERY,
        "K": K,
        "T": T,
        "n_samples": N_SAMPLES,
        "rho": RHO,
        "alpha": ALPHA,
        "delta": DELTA,
        "label_split": "disjoint",
        "min_group_size": 2,
        "seeds": SEEDS,
        "wall_clock_s": round(wall_clock, 1),
        "summary": summary_rows_sorted,
        "raw_rows": all_rows,
    }
    json_path = RESULTS_DIR / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON saved to {json_path}")

    # Also write a human-readable summary
    txt_path = RESULTS_DIR / "summary.txt"
    with open(txt_path, "w") as f:
        f.write("MATCHED-EPOCHS CIFAR-100 COMPARISON\n")
        f.write(f"n_epochs={N_EPOCHS}, lr={LR}, federation_every={FEDERATION_EVERY}\n")
        f.write(f"K={K}, T={T}, n_samples={N_SAMPLES}, rho={RHO}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Wall-clock: {wall_clock:.0f}s\n\n")
        f.write(f"{'Rank':<5} {'Method':<15} {'FinalAcc':<18} {'Re-ID':<18} {'AUC':<18}\n")
        f.write("-" * 75 + "\n")
        for i, r in enumerate(summary_rows_sorted, 1):
            f.write(
                f"{i:<5} {r['method']:<15} "
                f"{r['mean_final_acc']:.4f}+/-{r['std_final_acc']:.3f}   "
                f"{r['mean_re_id']:.4f}+/-{r['std_re_id']:.3f}   "
                f"{r['mean_auc']:.4f}+/-{r['std_auc']:.3f}\n"
            )
    print(f"  TXT saved to {txt_path}")


if __name__ == "__main__":
    main()
