from __future__ import annotations

"""Ablation experiments to identify WHY CFL beats FedProTrack by ~10%.

This script isolates each mechanistic difference between CFL and FedProTrack
through targeted ablations:

  Ablation 0: Fair data-split baseline (CFL with 50/50 split like FPT)
  Ablation 1: FPT with matched training strength (n_epochs=10)
  Ablation 2: CFL without warmup (warmup_rounds=0)
  Ablation 3: FPT with warmup (pure FedAvg for first 20 rounds)
  Ablation 4: FPT without blend_alpha momentum
  Ablation 5: FPT with model-update fingerprints (CFL-style signal)
  Ablation 6: CFL without memory / FPT without memory persistence

The key insight: CFL uses ALL data for both prediction and training,
while FPT splits data 50/50. This is the single biggest confound.
"""

import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fedprotrack.baselines.cfl import (
    CFLClient,
    CFLServer,
    _copy_params,
    _params_to_vector,
    _cosine_sim,
)
from fedprotrack.baselines.comm_tracker import model_bytes
from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.models import TorchLinearClassifier
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


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
            getattr(result, "predicted_concept_matrix"), dtype=np.int32
        ),
        accuracy_curve=np.asarray(
            getattr(result, "accuracy_matrix"), dtype=np.float64
        ),
        total_bytes=total_bytes,
        method_name=method_name,
    )


# ===========================================================================
# ABLATION 0: CFL with fair 50/50 data split (like FPT)
# ===========================================================================

def run_cfl_fair_split(dataset, federation_every=1, **kwargs):
    """CFL with 50/50 train/test split matching FPT's evaluation protocol."""
    from fedprotrack.drift_generator.generator import DriftDataset

    gc = dataset.config
    K, T = gc.K, gc.T

    # Infer dimensions
    first_key = next(iter(dataset.data))
    X0, y0 = dataset.data[first_key]
    n_features = X0.shape[1]
    all_labels = set()
    for _, y in dataset.data.values():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    warmup_rounds = kwargs.get("warmup_rounds", 20)
    n_epochs = kwargs.get("n_epochs", 10)
    lr = kwargs.get("lr", 0.1)

    clients = [
        CFLClient(k, n_features, n_classes, lr=lr, n_epochs=n_epochs, seed=42 + k)
        for k in range(K)
    ]
    server = CFLServer(
        n_features=n_features,
        n_classes=n_classes,
        seed=42,
        warmup_rounds=warmup_rounds,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for client in clients:
        params = server.broadcast(client.client_id)
        if params:
            client.set_model_params(params)

    for t in range(T):
        # Fair split: predict on first half (held-out)
        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            preds = clients[k].predict(X_test)
            accuracy_matrix[k, t] = float(np.mean(preds == y_test)) if len(y_test) > 0 else 0.0

        # Train on second half only
        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_train, y_train = X[mid:], y[mid:]
            clients[k].fit(X_train, y_train)

        if (t + 1) % federation_every == 0 and t < T - 1:
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

        for k in range(K):
            predicted_matrix[k, t] = server.client_cluster_map.get(k, clients[k]._cluster_id)

    return type("MethodResult", (), {
        "accuracy_matrix": accuracy_matrix,
        "predicted_concept_matrix": predicted_matrix,
        "total_bytes": total_bytes,
    })()


# ===========================================================================
# ABLATION 5: FPT with model-update fingerprints
# This replaces fingerprint similarity with cosine sim of update vectors
# ===========================================================================

def run_fpt_with_update_fingerprints(dataset, federation_every=1, seed=42,
                                      n_epochs=1, lr=0.1, blend_alpha=0.0):
    """FPT-style runner that uses model UPDATE vectors for Phase A routing.

    Instead of building data fingerprints, each client computes
    update_vector = params_after - params_before, and Phase A clusters
    clients by cosine similarity of these update vectors (like CFL does).
    """
    gc = dataset.config
    K, T = gc.K, gc.T

    first_key = next(iter(dataset.data))
    X0, _ = dataset.data[first_key]
    n_features = X0.shape[1]
    all_labels = set()
    for _, y in dataset.data.values():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    models = [
        TorchLinearClassifier(
            n_features=n_features, n_classes=n_classes,
            lr=lr, n_epochs=n_epochs, seed=seed + k,
        )
        for k in range(K)
    ]

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0
    prev_assignments: dict[int, int] = {k: 0 for k in range(K)}
    cluster_models: dict[int, dict[str, np.ndarray]] = {}

    for t in range(T):
        update_vectors = []
        model_params_list = []

        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            # Prequential eval
            y_pred = models[k].predict(X_test)
            accuracy_matrix[k, t] = float(np.mean(y_pred == y_test)) if len(y_test) > 0 else 0.0

            # Record params before
            before = {k2: v.copy() for k2, v in models[k].get_params().items()}
            before_vec = _params_to_vector(before)

            # Train
            if n_epochs > 1:
                models[k].fit(X_train, y_train)
            else:
                models[k].partial_fit(X_train, y_train)

            after = models[k].get_params()
            after_vec = _params_to_vector(after)
            update_vec = after_vec - before_vec
            update_vectors.append(update_vec)
            model_params_list.append({k2: v.copy() for k2, v in after.items()})

            # Blend alpha momentum
            if blend_alpha > 0.0 and k in prev_assignments and cluster_models:
                cid = prev_assignments.get(k, 0)
                if cid in cluster_models:
                    models[k].blend_params(cluster_models[cid], alpha=blend_alpha)
                    model_params_list[-1] = {k2: v.copy() for k2, v in models[k].get_params().items()}

        # Federation
        if (t + 1) % federation_every == 0 and t < T - 1:
            # Phase A: cluster by update vector cosine similarity
            # Simple K-means-like assignment using pairwise similarity
            n_concepts = max(2, int(dataset.concept_matrix.max()) + 1)

            # Compute pairwise cosine similarity matrix
            sim_matrix = np.eye(K, dtype=np.float64)
            for i in range(K):
                for j in range(i + 1, K):
                    sim = _cosine_sim(update_vectors[i], update_vectors[j])
                    sim_matrix[i, j] = sim_matrix[j, i] = sim

            # Agglomerative clustering on update vectors
            from sklearn.cluster import AgglomerativeClustering
            dist = np.clip(1.0 - sim_matrix, 0.0, None)
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=min(n_concepts, K),
                    metric="precomputed",
                    linkage="complete",
                ).fit(dist)
            except TypeError:
                clustering = AgglomerativeClustering(
                    n_clusters=min(n_concepts, K),
                    affinity="precomputed",
                    linkage="complete",
                ).fit(dist)
            labels = clustering.labels_

            # Build concept assignments
            new_assignments = {k: int(labels[k]) for k in range(K)}

            # Phase B: aggregate within clusters
            concept_groups: dict[int, list[int]] = {}
            for k, cid in new_assignments.items():
                concept_groups.setdefault(cid, []).append(k)

            new_cluster_models: dict[int, dict[str, np.ndarray]] = {}
            for cid, members in concept_groups.items():
                group_params = [model_params_list[k] for k in members]
                agg: dict[str, np.ndarray] = {}
                for key in group_params[0]:
                    agg[key] = np.mean(np.stack([p[key] for p in group_params]), axis=0)
                new_cluster_models[cid] = agg

            # Distribute
            for k in range(K):
                cid = new_assignments[k]
                models[k].set_params(new_cluster_models[cid])

            # Track bytes
            one_model_bytes = model_bytes(model_params_list[0])
            # Upload: each client sends model + update vector
            total_bytes += K * one_model_bytes
            # Download: each client receives aggregated model
            total_bytes += K * one_model_bytes

            prev_assignments = new_assignments
            cluster_models = new_cluster_models

        for k in range(K):
            predicted_matrix[k, t] = prev_assignments.get(k, 0)

    return type("MethodResult", (), {
        "accuracy_matrix": accuracy_matrix,
        "predicted_concept_matrix": predicted_matrix,
        "total_bytes": total_bytes,
    })()


# ===========================================================================
# Main ablation runner
# ===========================================================================

def run_ablations(seed: int, results_dir: Path) -> dict:
    """Run all ablations for a single seed."""
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")

    # Create CIFAR-100 dataset with disjoint labels
    cfg = CIFAR100RecurrenceConfig(
        K=10,
        T=30,
        n_samples=800,
        rho=3.0,
        alpha=0.75,
        delta=0.85,
        n_features=128,
        seed=seed,
        label_split="disjoint",
        min_group_size=2,
    )
    print(f"Preparing feature cache...")
    prepare_cifar100_recurrence_feature_cache(cfg)
    print(f"Generating dataset (K={cfg.K}, T={cfg.T}, label_split={cfg.label_split})...")
    dataset = generate_cifar100_recurrence_dataset(cfg)
    ground_truth = dataset.concept_matrix
    n_true_concepts = int(ground_truth.max()) + 1
    print(f"  True concepts: {n_true_concepts}")
    print(f"  Concept matrix shape: {ground_truth.shape}")

    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=1,
    )

    results = {}

    methods = {
        # ---- Reference methods ----
        "Oracle": lambda: run_oracle_baseline(exp_cfg, dataset=dataset),
        "FedAvg": lambda: run_fedavg_baseline(exp_cfg, dataset=dataset),
        "FedAvg-10ep": lambda: run_fedavg_baseline(exp_cfg, dataset=dataset, lr=0.1, n_epochs=10, seed=seed),

        # ---- CFL variants ----
        "CFL-original": lambda: run_cfl_full(dataset, federation_every=1),
        "CFL-fair-split": lambda: run_cfl_fair_split(dataset, federation_every=1),
        "CFL-fair-1ep": lambda: run_cfl_fair_split(dataset, federation_every=1, n_epochs=1),
        "CFL-no-warmup": lambda: run_cfl_fair_split(dataset, federation_every=1, warmup_rounds=0),

        # ---- FPT variants ----
        "FPT-base": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, n_true_concepts + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=1,
            detector_name="ADWIN",
            seed=seed,
            lr=0.1,
            n_epochs=1,
            soft_aggregation=True,
            blend_alpha=0.0,
        ).run(dataset),

        "FPT-10ep": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, n_true_concepts + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=1,
            detector_name="ADWIN",
            seed=seed,
            lr=0.1,
            n_epochs=10,
            soft_aggregation=True,
            blend_alpha=0.0,
        ).run(dataset),

        "FPT-10ep-blend05": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, n_true_concepts + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=1,
            detector_name="ADWIN",
            seed=seed,
            lr=0.1,
            n_epochs=10,
            soft_aggregation=True,
            blend_alpha=0.5,
        ).run(dataset),

        "FPT-update-fp": lambda: run_fpt_with_update_fingerprints(
            dataset, federation_every=1, seed=seed, n_epochs=1, lr=0.1, blend_alpha=0.0,
        ),

        "FPT-update-fp-10ep": lambda: run_fpt_with_update_fingerprints(
            dataset, federation_every=1, seed=seed, n_epochs=10, lr=0.1, blend_alpha=0.0,
        ),

        "FPT-calibrated": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, n_true_concepts + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=1,
            detector_name="ADWIN",
            seed=seed,
            lr=0.1,
            n_epochs=1,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        ).run(dataset),

        "FPT-hybrid-10ep": lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, n_true_concepts + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=1,
            detector_name="ADWIN",
            seed=seed,
            lr=0.1,
            n_epochs=10,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
            model_signature_weight=0.55,
            model_signature_dim=8,
        ).run(dataset),
    }

    for method_name, runner in methods.items():
        print(f"\n  Running {method_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = runner()
            log = _make_log(method_name, result, ground_truth)
            metrics = compute_all_metrics(log)
            elapsed = time.time() - t0

            acc = float(np.mean(log.accuracy_curve))
            final_acc = float(np.mean(log.accuracy_curve[:, -1]))
            re_id = metrics.concept_re_id_accuracy if metrics.concept_re_id_accuracy is not None else float("nan")
            total_bytes = getattr(result, "total_bytes", None)
            if total_bytes is None:
                total_bytes = log.total_bytes

            results[method_name] = {
                "mean_acc": round(acc, 4),
                "final_acc": round(final_acc, 4),
                "re_id": round(re_id, 4) if not np.isnan(re_id) else None,
                "total_bytes": float(total_bytes) if total_bytes else None,
                "elapsed": round(elapsed, 1),
            }
            print(f"acc={acc:.3f}  final={final_acc:.3f}  re_id={re_id:.3f}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            traceback.print_exc()
            results[method_name] = {"error": str(e)}

    return results


def main():
    results_dir = Path("E:/fedprotrack/.claude/worktrees/elegant-poitras/tmp/cfl_ablation")
    results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 123, 456]
    all_results: dict[int, dict] = {}

    for seed in seeds:
        seed_results = run_ablations(seed, results_dir)
        all_results[seed] = seed_results

        # Save per-seed
        with open(results_dir / f"seed_{seed}.json", "w") as f:
            json.dump(seed_results, f, indent=2, default=str)

    # Aggregate across seeds
    print("\n" + "=" * 80)
    print("  AGGREGATE RESULTS (mean +/- std across seeds)")
    print("=" * 80)

    method_names = sorted(set().union(*[r.keys() for r in all_results.values()]))

    summary_rows = []
    for method in method_names:
        accs = []
        final_accs = []
        re_ids = []
        for seed in seeds:
            r = all_results.get(seed, {}).get(method, {})
            if "error" in r:
                continue
            if "mean_acc" in r:
                accs.append(r["mean_acc"])
            if "final_acc" in r:
                final_accs.append(r["final_acc"])
            if r.get("re_id") is not None:
                re_ids.append(r["re_id"])

        row = {
            "method": method,
            "mean_acc": f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}" if accs else "N/A",
            "final_acc": f"{np.mean(final_accs):.3f} +/- {np.std(final_accs):.3f}" if final_accs else "N/A",
            "re_id": f"{np.mean(re_ids):.3f} +/- {np.std(re_ids):.3f}" if re_ids else "N/A",
            "mean_acc_val": float(np.mean(accs)) if accs else 0.0,
        }
        summary_rows.append(row)
        print(f"  {method:30s}  acc={row['mean_acc']:20s}  final={row['final_acc']:20s}  re_id={row['re_id']:20s}")

    # Sort by accuracy
    summary_rows.sort(key=lambda r: -r["mean_acc_val"])

    with open(results_dir / "aggregate_results.json", "w") as f:
        json.dump({
            "seeds": seeds,
            "per_seed": {str(s): all_results[s] for s in seeds},
            "summary": summary_rows,
        }, f, indent=2, default=str)

    # Print ranked table
    print("\n  RANKED BY ACCURACY:")
    print(f"  {'Rank':>4s}  {'Method':30s}  {'Mean Acc':20s}  {'Final Acc':20s}  {'Re-ID':20s}")
    print(f"  {'----':>4s}  {'------':30s}  {'--------':20s}  {'---------':20s}  {'-----':20s}")
    for i, row in enumerate(summary_rows, 1):
        print(f"  {i:4d}  {row['method']:30s}  {row['mean_acc']:20s}  {row['final_acc']:20s}  {row['re_id']:20s}")

    print(f"\n  Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
