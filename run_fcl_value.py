from __future__ import annotations

"""Measure FedProTrack's downstream value for Federated Continual Learning.

Three metrics capture practical FCL benefits:
1. Recovery Gap: accuracy gain at concept recurrence vs FedAvg baseline
2. Time-to-Recovery: steps to reach 80% of concept peak accuracy after switch
3. Knowledge Preservation: how well stored concept models retain performance

Runs on CIFAR-100 disjoint (K=10, T=30, n_epochs=5, label_split="disjoint",
min_group_size=2, rho=7.5, seeds=[42,123,456]).
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

from fedprotrack.baselines.runners import run_cfl_full, MethodResult
from fedprotrack.experiment.baselines import run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]
RESULTS_DIR = Path("tmp/fcl_value")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DS_CFG = dict(
    K=10,
    T=30,
    n_samples=400,
    rho=7.5,
    alpha=0.75,
    delta=0.85,
    n_features=128,
    label_split="disjoint",
    min_group_size=2,
)

FPT_CONFIG_OVERRIDES = dict(
    omega=2.0,
    kappa=0.7,
    loss_novelty_threshold=0.02,
    sticky_dampening=1.0,
    merge_threshold=0.80,
    max_concepts=6,
)

FPT_RUNNER_KWARGS = dict(
    federation_every=2,
    detector_name="ADWIN",
    similarity_calibration=True,
    soft_aggregation=False,
    blend_alpha=0.5,
    lr=0.1,
    n_epochs=5,
)


# ---------------------------------------------------------------------------
# FCL metric helpers
# ---------------------------------------------------------------------------

def _find_recurrence_events(concept_matrix: np.ndarray) -> list[dict]:
    """Find all recurrence events in the concept matrix.

    A recurrence event for client k at time t means:
    - concept_matrix[k, t] == concept_matrix[k, t_prev] for some t_prev < t
    - There exists at least one intervening step t' with t_prev < t' < t
      where concept_matrix[k, t'] != concept_matrix[k, t]
    """
    K, T = concept_matrix.shape
    events = []
    for k in range(K):
        for t in range(2, T):
            c = int(concept_matrix[k, t])
            # Find previous occurrence of same concept with gap
            for t_prev in range(t - 1, -1, -1):
                if int(concept_matrix[k, t_prev]) == c:
                    # Check there is at least one different concept between
                    has_gap = False
                    for t_mid in range(t_prev + 1, t):
                        if int(concept_matrix[k, t_mid]) != c:
                            has_gap = True
                            break
                    if has_gap:
                        events.append({
                            "client": k,
                            "t_recur": t,
                            "t_prev": t_prev,
                            "concept": c,
                        })
                    break  # found most recent previous, stop
    return events


def _find_switch_events(concept_matrix: np.ndarray) -> list[dict]:
    """Find all concept switch events.

    A switch at (k, t) means concept_matrix[k, t] != concept_matrix[k, t-1].
    """
    K, T = concept_matrix.shape
    events = []
    for k in range(K):
        for t in range(1, T):
            c_prev = int(concept_matrix[k, t - 1])
            c_new = int(concept_matrix[k, t])
            if c_prev != c_new:
                events.append({
                    "client": k,
                    "t_switch": t,
                    "concept_from": c_prev,
                    "concept_to": c_new,
                })
    return events


def compute_recovery_gaps(
    acc_matrices: dict[str, np.ndarray],
    concept_matrix: np.ndarray,
) -> dict[str, float]:
    """Metric 1: Recovery gap at recurrence points vs FedAvg."""
    events = _find_recurrence_events(concept_matrix)
    if not events:
        return {m: 0.0 for m in acc_matrices}

    fedavg_acc = acc_matrices["FedAvg"]
    gaps: dict[str, list[float]] = {m: [] for m in acc_matrices}

    for ev in events:
        k, t = ev["client"], ev["t_recur"]
        base = float(fedavg_acc[k, t])
        for method, acc in acc_matrices.items():
            gaps[method].append(float(acc[k, t]) - base)

    return {m: float(np.mean(vals)) if vals else 0.0 for m, vals in gaps.items()}


def compute_time_to_recovery(
    acc_matrices: dict[str, np.ndarray],
    concept_matrix: np.ndarray,
    threshold_frac: float = 0.80,
    max_horizon: int = 5,
) -> dict[str, float]:
    """Metric 2: Steps until accuracy reaches 80% of concept peak.

    For each concept switch, look forward up to max_horizon steps and find
    when accuracy reaches threshold_frac * peak accuracy for that concept
    on that client.
    """
    K, T = concept_matrix.shape
    switches = _find_switch_events(concept_matrix)
    if not switches:
        return {m: 0.0 for m in acc_matrices}

    ttr: dict[str, list[int]] = {m: [] for m in acc_matrices}

    for ev in switches:
        k, t_sw = ev["client"], ev["t_switch"]
        c_new = ev["concept_to"]

        for method, acc in acc_matrices.items():
            # Find peak accuracy for this concept on this client
            concept_steps = [
                t for t in range(T) if int(concept_matrix[k, t]) == c_new
            ]
            if not concept_steps:
                continue
            peak = max(float(acc[k, ts]) for ts in concept_steps)
            target = threshold_frac * peak if peak > 0 else 0.0

            # Count steps from switch until target reached
            recovered = False
            for dt in range(max_horizon + 1):
                t_check = t_sw + dt
                if t_check >= T:
                    break
                if int(concept_matrix[k, t_check]) != c_new:
                    break
                if float(acc[k, t_check]) >= target:
                    ttr[method].append(dt)
                    recovered = True
                    break
            if not recovered:
                ttr[method].append(max_horizon)

    return {
        m: float(np.mean(vals)) if vals else float(max_horizon)
        for m, vals in ttr.items()
    }


def compute_knowledge_preservation(
    acc_matrices: dict[str, np.ndarray],
    concept_matrix: np.ndarray,
) -> dict[str, float]:
    """Metric 3: Knowledge preservation across dormancy.

    For each recurrence event, compare accuracy at recurrence with accuracy
    at previous occurrence. Ratio close to 1.0 means knowledge preserved.
    """
    events = _find_recurrence_events(concept_matrix)
    if not events:
        return {m: 1.0 for m in acc_matrices}

    preservation: dict[str, list[float]] = {m: [] for m in acc_matrices}

    for ev in events:
        k = ev["client"]
        t_prev = ev["t_prev"]
        t_recur = ev["t_recur"]

        for method, acc in acc_matrices.items():
            prev_acc = float(acc[k, t_prev])
            recur_acc = float(acc[k, t_recur])
            if prev_acc > 0.05:  # avoid division by near-zero
                preservation[method].append(recur_acc / prev_acc)
            else:
                # If previous accuracy was near zero, skip
                pass

    return {
        m: float(np.mean(vals)) if vals else 1.0
        for m, vals in preservation.items()
    }


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------

def run_fpt(dataset, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Run FedProTrack and return (accuracy_matrix, predicted_matrix, bytes)."""
    n_concepts = int(dataset.concept_matrix.max()) + 1
    overrides = dict(FPT_CONFIG_OVERRIDES)
    overrides["max_concepts"] = max(6, n_concepts + 2)
    config = TwoPhaseConfig(**overrides)
    runner = FedProTrackRunner(
        config=config,
        seed=seed,
        **FPT_RUNNER_KWARGS,
    )
    result = runner.run(dataset)
    return result.accuracy_matrix, result.predicted_concept_matrix, result.total_bytes


def run_fedavg(dataset, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Run FedAvg and return (accuracy_matrix, predicted_matrix, bytes)."""
    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=2,
    )
    result = run_fedavg_baseline(
        exp_cfg, dataset=dataset, lr=0.1, n_epochs=5, seed=seed,
    )
    return result.accuracy_matrix, result.predicted_concept_matrix, result.total_bytes


def run_cfl(dataset, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Run CFL and return (accuracy_matrix, predicted_matrix, bytes).

    CFL's CFLClient uses n_epochs internally. We pass matched parameters
    by constructing CFL clients with n_epochs=5.
    """
    from fedprotrack.baselines.cfl import CFLClient, CFLServer
    from fedprotrack.baselines.comm_tracker import model_bytes

    K, T = dataset.config.K, dataset.config.T
    X0, y0 = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    all_labels: set[int] = set()
    for (_, _y), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    clients = [
        CFLClient(k, n_features, n_classes, lr=0.1, n_epochs=5, seed=seed + k)
        for k in range(K)
    ]
    server = CFLServer(
        n_features=n_features,
        n_classes=n_classes,
        seed=seed,
        eps_1=0.4,
        eps_2=1.6,
        warmup_rounds=20,
        max_clusters=8,
    )

    accuracy_matrix = np.zeros((K, T), dtype=np.float64)
    predicted_matrix = np.zeros((K, T), dtype=np.int32)
    total_bytes = 0.0

    for client in clients:
        params = server.broadcast(client.client_id)
        if params:
            client.set_model_params(params)
        client._cluster_id = server.client_cluster_map.get(client.client_id, 0)

    for t in range(T):
        for k in range(K):
            X, y = dataset.data[(k, t)]
            preds = clients[k].predict(X)
            accuracy_matrix[k, t] = float(np.mean(y == preds))

        for k in range(K):
            X, y = dataset.data[(k, t)]
            clients[k].fit(X, y)

        if (t + 1) % 2 == 0 and t < T - 1:
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
            predicted_matrix[k, t] = server.client_cluster_map.get(
                k, clients[k]._cluster_id
            )

    return accuracy_matrix, predicted_matrix, total_bytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("FCL Value Experiment: CIFAR-100 Disjoint")
    print(f"K={DS_CFG['K']}, T={DS_CFG['T']}, rho={DS_CFG['rho']}, "
          f"seeds={SEEDS}, n_epochs=5")
    print("=" * 70)

    # Pre-build feature cache
    print("\n[1/4] Building CIFAR-100 feature cache...")
    cache_cfg = CIFAR100RecurrenceConfig(**DS_CFG)
    prepare_cifar100_recurrence_feature_cache(cache_cfg)
    print("  Cache ready.")

    # Accumulate per-seed results
    all_recovery_gaps: dict[str, list[float]] = defaultdict(list)
    all_ttr: dict[str, list[float]] = defaultdict(list)
    all_preservation: dict[str, list[float]] = defaultdict(list)
    all_mean_acc: dict[str, list[float]] = defaultdict(list)
    all_recurrence_counts: list[int] = []
    all_switch_counts: list[int] = []

    methods = ["FedProTrack", "FedAvg", "CFL"]
    method_runners = {
        "FedProTrack": run_fpt,
        "FedAvg": run_fedavg,
        "CFL": run_cfl,
    }

    t_start = time.time()

    for si, seed in enumerate(SEEDS):
        print(f"\n[2/4] Seed {seed} ({si+1}/{len(SEEDS)})...")

        ds_cfg = CIFAR100RecurrenceConfig(**{**DS_CFG, "seed": seed})
        dataset = generate_cifar100_recurrence_dataset(ds_cfg)
        concept_matrix = dataset.concept_matrix

        recurrence_events = _find_recurrence_events(concept_matrix)
        switch_events = _find_switch_events(concept_matrix)
        all_recurrence_counts.append(len(recurrence_events))
        all_switch_counts.append(len(switch_events))

        print(f"  Concept matrix shape: {concept_matrix.shape}, "
              f"n_concepts={int(concept_matrix.max())+1}")
        print(f"  Recurrence events: {len(recurrence_events)}, "
              f"Switch events: {len(switch_events)}")

        acc_matrices: dict[str, np.ndarray] = {}

        for method in methods:
            print(f"  Running {method}...", end=" ", flush=True)
            t0 = time.time()
            acc_mat, pred_mat, total_bytes = method_runners[method](dataset, seed)
            elapsed = time.time() - t0
            acc_matrices[method] = acc_mat
            mean_acc = float(acc_mat.mean())
            all_mean_acc[method].append(mean_acc)
            print(f"done ({elapsed:.1f}s, mean_acc={mean_acc:.3f}, "
                  f"bytes={total_bytes:.0f})")

        # Compute per-seed metrics
        rg = compute_recovery_gaps(acc_matrices, concept_matrix)
        ttr = compute_time_to_recovery(acc_matrices, concept_matrix)
        kp = compute_knowledge_preservation(acc_matrices, concept_matrix)

        for m in methods:
            all_recovery_gaps[m].append(rg[m])
            all_ttr[m].append(ttr[m])
            all_preservation[m].append(kp[m])

        print(f"  Recovery gaps: { {m: f'{rg[m]:+.4f}' for m in methods} }")
        print(f"  Time-to-recovery: { {m: f'{ttr[m]:.2f}' for m in methods} }")
        print(f"  Knowledge preservation: { {m: f'{kp[m]:.3f}' for m in methods} }")

    wall_clock = time.time() - t_start

    # ---------------------------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[3/4] Aggregated Results (mean +/- std over seeds)")
    print("=" * 70)

    results = {}
    for m in methods:
        rg_mean = float(np.mean(all_recovery_gaps[m]))
        rg_std = float(np.std(all_recovery_gaps[m]))
        ttr_mean = float(np.mean(all_ttr[m]))
        ttr_std = float(np.std(all_ttr[m]))
        kp_mean = float(np.mean(all_preservation[m]))
        kp_std = float(np.std(all_preservation[m]))
        acc_mean = float(np.mean(all_mean_acc[m]))
        acc_std = float(np.std(all_mean_acc[m]))
        results[m] = {
            "recovery_gap_mean": rg_mean,
            "recovery_gap_std": rg_std,
            "time_to_recovery_mean": ttr_mean,
            "time_to_recovery_std": ttr_std,
            "knowledge_preservation_mean": kp_mean,
            "knowledge_preservation_std": kp_std,
            "mean_accuracy": acc_mean,
            "mean_accuracy_std": acc_std,
        }

    # Print table
    header = f"{'Method':<15} {'RecovGap':>12} {'TTR (steps)':>14} {'KnowPres':>12} {'MeanAcc':>12}"
    print(header)
    print("-" * len(header))
    for m in methods:
        r = results[m]
        print(f"{m:<15} "
              f"{r['recovery_gap_mean']:+.4f}+/-{r['recovery_gap_std']:.4f} "
              f"{r['time_to_recovery_mean']:.2f}+/-{r['time_to_recovery_std']:.2f}   "
              f"{r['knowledge_preservation_mean']:.3f}+/-{r['knowledge_preservation_std']:.3f} "
              f"{r['mean_accuracy']:.3f}+/-{r['mean_accuracy_std']:.3f}")

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    print(f"\n[4/4] Saving results to {RESULTS_DIR}")

    with open(RESULTS_DIR / "recovery_gaps.json", "w") as f:
        json.dump({
            "description": "Mean recovery gap vs FedAvg at concept recurrence points",
            "per_seed": {m: all_recovery_gaps[m] for m in methods},
            "aggregate": {m: results[m]["recovery_gap_mean"] for m in methods},
        }, f, indent=2)

    with open(RESULTS_DIR / "time_to_recovery.json", "w") as f:
        json.dump({
            "description": "Mean steps to reach 80% of concept peak accuracy after switch",
            "per_seed": {m: all_ttr[m] for m in methods},
            "aggregate": {m: results[m]["time_to_recovery_mean"] for m in methods},
        }, f, indent=2)

    with open(RESULTS_DIR / "knowledge_preservation.json", "w") as f:
        json.dump({
            "description": "Ratio of accuracy at recurrence / accuracy at previous occurrence",
            "per_seed": {m: all_preservation[m] for m in methods},
            "aggregate": {m: results[m]["knowledge_preservation_mean"] for m in methods},
        }, f, indent=2)

    # Summary
    summary_lines = [
        "FCL Value Experiment Summary",
        "=" * 50,
        f"Dataset: CIFAR-100 disjoint, K={DS_CFG['K']}, T={DS_CFG['T']}, rho={DS_CFG['rho']}",
        f"Seeds: {SEEDS}",
        f"n_epochs: 5 (all methods)",
        f"federation_every: 2",
        f"Wall-clock: {wall_clock:.1f}s",
        f"Mean recurrence events per seed: {np.mean(all_recurrence_counts):.1f}",
        f"Mean switch events per seed: {np.mean(all_switch_counts):.1f}",
        "",
        "Metric 1: Recovery Gap (vs FedAvg at recurrence points)",
        "  Higher = better recovery from concept memory",
    ]
    for m in methods:
        r = results[m]
        summary_lines.append(
            f"  {m}: {r['recovery_gap_mean']:+.4f} +/- {r['recovery_gap_std']:.4f}"
        )
    summary_lines += [
        "",
        "Metric 2: Time-to-Recovery (steps to 80% peak after switch)",
        "  Lower = faster adaptation",
    ]
    for m in methods:
        r = results[m]
        summary_lines.append(
            f"  {m}: {r['time_to_recovery_mean']:.2f} +/- {r['time_to_recovery_std']:.2f}"
        )
    summary_lines += [
        "",
        "Metric 3: Knowledge Preservation (recurrence_acc / previous_acc)",
        "  Closer to 1.0 = less catastrophic forgetting",
    ]
    for m in methods:
        r = results[m]
        summary_lines.append(
            f"  {m}: {r['knowledge_preservation_mean']:.3f} +/- {r['knowledge_preservation_std']:.3f}"
        )
    summary_lines += [
        "",
        "Overall Mean Accuracy:",
    ]
    for m in methods:
        r = results[m]
        summary_lines.append(
            f"  {m}: {r['mean_accuracy']:.3f} +/- {r['mean_accuracy_std']:.3f}"
        )
    summary_lines += [
        "",
        "Interpretation:",
        "  - Recovery Gap: FPT should show positive gap because it loads stored",
        "    concept models at recurrence instead of retraining from scratch.",
        "  - Time-to-Recovery: FPT with correct re-ID should recover in 0-1 steps",
        "    (model pre-loaded) while FedAvg must retrain from the global model.",
        "  - Knowledge Preservation: FPT stores concept models in memory bank,",
        "    so preservation should be near 1.0. FedAvg has no concept memory.",
    ]

    summary_text = "\n".join(summary_lines)
    with open(RESULTS_DIR / "summary.txt", "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
