"""Recurrence-with-gap experiment: the scenario where FPT should beat IFCA.

Setup:
  - 8 total concepts (disjoint class subsets from 20 coarse classes)
  - Only 2-3 concepts active at any time
  - T=30: phase1 (concepts 0,1,2) → phase2 (concepts 3,4,5) → phase3 (0,1,2 recur)
  - K=12 clients (avg 4 per active concept → meaningful aggregation)

IFCA's dilemma:
  - n_clusters=3: can't represent all 8 concepts → forced collapse
  - n_clusters=8: broadcasts 8 models/round, most are stale/irrelevant

FPT's advantage:
  - Memory bank stores dormant concept models
  - Phase A fingerprint is lightweight (doesn't scale with total concepts)
  - On recurrence, recalls stored model → instant recovery
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18

from fedprotrack.baselines.comm_tracker import model_bytes
from fedprotrack.baselines.runners import run_ifca_full
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.data_streams import ConceptSpec
from fedprotrack.drift_generator.generator import DriftDataset
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.models import TorchLinearClassifier
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig

os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

# ── 8 concepts: disjoint pairs of coarse classes ──
# Each concept gets 5 coarse classes (disjoint) = maximally different
CONCEPT_CLASSES = {
    0: [0, 1, 2, 3, 4],       # group A
    1: [5, 6, 7, 8, 9],       # group A
    2: [10, 11, 12, 13, 14],   # group A
    3: [15, 16, 17, 18, 19],   # group B
    4: [0, 1, 5, 6, 10],       # group B (recombined)
    5: [11, 15, 16, 2, 3],     # group B
    6: [4, 7, 8, 17, 18],      # group C (for phase 3 novelty)
    7: [9, 12, 13, 14, 19],    # group C
}
N_CLASSES_PER_CONCEPT = 5


def build_recurrence_concept_matrix(K: int, T: int, seed: int) -> np.ndarray:
    """Build a concept matrix with deliberate long-gap recurrence.

    Phase 1 (t=0..9):   concepts 0,1,2 active
    Phase 2 (t=10..19):  concepts 3,4,5 active (0,1,2 dormant)
    Phase 3 (t=20..29):  concepts 0,1,2 recur (3,4,5 dormant)

    Each client is assigned to one of the active concepts per phase.
    """
    rng = np.random.RandomState(seed)
    cm = np.zeros((K, T), dtype=np.int32)

    phase_concepts = [
        [0, 1, 2],   # phase 1
        [3, 4, 5],   # phase 2
        [0, 1, 2],   # phase 3 (recurrence!)
    ]
    phase_boundaries = [0, 10, 20, T]

    for phase_idx in range(3):
        t_start = phase_boundaries[phase_idx]
        t_end = phase_boundaries[phase_idx + 1]
        active = phase_concepts[phase_idx]

        for k in range(K):
            # Assign each client to one of the active concepts
            concept = active[k % len(active)]
            # Add some drift within phase (client switches between active concepts)
            for t in range(t_start, t_end):
                if t > t_start and rng.random() < 0.15:  # 15% drift chance
                    concept = active[rng.randint(len(active))]
                cm[k, t] = concept

    return cm


def _load_cifar100(data_root: str):
    datasets.CIFAR100(root=data_root, train=True, download=True)
    path = Path(data_root) / "cifar-100-python" / "train"
    with open(path, "rb") as f:
        payload = pickle.load(f, encoding="latin1")
    images = payload["data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    coarse_labels = np.asarray(payload["coarse_labels"], dtype=np.int64)
    return images, coarse_labels


def _extract_features(images, batch_size=256):
    cache_path = Path(".feature_cache") / "cifar100_raw_resnet18.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()
    parts = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[i:i+batch_size]).float() / 255.0
            batch = torch.stack([preprocess(img) for img in batch])
            feats = backbone(batch.to(device, non_blocking=True))
            parts.append(feats.cpu().numpy())
    all_feats = np.concatenate(parts, axis=0).astype(np.float32)
    np.save(cache_path, all_feats)
    return all_feats


def build_dataset(K, T, n_samples, n_features, seed, data_root=".cifar100_cache"):
    images, coarse_labels = _load_cifar100(data_root)
    raw_features = _extract_features(images)
    pca = PCA(n_components=n_features, random_state=2718)
    features = pca.fit_transform(raw_features).astype(np.float32)

    concept_matrix = build_recurrence_concept_matrix(K, T, seed)
    actual_n_concepts = int(concept_matrix.max()) + 1

    # Per-concept pools
    concept_pools = {}
    for cid in range(actual_n_concepts):
        cls_subset = CONCEPT_CLASSES.get(cid, list(range(N_CLASSES_PER_CONCEPT)))
        mask = np.isin(coarse_labels, cls_subset)
        X_pool = features[mask]
        y_raw = coarse_labels[mask]
        label_map = {c: i for i, c in enumerate(sorted(set(cls_subset)))}
        y_pool = np.array([label_map[int(y)] for y in y_raw], dtype=np.int64)
        concept_pools[cid] = (X_pool, y_pool)

    data = {}
    for k in range(K):
        for t in range(T):
            cid = int(concept_matrix[k, t])
            rng = np.random.RandomState(seed + 10000 + k * T + t)
            X_pool, y_pool = concept_pools[cid]
            classes = np.unique(y_pool)
            per_class = n_samples // len(classes)
            remainder = n_samples % len(classes)
            chosen = []
            for off, cls in enumerate(classes):
                cls_idx = np.flatnonzero(y_pool == cls)
                take = per_class + (1 if off < remainder else 0)
                chosen.append(rng.choice(cls_idx, size=take, replace=True))
            batch_idx = np.concatenate(chosen)
            rng.shuffle(batch_idx)
            data[(k, t)] = (X_pool[batch_idx], y_pool[batch_idx])

    gen_config = GeneratorConfig(
        K=K, T=T, n_samples=n_samples, rho=2.5, alpha=0.7, delta=0.85,
        generator_type="cifar100_recurrence", seed=seed)
    concept_specs = [ConceptSpec(concept_id=cid, generator_type="cifar100_recurrence",
                                  variant=cid, noise_scale=0.0)
                     for cid in range(actual_n_concepts)]
    return DriftDataset(concept_matrix=concept_matrix, data=data,
                        config=gen_config, concept_specs=concept_specs)


# ── Runners ──

def run_local(ds, epochs, lr, seed):
    K, T = ds.config.K, ds.config.T
    nf = ds.data[(0,0)][0].shape[1]
    nc = max(len(np.unique(ds.data[(k,t)][1])) for k in range(K) for t in range(T))
    models = [TorchLinearClassifier(n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed+k)
              for k in range(K)]
    acc = np.zeros((K, T), dtype=np.float64)
    for t in range(T):
        for k in range(K):
            X, y = ds.data[(k, t)]
            mid = len(X) // 2
            acc[k, t] = float(np.mean(models[k].predict(X[:mid]) == y[:mid]))
            if epochs > 1:
                models[k].fit(X[mid:], y[mid:])
            else:
                models[k].partial_fit(X[mid:], y[mid:])
    return acc, 0.0


def run_oracle(ds, fed_every, epochs, lr, seed):
    K, T = ds.config.K, ds.config.T
    gt = ds.concept_matrix
    nf = ds.data[(0,0)][0].shape[1]
    nc = max(len(np.unique(ds.data[(k,t)][1])) for k in range(K) for t in range(T))
    models = [TorchLinearClassifier(n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed+k)
              for k in range(K)]
    acc = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0
    # Store best model per concept for warm-start on recurrence
    concept_models: dict[int, dict[str, np.ndarray]] = {}
    for t in range(T):
        for k in range(K):
            X, y = ds.data[(k, t)]
            mid = len(X) // 2
            # On concept recurrence, warm-start from stored model
            cid = int(gt[k, t])
            if t > 0 and gt[k, t] != gt[k, t-1] and cid in concept_models:
                models[k].set_params(concept_models[cid])
            acc[k, t] = float(np.mean(models[k].predict(X[:mid]) == y[:mid]))
            if epochs > 1:
                models[k].fit(X[mid:], y[mid:])
            else:
                models[k].partial_fit(X[mid:], y[mid:])
        if (t + 1) % fed_every == 0 and t < T - 1:
            plist = [m.get_params() for m in models]
            one_b = model_bytes(plist[0], precision_bits=32)
            concepts_at_t = {k: int(gt[k, t]) for k in range(K)}
            for cid in set(concepts_at_t.values()):
                members = [k for k in range(K) if concepts_at_t[k] == cid]
                if len(members) < 2:
                    continue
                cp = {key: np.mean(np.stack([plist[k][key] for k in members]), axis=0)
                      for key in plist[0]}
                for k in members:
                    models[k].set_params(cp)
                total_bytes += len(members) * one_b * 2
                concept_models[cid] = {k: v.copy() for k, v in cp.items()}
    return acc, total_bytes


def run_fpt(ds, fed_every, epochs, lr, seed,
            dormant_recall=False,
            loss_novelty_threshold=0.15,
            merge_threshold=0.85,
            max_concepts=10):
    gt = ds.concept_matrix
    n_concepts = int(gt.max()) + 1
    runner = FedProTrackRunner(
        config=TwoPhaseConfig(
            omega=2.0, kappa=0.7,
            novelty_threshold=0.25,
            loss_novelty_threshold=loss_novelty_threshold,
            sticky_dampening=1.5, sticky_posterior_gate=0.35,
            merge_threshold=merge_threshold, min_count=5.0,
            max_concepts=max(max_concepts, n_concepts + 2),
            merge_every=2, shrink_every=8,
        ),
        federation_every=fed_every, detector_name="ADWIN",
        seed=seed, lr=lr, n_epochs=epochs,
        soft_aggregation=True, blend_alpha=0.0,
        dormant_recall=dormant_recall,
    )
    result = runner.run(ds)
    return result.accuracy_matrix, result.total_bytes, result.spawned_concepts, result.merged_concepts, result.active_concepts


def run_fedavg(ds, fed_every, epochs, lr, seed):
    K, T = ds.config.K, ds.config.T
    nf = ds.data[(0,0)][0].shape[1]
    nc = max(len(np.unique(ds.data[(k,t)][1])) for k in range(K) for t in range(T))
    gm = TorchLinearClassifier(n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed)
    cms = [TorchLinearClassifier(n_features=nf, n_classes=nc, lr=lr, n_epochs=epochs, seed=seed+k)
           for k in range(K)]
    acc = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0
    for t in range(T):
        for k in range(K):
            X, y = ds.data[(k, t)]
            mid = len(X) // 2
            acc[k, t] = float(np.mean(gm.predict(X[:mid]) == y[:mid]))
            if t > 0:
                cms[k].set_params(gm.get_params())
            if epochs > 1:
                cms[k].fit(X[mid:], y[mid:])
            else:
                cms[k].partial_fit(X[mid:], y[mid:])
        if (t + 1) % fed_every == 0 and t < T - 1:
            pl = [m.get_params() for m in cms]
            one_b = model_bytes(pl[0], precision_bits=32)
            total_bytes += K * one_b * 2
            gp = {key: np.mean(np.stack([p[key] for p in pl]), axis=0) for key in pl[0]}
            gm.set_params(gp)
    return acc, total_bytes


# ── Main ──

def main():
    results_dir = Path("results_cifar100_recurrence_gap")
    results_dir.mkdir(parents=True, exist_ok=True)

    K, T = 12, 30
    N_SAMPLES = 200
    N_FEATURES = 128
    EPOCHS, LR = 5, 0.05
    seeds = [42, 123, 456]

    print("=" * 65)
    print("Recurrence-with-Gap Experiment")
    print(f"K={K}, T={T}, 8 total concepts, 3 active/phase")
    print("Phase 1 (t=0-9): concepts 0,1,2")
    print("Phase 2 (t=10-19): concepts 3,4,5  (0,1,2 dormant)")
    print("Phase 3 (t=20-29): concepts 0,1,2 RECUR")
    print("=" * 65)

    all_rows = []
    all_curves: dict[str, list[np.ndarray]] = {}

    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        ds = build_dataset(K, T, N_SAMPLES, N_FEATURES, seed)
        gt = ds.concept_matrix
        print(f"  Concept matrix sample (client 0): {gt[0].tolist()}")

        fpt_configs = [
            # (name, dormant_recall, loss_novelty_th, merge_th, max_concepts)
            ("FPT", False, 0.15, 0.85, 10),
            ("FPT-tight", False, 0.5, 0.6, 8),  # conservative spawn, aggressive merge
            ("FPT-tight+DR", True, 0.5, 0.6, 8),  # + model probing
        ]

        all_methods = [
            ("LocalOnly", None),
            ("FedAvg f=2", None),
            ("IFCA-3 f=2", None),
            ("IFCA-8 f=2", None),
            ("Oracle f=2", None),
        ]
        for cfg_name, dr, lnt, mt, mc in fpt_configs:
            all_methods.append((f"{cfg_name} f=2", (dr, lnt, mt, mc)))

        for name, fpt_cfg in all_methods:
            extra = ""
            if name == "LocalOnly":
                acc_mat, tb = run_local(ds, EPOCHS, LR, seed)
            elif name == "FedAvg f=2":
                acc_mat, tb = run_fedavg(ds, 2, EPOCHS, LR, seed)
            elif name.startswith("IFCA"):
                nc = 3 if "3" in name else 8
                ifca_res = run_ifca_full(ds, federation_every=2, n_clusters=nc, lr=LR, n_epochs=EPOCHS)
                acc_mat, tb = ifca_res.accuracy_matrix, ifca_res.total_bytes
            elif name == "Oracle f=2":
                acc_mat, tb = run_oracle(ds, 2, EPOCHS, LR, seed)
            else:
                dr, lnt, mt, mc = fpt_cfg
                acc_mat, tb, sp, mg, ac = run_fpt(
                    ds, 2, EPOCHS, LR, seed,
                    dormant_recall=dr,
                    loss_novelty_threshold=lnt,
                    merge_threshold=mt,
                    max_concepts=mc,
                )
                extra = f" spawn={sp} merge={mg} active={ac}"

            final = float(acc_mat[:, -1].mean())
            # Phase-specific accuracy
            phase1_acc = float(acc_mat[:, :10].mean())
            phase2_acc = float(acc_mat[:, 10:20].mean())
            phase3_acc = float(acc_mat[:, 20:].mean())
            # Recovery: accuracy at t=20 (first step of recurrence)
            recovery_t20 = float(acc_mat[:, 20].mean()) if T > 20 else 0.0

            all_rows.append({
                "method": name, "seed": seed, "final": final,
                "phase1": phase1_acc, "phase2": phase2_acc, "phase3": phase3_acc,
                "recovery_t20": recovery_t20, "bytes": tb,
            })
            all_curves.setdefault(name, []).append(acc_mat.mean(axis=0))

            print(f"  {name:18s} final={final:.4f} "
                  f"P1={phase1_acc:.3f} P2={phase2_acc:.3f} P3={phase3_acc:.3f} "
                  f"recov@20={recovery_t20:.3f} bytes={tb:.0f}{extra}")

    # ── Summary ──
    print("\n" + "=" * 65)
    print(f"{'Method':14s} {'Final':>8s} {'Phase1':>8s} {'Phase2':>8s} "
          f"{'Phase3':>8s} {'Recov@20':>9s} {'Bytes':>10s}")
    print("-" * 65)

    method_names = ["LocalOnly", "FedAvg f=2", "IFCA-3 f=2", "IFCA-8 f=2",
                    "FPT f=2", "FPT-tight f=2", "FPT-tight+DR f=2", "Oracle f=2"]
    for method in method_names:
        rows = [r for r in all_rows if r["method"] == method]
        if not rows:
            continue
        mf = np.mean([r["final"] for r in rows])
        p1 = np.mean([r["phase1"] for r in rows])
        p2 = np.mean([r["phase2"] for r in rows])
        p3 = np.mean([r["phase3"] for r in rows])
        rc = np.mean([r["recovery_t20"] for r in rows])
        mb = np.mean([r["bytes"] for r in rows])
        print(f"{method:14s} {mf:8.4f} {p1:8.4f} {p2:8.4f} "
              f"{p3:8.4f} {rc:9.4f} {mb:10.0f}")

    # ── Plot per-step accuracy curves ──
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"LocalOnly": "gray", "FedAvg f=2": "C2", "IFCA-3 f=2": "C1",
              "IFCA-8 f=2": "C3", "FPT f=2": "C0",
              "FPT-tight f=2": "C4", "FPT-tight+DR f=2": "C5",
              "Oracle f=2": "C6"}
    linestyles = {"Oracle f=2": "--"}

    for method in method_names:
        curves = all_curves.get(method, [])
        if not curves:
            continue
        mean_curve = np.mean(curves, axis=0)
        ax.plot(range(T), mean_curve, linewidth=2,
                color=colors.get(method, "black"),
                linestyle=linestyles.get(method, "-"),
                label=method, marker="o" if "Oracle" not in method else "")

    # Mark phase boundaries
    ax.axvline(10, color="red", alpha=0.5, linestyle=":", linewidth=2)
    ax.axvline(20, color="red", alpha=0.5, linestyle=":", linewidth=2)
    ax.text(4, ax.get_ylim()[1]*0.95, "Phase 1\nconcepts 0,1,2",
            ha="center", fontsize=9, color="red")
    ax.text(14, ax.get_ylim()[1]*0.95, "Phase 2\nconcepts 3,4,5",
            ha="center", fontsize=9, color="red")
    ax.text(24, ax.get_ylim()[1]*0.95, "Phase 3\n0,1,2 RECUR",
            ha="center", fontsize=9, color="red", fontweight="bold")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean Accuracy (across clients)")
    ax.set_title("Recurrence-with-Gap: 8 concepts, 3 active/phase, K=12")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "recurrence_gap_curves.png", dpi=150)
    plt.close(fig)

    with open(results_dir / "results.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    print(f"\nSaved to {results_dir}/")


if __name__ == "__main__":
    main()
