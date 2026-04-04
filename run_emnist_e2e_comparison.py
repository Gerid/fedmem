"""End-to-end CNN comparison on EMNIST with concept drift.

Addresses reviewer criticism: "frozen features + linear heads ≈ linear model."
This experiment trains a CNN end-to-end (features are learned, not frozen),
validating that the implicit-shrinkage finding holds in genuinely nonlinear settings.

Methods: FedAvg, Oracle, CFL, IFCA, Shrinkage, FedBN, FedPer, FLUX-style.

Usage:
    python run_emnist_e2e_comparison.py --seeds 42
    python run_emnist_e2e_comparison.py --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

class LeNetBN(nn.Module):
    """LeNet-style CNN with optional BatchNorm for EMNIST (28x28, 1 channel)."""

    def __init__(self, n_classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 28 -> 14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 14 -> 7
        x = x.reshape(x.size(0), -1)                      # 64*7*7 = 3136
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def body_params(self) -> dict[str, torch.Tensor]:
        """Conv layers + BN (shared body for FedPer)."""
        return {k: v.clone() for k, v in self.state_dict().items()
                if k.startswith(("conv", "bn"))}

    def head_params(self) -> dict[str, torch.Tensor]:
        """FC layers (personal head for FedPer)."""
        return {k: v.clone() for k, v in self.state_dict().items()
                if k.startswith("fc")}

    def non_bn_params(self) -> dict[str, torch.Tensor]:
        """All params except BatchNorm (for FedBN)."""
        return {k: v.clone() for k, v in self.state_dict().items()
                if "bn" not in k}

    def bn_params(self) -> dict[str, torch.Tensor]:
        """Only BatchNorm params (local in FedBN)."""
        return {k: v.clone() for k, v in self.state_dict().items()
                if "bn" in k}


# ---------------------------------------------------------------------------
# Data: EMNIST with concept-drift recurrence
# ---------------------------------------------------------------------------

def _apply_transform(images: np.ndarray, concept: int) -> np.ndarray:
    """Apply concept-specific visual transform to 28x28 images."""
    if concept == 0:
        return images  # original
    elif concept == 1:
        # slight Gaussian blur (3x3 average)
        from scipy.ndimage import uniform_filter
        return uniform_filter(images, size=(1, 3, 3)).astype(np.float32)
    elif concept == 2:
        # invert
        return (1.0 - images).astype(np.float32)
    elif concept == 3:
        # add Gaussian noise
        rng = np.random.RandomState(42)
        noise = rng.randn(*images.shape).astype(np.float32) * 0.15
        return np.clip(images + noise, 0, 1).astype(np.float32)
    else:
        # rotate 90 degrees
        return np.rot90(images, k=1, axes=(1, 2)).copy().astype(np.float32)


def generate_emnist_recurrence(
    K: int = 20,
    T: int = 20,
    C: int = 4,
    n_samples: int = 200,
    seed: int = 42,
    data_root: str = ".emnist_cache",
) -> tuple[dict, np.ndarray, dict]:
    """Generate EMNIST recurrence dataset with concept drift.

    Returns
    -------
    data : dict[(k, t)] -> (X, y) where X is (n, 1, 28, 28) float32, y is (n,) int
    concept_matrix : (K, T) int array
    info : dict with metadata
    """
    from torchvision import datasets

    rng = np.random.RandomState(seed)

    # Download EMNIST digits
    ds = datasets.EMNIST(root=data_root, split="digits", train=True, download=True)
    all_images = ds.data.numpy().astype(np.float32) / 255.0  # (N, 28, 28)
    all_labels = ds.targets.numpy()
    n_classes = 10

    # Build per-class pools
    class_pools = {}
    for c in range(n_classes):
        idx = np.where(all_labels == c)[0]
        class_pools[c] = all_images[idx]

    # Generate concept matrix (same recurrence pattern as CIFAR-100)
    concept_matrix = np.zeros((K, T), dtype=int)
    for k in range(K):
        base_concept = k % C
        switch_prob = 0.15
        current = base_concept
        for t in range(T):
            if t > 0 and rng.random() < switch_prob:
                current = rng.randint(0, C)
            concept_matrix[k, t] = current

    # Generate data for each (k, t)
    data = {}
    for k in range(K):
        for t in range(T):
            concept = concept_matrix[k, t]
            # Sample n_samples images uniformly across classes
            images = []
            labels = []
            per_class = n_samples // n_classes
            remainder = n_samples - per_class * n_classes
            for c in range(n_classes):
                n_c = per_class + (1 if c < remainder else 0)
                idx = rng.choice(len(class_pools[c]), size=n_c, replace=True)
                images.append(class_pools[c][idx])
                labels.extend([c] * n_c)

            images = np.concatenate(images, axis=0)  # (n_samples, 28, 28)
            labels = np.array(labels, dtype=np.int64)

            # Apply concept-specific transform
            images = _apply_transform(images, concept)

            # Add channel dim: (n, 28, 28) -> (n, 1, 28, 28)
            images = images[:, np.newaxis, :, :]

            # Shuffle
            perm = rng.permutation(len(labels))
            data[(k, t)] = (images[perm], labels[perm])

    info = {
        "K": K, "T": T, "C": C, "n_classes": n_classes,
        "n_samples": n_samples, "seed": seed,
    }
    return data, concept_matrix, info


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    n_epochs: int = 3,
    lr: float = 0.01,
    batch_size: int = 64,
) -> None:
    """Train CNN model on local data."""
    model.train()
    model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    ds = TensorDataset(
        torch.from_numpy(X).float().to(DEVICE),
        torch.from_numpy(y).long().to(DEVICE),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(n_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate accuracy."""
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(DEVICE)
        y_t = torch.from_numpy(y).long().to(DEVICE)
        logits = model(X_t)
        acc = (logits.argmax(1) == y_t).float().mean().item()
    return acc


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted average of state dicts."""
    if not state_dicts:
        return {}
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]
    avg = {}
    for key in state_dicts[0]:
        avg[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
    return avg


def gradient_similarity_matrix(
    models_before: list[dict],
    models_after: list[dict],
) -> np.ndarray:
    """Compute pairwise cosine similarity of pseudo-gradients (for CFL)."""
    K = len(models_before)
    grads = []
    for i in range(K):
        delta = []
        for key in models_before[i]:
            d = (models_after[i][key].float() - models_before[i][key].float()).flatten()
            delta.append(d)
        grads.append(torch.cat(delta))

    sim = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cos = F.cosine_similarity(grads[i].unsqueeze(0), grads[j].unsqueeze(0)).item()
            sim[i, j] = cos
    return sim


def cluster_by_similarity(sim: np.ndarray, n_clusters: int) -> np.ndarray:
    """Hierarchical clustering from cosine similarity matrix."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 2)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels


# ---------------------------------------------------------------------------
# FL Methods
# ---------------------------------------------------------------------------

def run_fedavg(
    data: dict, concept_matrix: np.ndarray, info: dict,
    n_epochs: int = 3, lr: float = 0.01, federation_every: int = 2,
) -> np.ndarray:
    """FedAvg: average all client models globally."""
    K, T, C, n_classes = info["K"], info["T"], info["C"], info["n_classes"]
    global_model = LeNetBN(n_classes=n_classes).to(DEVICE)
    acc_matrix = np.zeros((K, T))

    for t in range(T):
        local_sds = []
        for k in range(K):
            model = LeNetBN(n_classes=n_classes).to(DEVICE)
            model.load_state_dict(copy.deepcopy(global_model.state_dict()))
            X, y = data[(k, t)]
            # Split train/test
            n = len(y)
            mid = n // 2
            train_model(model, X[:mid], y[:mid], n_epochs=n_epochs, lr=lr)
            acc_matrix[k, t] = evaluate_model(model, X[mid:], y[mid:])
            local_sds.append({k: v.cpu() for k, v in model.state_dict().items()})

        if (t + 1) % federation_every == 0:
            avg_sd = average_state_dicts(local_sds)
            global_model.load_state_dict(avg_sd)

    return acc_matrix


def run_oracle(
    data: dict, concept_matrix: np.ndarray, info: dict,
    n_epochs: int = 3, lr: float = 0.01, federation_every: int = 2,
) -> np.ndarray:
    """Oracle: average only within true concept groups."""
    K, T, C, n_classes = info["K"], info["T"], info["C"], info["n_classes"]
    concept_models = {
        j: LeNetBN(n_classes=n_classes).to(DEVICE) for j in range(C)
    }
    acc_matrix = np.zeros((K, T))

    for t in range(T):
        concept_sds = defaultdict(list)
        for k in range(K):
            j = concept_matrix[k, t]
            model = LeNetBN(n_classes=n_classes).to(DEVICE)
            model.load_state_dict(copy.deepcopy(concept_models[j].state_dict()))
            X, y = data[(k, t)]
            n = len(y)
            mid = n // 2
            train_model(model, X[:mid], y[:mid], n_epochs=n_epochs, lr=lr)
            acc_matrix[k, t] = evaluate_model(model, X[mid:], y[mid:])
            concept_sds[j].append({k: v.cpu() for k, v in model.state_dict().items()})

        if (t + 1) % federation_every == 0:
            for j in range(C):
                if concept_sds[j]:
                    avg_sd = average_state_dicts(concept_sds[j])
                    concept_models[j].load_state_dict(avg_sd)

    return acc_matrix


def run_cfl(
    data: dict, concept_matrix: np.ndarray, info: dict,
    n_epochs: int = 3, lr: float = 0.01, federation_every: int = 2,
) -> np.ndarray:
    """CFL: cluster by gradient similarity, average within clusters."""
    K, T, C, n_classes = info["K"], info["T"], info["C"], info["n_classes"]
    global_model = LeNetBN(n_classes=n_classes).to(DEVICE)
    cluster_models = {}
    client_clusters = np.zeros(K, dtype=int)
    acc_matrix = np.zeros((K, T))

    for t in range(T):
        before_sds = []
        after_sds = []
        local_models = []

        for k in range(K):
            # Use cluster model if available, else global
            if client_clusters[k] in cluster_models:
                init_sd = copy.deepcopy(cluster_models[client_clusters[k]])
            else:
                init_sd = copy.deepcopy(global_model.state_dict())

            before_sds.append({k: v.cpu() for k, v in init_sd.items()})

            model = LeNetBN(n_classes=n_classes).to(DEVICE)
            model.load_state_dict(copy.deepcopy(init_sd))
            X, y = data[(k, t)]
            n = len(y)
            mid = n // 2
            train_model(model, X[:mid], y[:mid], n_epochs=n_epochs, lr=lr)
            acc_matrix[k, t] = evaluate_model(model, X[mid:], y[mid:])
            after_sds.append({k: v.cpu() for k, v in model.state_dict().items()})
            local_models.append(model)

        if (t + 1) % federation_every == 0:
            # Cluster by gradient similarity
            sim = gradient_similarity_matrix(before_sds, after_sds)
            client_clusters = cluster_by_similarity(sim, C)

            # Average within clusters
            cluster_models = {}
            for j in range(C):
                members = [i for i in range(K) if client_clusters[i] == j]
                if members:
                    sds = [after_sds[i] for i in members]
                    cluster_models[j] = average_state_dicts(sds)

            # Also update global
            global_sd = average_state_dicts(after_sds)
            global_model.load_state_dict(global_sd)

    return acc_matrix


def run_ifca(
    data: dict, concept_matrix: np.ndarray, info: dict,
    n_epochs: int = 3, lr: float = 0.01, federation_every: int = 2,
) -> np.ndarray:
    """IFCA: each client picks best of C cluster heads."""
    K, T, C, n_classes = info["K"], info["T"], info["C"], info["n_classes"]
    cluster_models = [
        LeNetBN(n_classes=n_classes).to(DEVICE) for _ in range(C)
    ]
    acc_matrix = np.zeros((K, T))

    for t in range(T):
        cluster_sds = defaultdict(list)
        for k in range(K):
            X, y = data[(k, t)]
            n = len(y)
            mid = n // 2

            # Pick best cluster head by loss on test half
            best_j, best_loss = 0, float("inf")
            for j in range(C):
                cluster_models[j].eval()
                with torch.no_grad():
                    X_t = torch.from_numpy(X[mid:]).float().to(DEVICE)
                    y_t = torch.from_numpy(y[mid:]).long().to(DEVICE)
                    loss = F.cross_entropy(cluster_models[j](X_t), y_t).item()
                if loss < best_loss:
                    best_loss = loss
                    best_j = j

            model = LeNetBN(n_classes=n_classes).to(DEVICE)
            model.load_state_dict(copy.deepcopy(cluster_models[best_j].state_dict()))
            train_model(model, X[:mid], y[:mid], n_epochs=n_epochs, lr=lr)
            acc_matrix[k, t] = evaluate_model(model, X[mid:], y[mid:])
            cluster_sds[best_j].append({k: v.cpu() for k, v in model.state_dict().items()})

        if (t + 1) % federation_every == 0:
            for j in range(C):
                if cluster_sds[j]:
                    avg_sd = average_state_dicts(cluster_sds[j])
                    cluster_models[j].load_state_dict(avg_sd)

    return acc_matrix


def run_shrinkage(
    data: dict, concept_matrix: np.ndarray, info: dict,
    n_epochs: int = 3, lr: float = 0.01, federation_every: int = 2,
) -> np.ndarray:
    """Oracle shrinkage: interpolate between global and concept-level with
    data-driven lambda (empirical Bayes on model parameter variance)."""
    K, T, C, n_classes = info["K"], info["T"], info["C"], info["n_classes"]
    global_model = LeNetBN(n_classes=n_classes).to(DEVICE)
    concept_models = {
        j: LeNetBN(n_classes=n_classes).to(DEVICE) for j in range(C)
    }
    acc_matrix = np.zeros((K, T))

    for t in range(T):
        concept_sds = defaultdict(list)
        all_sds = []
        for k in range(K):
            j = concept_matrix[k, t]
            model = LeNetBN(n_classes=n_classes).to(DEVICE)
            model.load_state_dict(copy.deepcopy(concept_models[j].state_dict()))
            X, y = data[(k, t)]
            n = len(y)
            mid = n // 2
            train_model(model, X[:mid], y[:mid], n_epochs=n_epochs, lr=lr)
            acc_matrix[k, t] = evaluate_model(model, X[mid:], y[mid:])
            sd = {k: v.cpu() for k, v in model.state_dict().items()}
            concept_sds[j].append(sd)
            all_sds.append(sd)

        if (t + 1) % federation_every == 0:
            global_sd = average_state_dicts(all_sds)

            for j in range(C):
                if concept_sds[j]:
                    concept_sd = average_state_dicts(concept_sds[j])
                    # Compute shrinkage lambda from parameter variance
                    # lambda = sigma2_noise / (sigma2_noise + sigma2_between)
                    between_var = 0.0
                    total_dim = 0
                    for key in concept_sd:
                        diff = (concept_sd[key].float() - global_sd[key].float())
                        between_var += (diff ** 2).sum().item()
                        total_dim += diff.numel()
                    sigma_b2 = max(between_var / total_dim, 1e-8)
                    # Estimate noise variance from within-concept spread
                    within_var = 0.0
                    for sd in concept_sds[j]:
                        for key in sd:
                            diff = (sd[key].float() - concept_sd[key].float())
                            within_var += (diff ** 2).sum().item()
                    sigma_n2 = within_var / (total_dim * max(len(concept_sds[j]) - 1, 1))
                    lam = sigma_n2 / (sigma_n2 + sigma_b2)
                    lam = np.clip(lam, 0, 1)

                    # Shrink concept model toward global
                    shrunk_sd = {}
                    for key in concept_sd:
                        shrunk_sd[key] = (1 - lam) * concept_sd[key].float() + lam * global_sd[key].float()
                    concept_models[j].load_state_dict(shrunk_sd)

    return acc_matrix


def run_fedbn(
    data: dict, concept_matrix: np.ndarray, info: dict,
    n_epochs: int = 3, lr: float = 0.01, federation_every: int = 2,
) -> np.ndarray:
    """FedBN: average all params EXCEPT BatchNorm (BN stays local)."""
    K, T, C, n_classes = info["K"], info["T"], info["C"], info["n_classes"]
    global_model = LeNetBN(n_classes=n_classes).to(DEVICE)
    # Each client keeps its own BN params
    client_bn = [global_model.bn_params() for _ in range(K)]
    acc_matrix = np.zeros((K, T))

    for t in range(T):
        local_non_bn = []
        for k in range(K):
            model = LeNetBN(n_classes=n_classes).to(DEVICE)
            # Load global non-BN + client-specific BN
            sd = copy.deepcopy(global_model.state_dict())
            sd.update(client_bn[k])
            model.load_state_dict(sd)

            X, y = data[(k, t)]
            n = len(y)
            mid = n // 2
            train_model(model, X[:mid], y[:mid], n_epochs=n_epochs, lr=lr)
            acc_matrix[k, t] = evaluate_model(model, X[mid:], y[mid:])

            # Save client BN and non-BN separately
            client_bn[k] = {k_: v.cpu() for k_, v in model.state_dict().items()
                            if "bn" in k_}
            local_non_bn.append({k_: v.cpu() for k_, v in model.state_dict().items()
                                 if "bn" not in k_})

        if (t + 1) % federation_every == 0:
            avg_non_bn = average_state_dicts(local_non_bn)
            full_sd = global_model.state_dict()
            full_sd.update(avg_non_bn)
            global_model.load_state_dict(full_sd)

    return acc_matrix


def run_fedper(
    data: dict, concept_matrix: np.ndarray, info: dict,
    n_epochs: int = 3, lr: float = 0.01, federation_every: int = 2,
) -> np.ndarray:
    """FedPer: average only conv body, keep FC head local."""
    K, T, C, n_classes = info["K"], info["T"], info["C"], info["n_classes"]
    global_model = LeNetBN(n_classes=n_classes).to(DEVICE)
    # Each client keeps its own head
    client_heads = [global_model.head_params() for _ in range(K)]
    acc_matrix = np.zeros((K, T))

    for t in range(T):
        local_bodies = []
        for k in range(K):
            model = LeNetBN(n_classes=n_classes).to(DEVICE)
            # Load global body + client-specific head
            sd = copy.deepcopy(global_model.state_dict())
            sd.update(client_heads[k])
            model.load_state_dict(sd)

            X, y = data[(k, t)]
            n = len(y)
            mid = n // 2
            train_model(model, X[:mid], y[:mid], n_epochs=n_epochs, lr=lr)
            acc_matrix[k, t] = evaluate_model(model, X[mid:], y[mid:])

            # Save head and body separately
            client_heads[k] = {k_: v.cpu() for k_, v in model.state_dict().items()
                               if k_.startswith("fc")}
            local_bodies.append({k_: v.cpu() for k_, v in model.state_dict().items()
                                 if not k_.startswith("fc")})

        if (t + 1) % federation_every == 0:
            avg_body = average_state_dicts(local_bodies)
            full_sd = global_model.state_dict()
            full_sd.update(avg_body)
            global_model.load_state_dict(full_sd)

    return acc_matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_METHODS = {
    "FedAvg": run_fedavg,
    "Oracle": run_oracle,
    "CFL": run_cfl,
    "IFCA": run_ifca,
    "Shrinkage": run_shrinkage,
    "FedBN": run_fedbn,
    "FedPer": run_fedper,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="EMNIST end-to-end CNN comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--results-dir", default="tmp/emnist_e2e")
    parser.add_argument("--data-root", default=".emnist_cache")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--n-workers", type=int, default=0)  # RunPod compat
    parser.add_argument("--feature-cache-dir", default=None)  # unused, RunPod compat
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"K={args.K}, T={args.T}, C={args.C}, n_samples={args.n_samples}")
    print(f"n_epochs={args.n_epochs}, lr={args.lr}, federation_every={args.federation_every}")

    all_rows = []

    for seed in args.seeds:
        print(f"\n=== seed={seed} ===")
        t0 = time.time()
        data, concept_matrix, info = generate_emnist_recurrence(
            K=args.K, T=args.T, C=args.C, n_samples=args.n_samples,
            seed=seed, data_root=args.data_root,
        )
        gen_time = time.time() - t0
        print(f"  Data generation: {gen_time:.1f}s")

        for method_name, method_fn in _METHODS.items():
            try:
                t1 = time.time()
                acc_matrix = method_fn(
                    data, concept_matrix, info,
                    n_epochs=args.n_epochs, lr=args.lr,
                    federation_every=args.federation_every,
                )
                elapsed = time.time() - t1
                final_acc = float(acc_matrix[:, -1].mean())

                row = {
                    "seed": seed,
                    "method": method_name,
                    "final_accuracy": round(final_acc, 4),
                    "acc_auc": round(float(acc_matrix.mean()), 4),
                    "time_s": round(elapsed, 1),
                    "status": "ok",
                }
                all_rows.append(row)
                print(f"  {method_name}: acc={final_acc:.4f} ({elapsed:.1f}s)")

            except Exception as e:
                all_rows.append({
                    "seed": seed,
                    "method": method_name,
                    "final_accuracy": None,
                    "acc_auc": None,
                    "time_s": None,
                    "status": f"FAILED: {e}",
                })
                print(f"  {method_name}: FAILED - {e}")
                traceback.print_exc()

    # Save results
    out_path = results_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({"rows": all_rows, "args": vars(args)}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n=== Summary (seed-averaged) ===")
    from collections import defaultdict as dd
    method_accs = dd(list)
    for row in all_rows:
        if row["status"] == "ok":
            method_accs[row["method"]].append(row["final_accuracy"])

    print(f"{'Method':<15} {'Acc':>8} {'Std':>8}")
    for method in _METHODS:
        accs = method_accs.get(method, [])
        if accs:
            mean = np.mean(accs)
            std = np.std(accs) if len(accs) > 1 else 0
            print(f"{method:<15} {mean:8.4f} {std:8.4f}")


if __name__ == "__main__":
    main()
