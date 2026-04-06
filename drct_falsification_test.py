from __future__ import annotations

"""DRCT falsification test.

Standalone script to empirically test the three core claims of DRCT
(Drift via Ratio of Covariance Traces):

    Claim 1:  r^G  !=  r_Sigma  during E2E CNN training (they differ non-trivially).
    Claim 2:  rho = r^G / r_Sigma  is temporally stable within a concept.
    Claim 3:  rho is discriminative across different concepts (class subsets).

Definitions
-----------
    r_Sigma = (tr Sigma)^2 / tr(Sigma^2),
        where Sigma is the penultimate-feature covariance on a held-out batch.
    r^G     = (tr G)^2 / tr(G^2),
        where G is the per-sample last-layer (flattened) gradient covariance;
        since full G is huge, we use its eigen-equivalent via sample-outer-product
        (Gram) formulation:  if J is the (N, p) per-sample flattened gradient
        matrix of the cross-entropy loss w.r.t. the last-layer weights W, then
        G = (1/N) J^T J  (p x p)  has the same non-zero eigenvalues as
        (1/N) J J^T  (N x N) -- we compute the latter and use
            tr(G)   = sum_i lambda_i
            tr(G^2) = sum_i lambda_i^2
        to get r^G exactly without building the p x p matrix.

Experiments (in order)
----------------------
    A: train small CNN for 10 epochs on a 10-class CIFAR-100 subset,
       record (r_Sigma, r^G, rho) after every epoch.
    B: at the final epoch, redo the computation on 3 independent 512-sample
       batches to estimate batch-level variance / CV of rho.
    C: if time permits, train independent models on 3 disjoint 10-class
       subsets and compare final rho.

Outputs
-------
    drct_falsification_timeseries.csv   (experiment A)
    drct_falsification_batch_var.csv    (experiment B)
    drct_falsification_concepts.csv     (experiment C)
    drct_rho_timeseries.png             (plot, if matplotlib available)
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Small CNN
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """Two conv blocks + two FC layers. Penultimate feature dim is configurable."""

    def __init__(self, num_classes: int = 10, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After two 2x2 pools on 32x32 -> 8x8 feature maps with 64 channels.
        self.fc1 = nn.Linear(64 * 8 * 8, feat_dim)
        self.fc_out = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = self.pool(h)
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = h.flatten(1)
        h = F.relu(self.fc1(h))
        return h  # (N, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.fc_out(z)


# ---------------------------------------------------------------------------
# Data (CIFAR-100, but only a K-class subset, remapped to 0..K-1)
# ---------------------------------------------------------------------------


class CIFARSubset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, 3, 32, 32) float32, labels: (N,) int64
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def load_cifar100_numpy(cache_dir: Path):
    """Load CIFAR-100 train/test from the extracted pickle cache.

    Returns
    -------
    (train_x, train_y, test_x, test_y, train_fine, test_fine) where fine are
    the original 0..99 labels and the x arrays are (N, 3, 32, 32) float32 in
    [0, 1].
    """
    import pickle

    train_path = cache_dir / "cifar-100-python" / "train"
    test_path = cache_dir / "cifar-100-python" / "test"

    with open(train_path, "rb") as fh:
        train = pickle.load(fh, encoding="latin1")
    with open(test_path, "rb") as fh:
        test = pickle.load(fh, encoding="latin1")

    def _reshape(raw):
        x = np.asarray(raw, dtype=np.float32) / 255.0  # (N, 3072)
        x = x.reshape(-1, 3, 32, 32)
        return x

    train_x = _reshape(train["data"])
    train_y = np.asarray(train["fine_labels"], dtype=np.int64)
    test_x = _reshape(test["data"])
    test_y = np.asarray(test["fine_labels"], dtype=np.int64)

    # Simple standardization using train statistics per channel.
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = train_x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    return train_x, train_y, test_x, test_y


def make_subset(
    images: np.ndarray,
    labels: np.ndarray,
    class_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    class_ids = sorted(class_ids)
    remap = {c: i for i, c in enumerate(class_ids)}
    mask = np.isin(labels, class_ids)
    sub_x = images[mask]
    sub_y = np.array([remap[int(c)] for c in labels[mask]], dtype=np.int64)
    return sub_x, sub_y


# ---------------------------------------------------------------------------
# Statistics: r_Sigma and r^G
# ---------------------------------------------------------------------------


def participation_ratio_from_eigs(eigs: np.ndarray) -> float:
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = eigs[eigs > 0.0]
    if eigs.size == 0:
        return float("nan")
    num = float(eigs.sum()) ** 2
    den = float((eigs ** 2).sum())
    if den <= 0.0:
        return float("nan")
    return num / den


@torch.no_grad()
def compute_r_sigma(
    model: SmallCNN,
    x: torch.Tensor,
    device: torch.device,
    center: bool = True,
) -> tuple[float, np.ndarray]:
    model.eval()
    x = x.to(device)
    z = model.features(x).detach().cpu().numpy().astype(np.float64)  # (N, d)
    if center:
        z = z - z.mean(axis=0, keepdims=True)
    N = z.shape[0]
    # Sigma = (1/N) z^T z; use Gram K = (1/N) z z^T which has same non-zero
    # spectrum as Sigma.
    K = (z @ z.T) / float(N)
    eigs = np.linalg.eigvalsh(K)
    return participation_ratio_from_eigs(eigs), eigs


def compute_r_g_last_layer(
    model: SmallCNN,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Compute r^G = (tr G)^2 / tr(G^2) for the last-layer weight gradient.

    The last layer is  logits = z W^T + b,  so
        d loss_i / d W  =  (p_i - e_{y_i}) outer z_i          (C x d)
    flattened to a (C*d,) vector g_i. The per-sample gradient covariance is
    G = (1/N) sum_i g_i g_i^T (C*d x C*d). Its non-zero eigenvalues coincide
    with those of the Gram matrix K = (1/N) [G_flat] [G_flat]^T (N x N):
        K_ij = (1/N) <g_i, g_j>
              = (1/N) <(p_i - e_{y_i}), (p_j - e_{y_j})> * <z_i, z_j>
    which we compute from the C-dim residuals and d-dim features without ever
    materialising g_i.
    """
    model.eval()
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        z = model.features(x)           # (N, d)
        logits = model.fc_out(z)        # (N, C)
        p = torch.softmax(logits, dim=1)  # (N, C)
        e_y = F.one_hot(y, num_classes=logits.shape[1]).float()
        r = (p - e_y)                     # (N, C) residual
    z_np = z.detach().cpu().numpy().astype(np.float64)
    r_np = r.detach().cpu().numpy().astype(np.float64)
    N = z_np.shape[0]
    # Gram over flattened last-layer grads: K = (1/N) (R R^T) elementwise * (Z Z^T)
    RRT = r_np @ r_np.T
    ZZT = z_np @ z_np.T
    K = (RRT * ZZT) / float(N)
    eigs = np.linalg.eigvalsh(K)
    return participation_ratio_from_eigs(eigs), eigs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: SmallCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach().cpu()) * xb.size(0)
        correct += int((logits.argmax(dim=1) == yb).sum().cpu())
        total += xb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def test_accuracy(model: SmallCNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        correct += int((logits.argmax(dim=1) == yb).sum().cpu())
        total += xb.size(0)
    return correct / max(total, 1)


def sample_batch(
    images: np.ndarray,
    labels: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = rng.choice(images.shape[0], size=n, replace=False)
    x = torch.from_numpy(images[idx]).float()
    y = torch.from_numpy(labels[idx]).long()
    return x, y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment_a(
    train_x, train_y, test_x, test_y,
    device, epochs=10, batch_size=128, stat_batch_size=512, seed=42,
    tag="concept0",
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train_ds = CIFARSubset(train_x, train_y)
    test_ds = CIFARSubset(test_x, test_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    num_classes = int(train_y.max()) + 1
    model = SmallCNN(num_classes=num_classes, feat_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,
                                weight_decay=5e-4)

    # Epoch 0 (untrained) baseline statistic
    x_stat, y_stat = sample_batch(test_x, test_y, stat_batch_size, rng)
    r_sigma, _ = compute_r_sigma(model, x_stat, device)
    r_g, _ = compute_r_g_last_layer(model, x_stat, y_stat, device)
    rho = r_g / r_sigma if r_sigma > 0 else float("nan")
    rows = [{
        "tag": tag, "epoch": 0, "train_loss": float("nan"), "train_acc": float("nan"),
        "test_acc": test_accuracy(model, test_loader, device),
        "r_sigma": r_sigma, "r_g": r_g, "rho": rho,
    }]
    print(f"[{tag}] epoch 0  r_sigma={r_sigma:.3f}  r_g={r_g:.3f}  rho={rho:.3f}  "
          f"test_acc={rows[-1]['test_acc']:.3f}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        x_stat, y_stat = sample_batch(test_x, test_y, stat_batch_size, rng)
        r_sigma, _ = compute_r_sigma(model, x_stat, device)
        r_g, _ = compute_r_g_last_layer(model, x_stat, y_stat, device)
        rho = r_g / r_sigma if r_sigma > 0 else float("nan")
        test_acc = test_accuracy(model, test_loader, device)
        rows.append({
            "tag": tag, "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "test_acc": test_acc,
            "r_sigma": r_sigma, "r_g": r_g, "rho": rho,
        })
        print(f"[{tag}] epoch {epoch}  train_loss={train_loss:.3f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
              f"r_sigma={r_sigma:.2f}  r_g={r_g:.2f}  rho={rho:.3f}")

    return model, rows


def run_experiment_b(model, test_x, test_y, device, n_batches=5, stat_batch_size=512, seed=123):
    """Batch-level variance of rho at fixed model."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_batches):
        x_stat, y_stat = sample_batch(test_x, test_y, stat_batch_size, rng)
        r_sigma, _ = compute_r_sigma(model, x_stat, device)
        r_g, _ = compute_r_g_last_layer(model, x_stat, y_stat, device)
        rho = r_g / r_sigma if r_sigma > 0 else float("nan")
        rows.append({"batch": b, "r_sigma": r_sigma, "r_g": r_g, "rho": rho})
        print(f"[batch-var] batch {b}  r_sigma={r_sigma:.2f}  r_g={r_g:.2f}  rho={rho:.3f}")
    return rows


def save_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {path}")


def maybe_plot(rows_by_tag, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] matplotlib unavailable: {exc}")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for tag, rows in rows_by_tag.items():
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["r_sigma"] for r in rows], label=f"{tag}: r_sigma")
        axes[0].plot(epochs, [r["r_g"] for r in rows], linestyle="--", label=f"{tag}: r_g")
        axes[1].plot(epochs, [r["rho"] for r in rows], label=tag)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("participation ratio")
    axes[0].set_title("r_Sigma (solid) vs r^G (dashed)")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("rho = r^G / r_Sigma")
    axes[1].axhline(1.0, color="k", linestyle=":", alpha=0.5)
    axes[1].set_title("DRCT ratio over training"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stat-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=".cifar100_cache")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--skip-concept-c", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    cache_dir = Path(args.cache_dir)
    train_x, train_y, test_x, test_y = load_cifar100_numpy(cache_dir)
    print(f"[data] train={train_x.shape}  test={test_x.shape}")

    # Define three 10-class "concepts" from CIFAR-100 for experiment C.
    concept_classes = {
        "concept0": list(range(0, 10)),
        "concept1": list(range(40, 50)),
        "concept2": list(range(80, 90)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_tag: dict[str, list[dict]] = {}
    # Experiment A on concept0
    t0 = time.time()
    tr_x, tr_y = make_subset(train_x, train_y, concept_classes["concept0"])
    te_x, te_y = make_subset(test_x, test_y, concept_classes["concept0"])
    print(f"[concept0] train={tr_x.shape}  test={te_x.shape}")
    model_a, rows_a = run_experiment_a(
        tr_x, tr_y, te_x, te_y, device,
        epochs=args.epochs, batch_size=args.batch_size,
        stat_batch_size=args.stat_batch_size, seed=args.seed, tag="concept0",
    )
    rows_by_tag["concept0"] = rows_a
    print(f"[concept0] elapsed {time.time()-t0:.1f}s")

    # Experiment B: batch-level variance of rho at the final epoch of model_a.
    rows_b = run_experiment_b(
        model_a, te_x, te_y, device,
        n_batches=5, stat_batch_size=args.stat_batch_size, seed=args.seed + 777,
    )

    # Experiment C: two more concepts, shorter epoch budget.
    rows_c_summary: list[dict] = []
    if not args.skip_concept_c:
        for tag in ("concept1", "concept2"):
            t0 = time.time()
            tr_x, tr_y = make_subset(train_x, train_y, concept_classes[tag])
            te_x, te_y = make_subset(test_x, test_y, concept_classes[tag])
            print(f"[{tag}] train={tr_x.shape}  test={te_x.shape}")
            _, rows = run_experiment_a(
                tr_x, tr_y, te_x, te_y, device,
                epochs=max(5, args.epochs // 2), batch_size=args.batch_size,
                stat_batch_size=args.stat_batch_size, seed=args.seed + 100, tag=tag,
            )
            rows_by_tag[tag] = rows
            print(f"[{tag}] elapsed {time.time()-t0:.1f}s")

        for tag, rows in rows_by_tag.items():
            late = rows[-3:]
            rhos = np.array([r["rho"] for r in late], dtype=np.float64)
            rows_c_summary.append({
                "tag": tag, "mean_rho_last3": float(rhos.mean()),
                "std_rho_last3": float(rhos.std(ddof=0)),
                "min_rho_last3": float(rhos.min()),
                "max_rho_last3": float(rhos.max()),
            })

    # Flatten timeseries rows from all tags for CSV export.
    ts_rows: list[dict] = []
    for tag, rows in rows_by_tag.items():
        ts_rows.extend(rows)
    save_csv(ts_rows, out_dir / "drct_falsification_timeseries.csv")
    save_csv(rows_b, out_dir / "drct_falsification_batch_var.csv")
    save_csv(rows_c_summary, out_dir / "drct_falsification_concepts.csv")
    maybe_plot(rows_by_tag, out_dir / "drct_rho_timeseries.png")

    # Final summary.
    print("\n=== SUMMARY ===")
    for tag, rows in rows_by_tag.items():
        rhos = np.array([r["rho"] for r in rows[1:]], dtype=np.float64)
        r_sigmas = np.array([r["r_sigma"] for r in rows[1:]], dtype=np.float64)
        r_gs = np.array([r["r_g"] for r in rows[1:]], dtype=np.float64)
        print(f"[{tag}] rho: mean={rhos.mean():.3f} std={rhos.std(ddof=0):.3f} "
              f"CV={rhos.std(ddof=0)/max(abs(rhos.mean()),1e-12):.3f}  "
              f"min={rhos.min():.3f} max={rhos.max():.3f}")
        print(f"        r_sigma range [{r_sigmas.min():.2f}, {r_sigmas.max():.2f}]  "
              f"r_g range [{r_gs.min():.2f}, {r_gs.max():.2f}]")

    rhos_b = np.array([r["rho"] for r in rows_b], dtype=np.float64)
    print(f"[batch-var] rho mean={rhos_b.mean():.3f} std={rhos_b.std(ddof=0):.3f} "
          f"CV={rhos_b.std(ddof=0)/max(abs(rhos_b.mean()),1e-12):.3f}")


if __name__ == "__main__":
    main()
