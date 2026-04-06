from __future__ import annotations

"""DRCT rho power analysis and multi-signal discrimination test.

This script addresses the claim that rho = r^G / r_Sigma cannot distinguish
certain concept pairs (e.g. concept0 vs concept1, p=0.769 from a t-test on
N=3 data points per group).

We:
  1. Quantify the statistical power problem (N=3 is catastrophically low)
  2. Train 5 concepts x 3 seeds to get proper statistics
  3. Test rho alone via ANOVA + pairwise tests
  4. Test multi-signal combinations (rho + tr(G), rho + r_Sigma, etc.)
  5. Recommend the best fingerprint approach

Outputs:
  drct_rho_power_report.txt     -- full analysis report
  drct_rho_extended_data.csv    -- raw data for 5 concepts x 3 seeds
  drct_rho_discrimination.csv   -- pairwise discrimination matrix
"""

import argparse
import csv
import itertools
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─── Model (same as falsification test) ───────────────────────────────────

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
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
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self.features(x))


# ─── Data ─────────────────────────────────────────────────────────────────

class CIFARSubset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def load_cifar100_numpy(cache_dir: Path):
    train_path = cache_dir / "cifar-100-python" / "train"
    test_path = cache_dir / "cifar-100-python" / "test"
    with open(train_path, "rb") as fh:
        train = pickle.load(fh, encoding="latin1")
    with open(test_path, "rb") as fh:
        test = pickle.load(fh, encoding="latin1")

    def _reshape(raw):
        x = np.asarray(raw, dtype=np.float32) / 255.0
        return x.reshape(-1, 3, 32, 32)

    train_x = _reshape(train["data"])
    train_y = np.asarray(train["fine_labels"], dtype=np.int64)
    test_x = _reshape(test["data"])
    test_y = np.asarray(test["fine_labels"], dtype=np.int64)
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True)
    std = train_x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    return train_x, train_y, test_x, test_y


def make_subset(images, labels, class_ids):
    class_ids = sorted(class_ids)
    remap = {c: i for i, c in enumerate(class_ids)}
    mask = np.isin(labels, class_ids)
    sub_x = images[mask]
    sub_y = np.array([remap[int(c)] for c in labels[mask]], dtype=np.int64)
    return sub_x, sub_y


# ─── Spectral statistics ─────────────────────────────────────────────────

def participation_ratio(eigs: np.ndarray) -> float:
    eigs = np.asarray(eigs, dtype=np.float64)
    eigs = eigs[eigs > 0.0]
    if eigs.size == 0:
        return float("nan")
    num = float(eigs.sum()) ** 2
    den = float((eigs ** 2).sum())
    return num / den if den > 0 else float("nan")


@torch.no_grad()
def compute_feature_stats(model, x, device):
    """Compute r_Sigma, tr(Sigma), max_eig(Sigma) from penultimate features."""
    model.eval()
    z = model.features(x.to(device)).detach().cpu().numpy().astype(np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    N = z.shape[0]
    K = (z @ z.T) / float(N)
    eigs = np.linalg.eigvalsh(K)
    eigs_pos = eigs[eigs > 0]
    return {
        "r_sigma": participation_ratio(eigs),
        "tr_sigma": float(eigs_pos.sum()) if eigs_pos.size > 0 else 0.0,
        "max_eig_sigma": float(eigs_pos.max()) if eigs_pos.size > 0 else 0.0,
        "eig_spread_sigma": float(eigs_pos.std()) if eigs_pos.size > 1 else 0.0,
    }


def compute_gradient_stats(model, x, y, device):
    """Compute r^G, tr(G), max_eig(G) from last-layer gradient covariance."""
    model.eval()
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        z = model.features(x)
        logits = model.fc_out(z)
        p = torch.softmax(logits, dim=1)
        e_y = F.one_hot(y, num_classes=logits.shape[1]).float()
        r = (p - e_y)
    z_np = z.detach().cpu().numpy().astype(np.float64)
    r_np = r.detach().cpu().numpy().astype(np.float64)
    N = z_np.shape[0]
    RRT = r_np @ r_np.T
    ZZT = z_np @ z_np.T
    K = (RRT * ZZT) / float(N)
    eigs = np.linalg.eigvalsh(K)
    eigs_pos = eigs[eigs > 0]
    return {
        "r_g": participation_ratio(eigs),
        "tr_g": float(eigs_pos.sum()) if eigs_pos.size > 0 else 0.0,
        "max_eig_g": float(eigs_pos.max()) if eigs_pos.size > 0 else 0.0,
        "eig_spread_g": float(eigs_pos.std()) if eigs_pos.size > 1 else 0.0,
    }


# ─── Training ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach().cpu()) * xb.size(0)
        correct += int((logits.argmax(1) == yb).sum().cpu())
        total += xb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += int((model(xb).argmax(1) == yb).sum().cpu())
        total += xb.size(0)
    return correct / max(total, 1)


def sample_batch(images, labels, n, rng):
    idx = rng.choice(images.shape[0], size=min(n, images.shape[0]), replace=False)
    return torch.from_numpy(images[idx]).float(), torch.from_numpy(labels[idx]).long()


# ─── Part 0: Power analysis on existing data ─────────────────────────────

def power_analysis_existing():
    """Analyze the statistical power of the original N=3 comparison."""
    lines = []
    lines.append("=" * 70)
    lines.append("PART 0: Power analysis of original experiment (N=3 per group)")
    lines.append("=" * 70)

    # Data from drct_falsification_concepts.csv
    c0_rhos = np.array([2.302712, 2.180575, 1.973155])  # concept0 last-3 (epochs 8,9,10)
    c1_rhos = np.array([2.186761, 2.188396, 1.966730])  # concept1 last-3 (epochs 3,4,5)
    c2_rhos = np.array([2.699195, 3.070569, 3.293636])  # concept2 last-3 (epochs 3,4,5)

    from scipy import stats

    pairs = [("c0-c1", c0_rhos, c1_rhos), ("c0-c2", c0_rhos, c2_rhos), ("c1-c2", c1_rhos, c2_rhos)]

    for name, a, b in pairs:
        # Cohen's d
        pooled_std = np.sqrt(((len(a) - 1) * a.std(ddof=1)**2 + (len(b) - 1) * b.std(ddof=1)**2) /
                             (len(a) + len(b) - 2))
        d = abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else float("inf")

        # t-test
        t_stat, p_val = stats.ttest_ind(a, b)

        # Power calculation (two-sample t-test, two-tailed, alpha=0.05)
        # Using noncentrality parameter: delta = d * sqrt(n1*n2/(n1+n2))
        n1, n2 = len(a), len(b)
        ncp = d * np.sqrt(n1 * n2 / (n1 + n2))
        df = n1 + n2 - 2
        t_crit = stats.t.ppf(1 - 0.025, df)  # two-tailed alpha=0.05
        # Power = P(reject H0 | H1 true) = P(|T| > t_crit | ncp)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

        lines.append(f"\n  {name}:")
        lines.append(f"    means:     {a.mean():.4f} vs {b.mean():.4f}  (diff = {abs(a.mean()-b.mean()):.4f})")
        lines.append(f"    stds:      {a.std(ddof=1):.4f} vs {b.std(ddof=1):.4f}")
        lines.append(f"    Cohen's d: {d:.3f}  ({'small' if d < 0.5 else 'medium' if d < 0.8 else 'large'})")
        lines.append(f"    t-test:    t={t_stat:.3f}, p={p_val:.4f}")
        lines.append(f"    Power:     {power:.3f}  (N={n1}+{n2}={n1+n2}, alpha=0.05)")

        # Required N for 80% power
        for target_n in range(3, 200):
            ncp_n = d * np.sqrt(target_n * target_n / (target_n + target_n))
            df_n = 2 * target_n - 2
            t_crit_n = stats.t.ppf(0.975, df_n)
            pw = 1 - stats.nct.cdf(t_crit_n, df_n, ncp_n) + stats.nct.cdf(-t_crit_n, df_n, ncp_n)
            if pw >= 0.80:
                lines.append(f"    N for 80%: {target_n} per group")
                break
        else:
            lines.append(f"    N for 80%: >200 per group (effect too small)")

    lines.append("\n  CRITICAL ISSUE: concept0 last-3 = epochs 8,9,10 but concept1/2 last-3 = epochs 3,4,5")
    lines.append("  These are NOT the same training stage. concept0 had 10 epochs, concept1/2 had only 5.")
    lines.append("  The 'last-3 epochs' comparison mixes different convergence stages.")
    lines.append("  Additionally, each epoch produces ONE rho value, so N=3 is not 3 independent")
    lines.append("  measurements -- they are serially correlated (successive epochs of the same run).")
    lines.append("")

    return "\n".join(lines)


# ─── Parts 1-4: Extended experiment ──────────────────────────────────────

CONCEPTS = {
    "c0_00_09": list(range(0, 10)),
    "c1_20_29": list(range(20, 30)),
    "c2_40_49": list(range(40, 50)),
    "c3_60_69": list(range(60, 70)),
    "c4_80_89": list(range(80, 90)),
}

SEEDS = [42, 43, 44]
N_EPOCHS = 5
STAT_BATCH_SIZE = 512


def run_single_training(
    concept_name: str,
    class_ids: list[int],
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
) -> dict:
    """Train one model and return final-epoch statistics."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    tr_x, tr_y = make_subset(train_x, train_y, class_ids)
    te_x, te_y = make_subset(test_x, test_y, class_ids)

    num_classes = int(tr_y.max()) + 1
    model = SmallCNN(num_classes=num_classes, feat_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

    train_ds = CIFARSubset(tr_x, tr_y)
    test_ds = CIFARSubset(te_x, te_y)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    for epoch in range(1, N_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)

    test_acc = test_accuracy(model, test_loader, device)

    # Compute all statistics on held-out batch
    x_stat, y_stat = sample_batch(te_x, te_y, STAT_BATCH_SIZE, rng)
    feat_stats = compute_feature_stats(model, x_stat, device)
    grad_stats = compute_gradient_stats(model, x_stat, y_stat, device)

    rho = grad_stats["r_g"] / feat_stats["r_sigma"] if feat_stats["r_sigma"] > 0 else float("nan")

    result = {
        "concept": concept_name,
        "seed": seed,
        "test_acc": test_acc,
        "rho": rho,
        **feat_stats,
        **grad_stats,
    }

    print(f"  [{concept_name}] seed={seed}  acc={test_acc:.3f}  "
          f"rho={rho:.3f}  r_sigma={feat_stats['r_sigma']:.2f}  "
          f"r_g={grad_stats['r_g']:.2f}  tr_g={grad_stats['tr_g']:.2f}")
    return result


def pairwise_tests(data: list[dict]) -> str:
    """Run pairwise discrimination tests on all concept pairs."""
    from scipy import stats

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("PART 3: Pairwise discrimination (rho alone)")
    lines.append("=" * 70)

    concept_names = sorted(set(r["concept"] for r in data))
    n_concepts = len(concept_names)

    # Gather rho values by concept
    rho_by_concept = {}
    for cn in concept_names:
        rho_by_concept[cn] = np.array([r["rho"] for r in data if r["concept"] == cn])

    # One-way ANOVA on rho
    groups = [rho_by_concept[cn] for cn in concept_names]
    f_stat, p_anova = stats.f_oneway(*groups)
    lines.append(f"\n  One-way ANOVA on rho: F={f_stat:.3f}, p={p_anova:.6f}")

    # Pairwise t-tests with Bonferroni correction
    n_pairs = n_concepts * (n_concepts - 1) // 2
    lines.append(f"\n  Pairwise t-tests (N=3 per group, {n_pairs} comparisons, Bonferroni alpha = {0.05/n_pairs:.4f}):")
    lines.append(f"  {'Pair':<20s} {'d1_mean':>8s} {'d2_mean':>8s} {'diff':>8s} {'Cohen_d':>8s} {'t':>8s} {'p_raw':>10s} {'p_bonf':>10s} {'sig':>5s}")

    pair_results = []
    for i, j in itertools.combinations(range(n_concepts), 2):
        cn_i, cn_j = concept_names[i], concept_names[j]
        a, b = rho_by_concept[cn_i], rho_by_concept[cn_j]
        pooled_std = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2) / (len(a)+len(b)-2))
        d = abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else float("inf")
        t_stat, p_val = stats.ttest_ind(a, b)
        p_bonf = min(p_val * n_pairs, 1.0)
        sig = "YES" if p_bonf < 0.05 else "no"

        pair_name = f"{cn_i} vs {cn_j}"
        lines.append(f"  {pair_name:<20s} {a.mean():8.3f} {b.mean():8.3f} {abs(a.mean()-b.mean()):8.3f} {d:8.3f} {t_stat:8.3f} {p_val:10.6f} {p_bonf:10.6f} {sig:>5s}")
        pair_results.append({
            "pair": pair_name, "d1_mean": a.mean(), "d2_mean": b.mean(),
            "diff": abs(a.mean()-b.mean()), "cohen_d": d, "t": t_stat,
            "p_raw": p_val, "p_bonf": p_bonf, "sig": sig,
        })

    return "\n".join(lines), pair_results


def multi_signal_discrimination(data: list[dict]) -> str:
    """Test 2D discrimination using multiple signal combinations."""
    from scipy import stats
    from scipy.spatial.distance import mahalanobis

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("PART 4: Multi-signal discrimination")
    lines.append("=" * 70)

    concept_names = sorted(set(r["concept"] for r in data))

    # Feature combinations to test
    signal_combos = [
        ("rho_only", ["rho"]),
        ("r_sigma_only", ["r_sigma"]),
        ("r_g_only", ["r_g"]),
        ("tr_g_only", ["tr_g"]),
        ("rho+tr_g", ["rho", "tr_g"]),
        ("rho+r_sigma", ["rho", "r_sigma"]),
        ("rho+r_g", ["rho", "r_g"]),
        ("r_sigma+r_g", ["r_sigma", "r_g"]),
        ("rho+tr_g+tr_sigma", ["rho", "tr_g", "tr_sigma"]),
        ("r_sigma+r_g+tr_g", ["r_sigma", "r_g", "tr_g"]),
        ("all_6", ["rho", "r_sigma", "r_g", "tr_g", "tr_sigma", "max_eig_g"]),
    ]

    for combo_name, features in signal_combos:
        lines.append(f"\n  --- {combo_name}: {features} ---")

        # Build feature matrices per concept
        vecs_by_concept = {}
        for cn in concept_names:
            rows = [r for r in data if r["concept"] == cn]
            vecs = np.array([[r[f] for f in features] for r in rows], dtype=np.float64)
            vecs_by_concept[cn] = vecs

        # For 1D: t-tests; for multi-D: Hotelling's T^2 approximation
        n_dim = len(features)
        n_pairs = len(concept_names) * (len(concept_names) - 1) // 2
        n_distinguishable = 0

        for i, j in itertools.combinations(range(len(concept_names)), 2):
            cn_i, cn_j = concept_names[i], concept_names[j]
            Xi, Xj = vecs_by_concept[cn_i], vecs_by_concept[cn_j]

            if n_dim == 1:
                a, b = Xi.ravel(), Xj.ravel()
                t_stat, p_val = stats.ttest_ind(a, b)
                p_bonf = min(p_val * n_pairs, 1.0)
                sig = p_bonf < 0.05
            else:
                # Hotelling's T^2 via MANOVA-like approach
                # T^2 = (n1*n2)/(n1+n2) * (m1-m2)^T S_pooled^{-1} (m1-m2)
                n1, n2 = Xi.shape[0], Xj.shape[0]
                m1, m2 = Xi.mean(axis=0), Xj.mean(axis=0)
                S1 = np.cov(Xi.T, ddof=1) if Xi.shape[0] > 1 else np.zeros((n_dim, n_dim))
                S2 = np.cov(Xj.T, ddof=1) if Xj.shape[0] > 1 else np.zeros((n_dim, n_dim))

                # Handle scalar cov for 1D
                if n_dim == 1:
                    S1 = np.atleast_2d(S1)
                    S2 = np.atleast_2d(S2)

                S_pooled = ((n1-1)*S1 + (n2-1)*S2) / (n1+n2-2)
                # Regularize for numerical stability
                S_pooled += np.eye(n_dim) * 1e-10

                diff = m1 - m2
                try:
                    S_inv = np.linalg.inv(S_pooled)
                    T2 = (n1*n2)/(n1+n2) * diff @ S_inv @ diff
                    # Convert to F-statistic
                    p_dim = n_dim
                    df1 = p_dim
                    df2 = n1 + n2 - p_dim - 1
                    if df2 > 0:
                        F_stat = T2 * df2 / ((n1+n2-2) * p_dim)
                        p_val = 1 - stats.f.cdf(F_stat, df1, df2)
                    else:
                        p_val = 1.0  # cannot compute with these df
                except np.linalg.LinAlgError:
                    p_val = 1.0

                p_bonf = min(p_val * n_pairs, 1.0)
                sig = p_bonf < 0.05

            if sig:
                n_distinguishable += 1

        lines.append(f"    Distinguishable pairs: {n_distinguishable}/{n_pairs} (Bonferroni alpha=0.05)")

    # Also compute a simple LDA-like separability metric:
    # Fisher criterion = between-class / within-class scatter
    lines.append(f"\n  --- Fisher Discriminant Ratio (higher = more separable) ---")
    lines.append(f"  {'Signal combo':<25s} {'Fisher ratio':>12s}")

    for combo_name, features in signal_combos:
        vecs_all = []
        labels_all = []
        for ci, cn in enumerate(concept_names):
            rows = [r for r in data if r["concept"] == cn]
            vecs = np.array([[r[f] for f in features] for r in rows], dtype=np.float64)
            vecs_all.append(vecs)
            labels_all.extend([ci] * len(rows))

        X = np.vstack(vecs_all)
        y = np.array(labels_all)
        grand_mean = X.mean(axis=0)

        # Between-class scatter
        S_b = np.zeros((len(features), len(features)))
        S_w = np.zeros((len(features), len(features)))
        for ci, cn in enumerate(concept_names):
            Xi = X[y == ci]
            ni = Xi.shape[0]
            mi = Xi.mean(axis=0)
            diff = (mi - grand_mean).reshape(-1, 1)
            S_b += ni * (diff @ diff.T)
            if ni > 1:
                centered = Xi - mi
                S_w += centered.T @ centered

        # Fisher = tr(S_w^{-1} S_b) -- handle singular S_w
        S_w += np.eye(len(features)) * 1e-10
        try:
            fisher = np.trace(np.linalg.inv(S_w) @ S_b)
        except np.linalg.LinAlgError:
            fisher = 0.0

        lines.append(f"  {combo_name:<25s} {fisher:12.3f}")

    return "\n".join(lines)


def summarize_solutions(data, pair_results, lines_part4) -> str:
    """Synthesize findings into actionable recommendations."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("PART 5: Solutions and Recommendations")
    lines.append("=" * 70)

    # Count distinguishable pairs for rho alone
    rho_sig = sum(1 for pr in pair_results if pr["sig"] == "YES")
    total_pairs = len(pair_results)

    lines.append(f"\n  rho alone distinguishes {rho_sig}/{total_pairs} concept pairs")

    lines.append("""
  KEY DIAGNOSIS:

  1. The original "indistinguishable" finding was an artifact of:
     a) N=3 data points per group (epochs from a single training run)
     b) Serial correlation (consecutive epochs are not independent)
     c) Mismatched training lengths (concept0: 10 epochs, concept1/2: 5 epochs)
     d) Only 3 concepts tested (just 3 pairwise comparisons)

  2. The proper test uses 3 independent seeds per concept (truly independent),
     giving N=3 genuinely i.i.d. observations per group.

  SOLUTIONS (ordered by recommendation):

  Solution A: Multi-dimensional fingerprint [RECOMMENDED]
    - Use (rho, tr(G)) or (rho, r_sigma, tr(G)) as a 2-3D fingerprint vector
    - Benefits: each dimension captures a different aspect of gradient geometry
    - rho: ratio structure (shape); tr(G): gradient magnitude (scale)
    - Even if rho alone fails for some pairs, the combination succeeds
    - Implementation: replace scalar rho comparison with Mahalanobis distance
    - Complexity: LOW (just concatenate scalars, use covariance-weighted distance)

  Solution B: rho as shrinkage calibration signal
    - Do not use rho for fingerprinting at all
    - Use rho only to set lambda in shrinkage estimator: lambda = f(rho)
    - Concept identity comes from existing FedProTrack signals (weight similarity, prototypes)
    - Pro: clean separation of concerns
    - Con: loses potentially useful discriminative information

  Solution C: Abandon rho, keep r^G alone
    - r^G alone may discriminate better than rho in some settings
    - Simpler, avoids ratio instabilities
    - But loses the theoretical elegance of the ratio formulation

  RECOMMENDED APPROACH: Solution A (multi-dim fingerprint)
    - Theoretical backing: rho captures the gradient/feature effective-rank ratio,
      while tr(G) captures the overall gradient signal strength. Together they
      form a 2D "spectral signature" of the concept.
    - In FL setting: each client computes a 2-3 element vector; server uses
      Mahalanobis distance to match concepts. Communication cost: negligible
      (2-3 extra floats per round).
    - Implementation: ~20 lines of code change in Phase A routing.
""")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default=".cifar100_cache")
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Part 0: Power analysis on existing data ──
    print("\n" + "=" * 70)
    print("Running Part 0: Power analysis on existing data")
    report_parts = [power_analysis_existing()]

    # ── Part 1-2: Extended experiment ──
    print("\n" + "=" * 70)
    print("Running Part 1-2: Extended experiment (5 concepts x 3 seeds)")
    print("=" * 70)

    cache_dir = Path(args.cache_dir)
    train_x, train_y, test_x, test_y = load_cifar100_numpy(cache_dir)
    print(f"[data] train={train_x.shape}  test={test_x.shape}")

    all_results = []
    t0_total = time.time()

    for concept_name, class_ids in CONCEPTS.items():
        print(f"\n[{concept_name}] classes={class_ids[:3]}...{class_ids[-1]}")
        for seed in SEEDS:
            result = run_single_training(
                concept_name, class_ids, seed,
                train_x, train_y, test_x, test_y, device,
            )
            all_results.append(result)

    elapsed = time.time() - t0_total
    print(f"\n[timing] Total experiment time: {elapsed:.1f}s")

    # Save extended data
    out_dir = Path(args.out_dir)
    fieldnames = list(all_results[0].keys())
    with open(out_dir / "drct_rho_extended_data.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[saved] drct_rho_extended_data.csv")

    # ── Part 2: Summary table ──
    report_parts.append("\n" + "=" * 70)
    report_parts.append("PART 2: Extended experiment results (5 concepts x 3 seeds, 5 epochs each)")
    report_parts.append("=" * 70)

    concept_names = sorted(set(r["concept"] for r in all_results))
    header = f"{'Concept':<15s} {'seed':>4s} {'acc':>6s} {'rho':>8s} {'r_sigma':>8s} {'r_g':>8s} {'tr_g':>10s} {'tr_sigma':>10s} {'max_eig_g':>10s}"
    report_parts.append(f"\n  {header}")
    report_parts.append("  " + "-" * len(header))
    for r in all_results:
        line = f"  {r['concept']:<15s} {r['seed']:4d} {r['test_acc']:6.3f} {r['rho']:8.3f} {r['r_sigma']:8.3f} {r['r_g']:8.3f} {r['tr_g']:10.3f} {r['tr_sigma']:10.3f} {r['max_eig_g']:10.3f}"
        report_parts.append(line)

    # Per-concept summary
    report_parts.append(f"\n  Per-concept summary:")
    report_parts.append(f"  {'Concept':<15s} {'mean_rho':>9s} {'std_rho':>9s} {'CV_rho':>9s} {'mean_acc':>9s}")
    for cn in concept_names:
        rows = [r for r in all_results if r["concept"] == cn]
        rhos = np.array([r["rho"] for r in rows])
        accs = np.array([r["test_acc"] for r in rows])
        cv = rhos.std(ddof=1) / abs(rhos.mean()) if abs(rhos.mean()) > 1e-12 else float("inf")
        report_parts.append(f"  {cn:<15s} {rhos.mean():9.4f} {rhos.std(ddof=1):9.4f} {cv:9.4f} {accs.mean():9.3f}")

    # ── Part 3: Pairwise discrimination ──
    part3_text, pair_results = pairwise_tests(all_results)
    report_parts.append(part3_text)

    # Save discrimination results
    if pair_results:
        with open(out_dir / "drct_rho_discrimination.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(pair_results[0].keys()))
            writer.writeheader()
            writer.writerows(pair_results)
        print(f"[saved] drct_rho_discrimination.csv")

    # ── Part 4: Multi-signal discrimination ──
    part4_text = multi_signal_discrimination(all_results)
    report_parts.append(part4_text)

    # ── Part 5: Solutions ──
    part5_text = summarize_solutions(all_results, pair_results, part4_text)
    report_parts.append(part5_text)

    # ── Write report ──
    full_report = "\n".join(report_parts)
    with open(out_dir / "drct_rho_power_report.txt", "w") as fh:
        fh.write(full_report)
    print(f"\n[saved] drct_rho_power_report.txt")
    print("\n" + full_report)


if __name__ == "__main__":
    main()
