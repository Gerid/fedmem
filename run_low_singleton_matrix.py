from __future__ import annotations

"""Low-singleton concept matrix generator and validation.

Provides ``generate_concept_matrix_low_singleton``, a wrapper around the
existing ``generate_concept_matrix`` that post-processes each time column to
ensure every active concept has at least ``min_group_size`` clients.  When
``min_group_size=1`` (the default), behaviour is identical to the original.

Design
------
The post-processing works column-by-column (each time step independently):

1. Count how many clients are assigned to each concept.
2. Identify *singleton* concepts (count < min_group_size).
3. Identify *donor* concepts (count > min_group_size) that can give up a
   client without themselves falling below min_group_size.
4. For each singleton concept, steal a client from the largest donor and
   reassign it to the singleton concept.
5. If no donors exist (e.g. K is too small relative to active concepts),
   *merge* the smallest singleton into the largest group to reduce
   fragmentation.

The algorithm is deterministic given the same seed.

Recurrence is preserved because only the *assignment* within a column changes,
not the set of concepts that appear across the full matrix.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so fedprotrack is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fedprotrack.drift_generator.concept_matrix import generate_concept_matrix


def _fix_column(
    col: np.ndarray,
    min_group_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Redistribute assignments in one time-step column.

    Parameters
    ----------
    col : np.ndarray of shape (K,)
        Concept assignments for all clients at one time step.
    min_group_size : int
        Minimum number of clients per active concept.
    rng : np.random.Generator
        Used for tie-breaking when choosing which client to reassign.

    Returns
    -------
    np.ndarray of shape (K,)
        Adjusted assignments.
    """
    col = col.copy()
    K = len(col)

    # Iterative: keep fixing until no singletons remain or we cannot improve.
    for _ in range(K):
        concepts, counts = np.unique(col, return_counts=True)
        small_mask = counts < min_group_size
        if not small_mask.any():
            break

        # Sort small concepts by count ascending (fix smallest first).
        small_order = np.argsort(counts[small_mask])
        small_concepts = concepts[small_mask][small_order]

        large_mask = counts > min_group_size
        if not large_mask.any():
            # No donor available: merge the smallest group into the largest.
            largest_concept = concepts[np.argmax(counts)]
            smallest_concept = small_concepts[0]
            col[col == smallest_concept] = largest_concept
            continue

        # Pick the smallest singleton and steal one client from the largest donor.
        target_concept = small_concepts[0]
        large_concepts = concepts[large_mask]
        large_counts = counts[large_mask]
        donor_concept = large_concepts[np.argmax(large_counts)]

        # Among donor's clients, pick one at random to reassign.
        donor_clients = np.flatnonzero(col == donor_concept)
        victim = rng.choice(donor_clients)
        col[victim] = target_concept

    return col


def generate_concept_matrix_low_singleton(
    K: int,
    T: int,
    n_concepts: int,
    alpha: float,
    seed: int,
    min_group_size: int = 1,
) -> np.ndarray:
    """Generate a (K, T) concept matrix with bounded singleton ratio.

    Parameters
    ----------
    K : int
        Number of clients.
    T : int
        Number of time steps.
    n_concepts : int
        Size of the shared concept pool.
    alpha : float
        Asynchrony level in [0, 1].
    seed : int
        Random seed for reproducibility.
    min_group_size : int
        Minimum number of clients that must share any active concept at each
        time step.  Set to 1 to preserve original behaviour; set to 2 to
        eliminate singletons.  Must satisfy
        ``min_group_size * n_concepts <= K`` for a feasible solution at
        every step (though the algorithm handles infeasible cases gracefully
        by merging the smallest groups).

    Returns
    -------
    np.ndarray of shape (K, T), dtype int32
    """
    matrix = generate_concept_matrix(
        K=K, T=T, n_concepts=n_concepts, alpha=alpha, seed=seed,
    )

    if min_group_size <= 1:
        return matrix

    rng = np.random.default_rng(seed + 999_999)

    for t in range(T):
        matrix[:, t] = _fix_column(matrix[:, t], min_group_size, rng)

    return matrix.astype(np.int32)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_singleton_ratio(matrix: np.ndarray) -> float:
    """Fraction of (k, t) positions where concept has only 1 client at time t."""
    K, T = matrix.shape
    singleton_count = 0
    for t in range(T):
        col = matrix[:, t]
        concepts, counts = np.unique(col, return_counts=True)
        count_map = dict(zip(concepts, counts))
        for k in range(K):
            if count_map[col[k]] == 1:
                singleton_count += 1
    return singleton_count / (K * T)


def compute_recurrence_ratio(matrix: np.ndarray) -> float:
    """Fraction of clients that revisit a concept they had earlier."""
    K, T = matrix.shape
    recurrence_count = 0
    for k in range(K):
        seen = set()
        for t in range(T):
            c = matrix[k, t]
            if c in seen:
                recurrence_count += 1
            seen.add(c)
    return recurrence_count / (K * T)


def compute_active_concepts_per_step(matrix: np.ndarray) -> np.ndarray:
    """Number of distinct concepts active at each time step."""
    T = matrix.shape[1]
    return np.array([len(np.unique(matrix[:, t])) for t in range(T)])


def analyze_matrix(label: str, matrix: np.ndarray) -> dict:
    """Compute and print summary statistics for a concept matrix."""
    K, T = matrix.shape
    sr = compute_singleton_ratio(matrix)
    rr = compute_recurrence_ratio(matrix)
    active = compute_active_concepts_per_step(matrix)
    n_unique = len(np.unique(matrix))

    stats = {
        "label": label,
        "K": K,
        "T": T,
        "n_unique_concepts": n_unique,
        "singleton_ratio": sr,
        "recurrence_ratio": rr,
        "mean_active_concepts": float(active.mean()),
        "min_active_concepts": int(active.min()),
        "max_active_concepts": int(active.max()),
    }

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Shape:              ({K}, {T})")
    print(f"  Unique concepts:    {n_unique}")
    print(f"  Singleton ratio:    {sr:.3f}  ({sr*100:.1f}%)")
    print(f"  Recurrence ratio:   {rr:.3f}  ({rr*100:.1f}%)")
    print(f"  Active concepts:    mean={active.mean():.1f}  "
          f"min={active.min()}  max={active.max()}")
    print(f"  Matrix:\n{matrix}")
    return stats


# ---------------------------------------------------------------------------
# Main: validation
# ---------------------------------------------------------------------------

def main() -> None:
    import json

    output_lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        output_lines.append(msg)

    log("Low-Singleton Concept Matrix Validation")
    log("=" * 60)

    configs = [
        # (label, K, T, n_concepts, alpha, seed, min_group_size)
        ("Original (CIFAR default: K=6 T=12 C=4 a=0.75)", 6, 12, 4, 0.75, 42, 1),
        ("Low-singleton (min_group=2, same params)",       6, 12, 4, 0.75, 42, 2),
        ("Original (K=10 T=30 C=4 a=0.75)",              10, 30, 4, 0.75, 42, 1),
        ("Low-singleton (K=10 T=30 C=4, min_group=2)",   10, 30, 4, 0.75, 42, 2),
        ("Low-singleton (K=10 T=30 C=4, min_group=3)",   10, 30, 4, 0.75, 42, 3),
        ("Original (K=10 T=30 C=6 a=0.75)",              10, 30, 6, 0.75, 42, 1),
        ("Low-singleton (K=10 T=30 C=6, min_group=2)",   10, 30, 6, 0.75, 42, 2),
        ("Original (K=10 T=30 C=4 a=0.50)",              10, 30, 4, 0.50, 42, 1),
        ("Low-singleton (K=10 T=30 C=4 a=0.50 mg=2)",    10, 30, 4, 0.50, 42, 2),
        # Synchronous case: alpha=0 -> all same, no singletons anyway
        ("Synchronous (K=10 T=30 C=4 a=0.0)",            10, 30, 4, 0.0,  42, 1),
        # Stress: many concepts relative to K
        ("Stress: K=6 C=5 (tight, min_group=2)",          6, 12, 5, 0.75, 42, 2),
        # Ideal scenario for FedProTrack
        ("Ideal: K=20 C=4 min_group=3",                  20, 30, 4, 0.75, 42, 3),
    ]

    all_stats: list[dict] = []

    for label, K, T, C, alpha, seed, mg in configs:
        matrix = generate_concept_matrix_low_singleton(
            K=K, T=T, n_concepts=C, alpha=alpha, seed=seed,
            min_group_size=mg,
        )
        stats = analyze_matrix(label, matrix)
        all_stats.append(stats)
        # Capture the printed output
        log(f"\n--- {label} ---")
        log(f"  Singleton ratio: {stats['singleton_ratio']:.3f}")
        log(f"  Recurrence ratio: {stats['recurrence_ratio']:.3f}")
        log(f"  Active concepts: mean={stats['mean_active_concepts']:.1f}")

    # Reproducibility check
    log("\n" + "=" * 60)
    log("Reproducibility check:")
    m1 = generate_concept_matrix_low_singleton(10, 30, 4, 0.75, 42, min_group_size=2)
    m2 = generate_concept_matrix_low_singleton(10, 30, 4, 0.75, 42, min_group_size=2)
    match = np.array_equal(m1, m2)
    log(f"  Same seed produces identical matrix: {match}")

    # Backward compatibility check
    log("\nBackward compatibility check:")
    m_orig = generate_concept_matrix(10, 30, 4, 0.75, 42)
    m_compat = generate_concept_matrix_low_singleton(10, 30, 4, 0.75, 42, min_group_size=1)
    compat = np.array_equal(m_orig, m_compat)
    log(f"  min_group_size=1 identical to original: {compat}")

    # Summary table
    log("\n" + "=" * 60)
    log("SUMMARY TABLE")
    log("=" * 60)
    log(f"{'Config':<50} {'Singleton%':>10} {'Recurrence%':>12} {'Concepts':>10}")
    log("-" * 82)
    for s in all_stats:
        log(f"{s['label']:<50} {s['singleton_ratio']*100:>9.1f}% "
            f"{s['recurrence_ratio']*100:>10.1f}% "
            f"{s['mean_active_concepts']:>9.1f}")

    # Write to file
    out_dir = Path(__file__).resolve().parent / "tmp"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "matrix_validation.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nResults saved to {out_path}")

    # Also save structured JSON
    json_path = out_dir / "matrix_validation.json"
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
