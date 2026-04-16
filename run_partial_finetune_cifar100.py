from __future__ import annotations

"""Partial fine-tuning robustness experiment for FedProTrack rebuttal W4/Q4.

Tests whether FPT fingerprints remain stable under partial backbone
fine-tuning by simulating feature drift with concept-consistent Gaussian
noise at three levels:

1. **Fully frozen** (noise_scale=0.0): standard FPT with frozen ResNet-18
2. **Last-layer unfrozen** (noise_scale=0.05): simulates last-FC drift
3. **Last-2-layers unfrozen** (noise_scale=0.15): simulates deeper drift

For each noise level, we run FPT, FedAvg, and Oracle across 3 seeds and
measure accuracy, clustering error eta, and cross-round fingerprint
stability (cosine similarity of same-client fingerprints across
consecutive rounds).

Outputs:
  - results_partial_finetune/partial_finetune.json
"""

import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.drift_generator.generator import DriftDataset
from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]

NOISE_LEVELS: list[dict[str, object]] = [
    {"label": "frozen",              "noise_scale": 0.0},
    {"label": "last_layer_unfrozen", "noise_scale": 0.05},
    {"label": "last_2_layers_unfrozen", "noise_scale": 0.15},
]

# Dataset settings: CIFAR-100 recurrence, K=20, T=100, rho=25
CIFAR_K = 20
CIFAR_T = 100
CIFAR_RHO = 25
CIFAR_N_FEATURES = 128
LR = 0.01
N_EPOCHS = 5


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def _add_concept_consistent_noise(
    dataset: DriftDataset,
    noise_scale: float,
    seed: int,
) -> DriftDataset:
    """Return a copy of *dataset* with concept-consistent per-round noise.

    For each (concept_id, round), a fixed random direction is sampled
    once and added to all feature vectors belonging to that concept in
    that round.  The noise magnitude is proportional to *noise_scale*.
    This simulates gradual feature drift from partial backbone
    fine-tuning while ensuring intra-concept consistency.

    Parameters
    ----------
    dataset : DriftDataset
        Original frozen-feature dataset.
    noise_scale : float
        Standard deviation of the additive Gaussian noise.  0.0 means
        no noise (fully frozen baseline).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    DriftDataset
        New dataset with noisy features (labels unchanged).
    """
    if noise_scale <= 0.0:
        return dataset

    K, T = dataset.concept_matrix.shape
    n_features = next(iter(dataset.data.values()))[0].shape[1]
    rng = np.random.RandomState(seed + 99999)

    # Pre-generate a fixed drift direction per (concept, round).
    # Each concept gets a stable base direction; per-round magnitude grows
    # linearly to simulate cumulative fine-tuning drift.
    n_concepts = int(dataset.concept_matrix.max()) + 1
    concept_base_dirs: dict[int, np.ndarray] = {}
    for c in range(n_concepts):
        direction = rng.randn(n_features).astype(np.float32)
        direction /= max(np.linalg.norm(direction), 1e-8)
        concept_base_dirs[c] = direction

    new_data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            X, y = dataset.data[(k, t)]
            concept_id = int(dataset.concept_matrix[k, t])
            # Scale grows with round index to mimic cumulative drift.
            round_fraction = (t + 1) / T
            magnitude = noise_scale * round_fraction
            # Deterministic per-(concept, round) noise vector.
            per_round_rng = np.random.RandomState(
                seed + 77777 + concept_id * T + t
            )
            noise_vec = (
                concept_base_dirs[concept_id] * magnitude
                + per_round_rng.randn(n_features).astype(np.float32)
                * (noise_scale * 0.3)
            )
            X_noisy = X.astype(np.float32) + noise_vec[np.newaxis, :]
            new_data[(k, t)] = (X_noisy, y.copy())

    return DriftDataset(
        concept_matrix=dataset.concept_matrix.copy(),
        data=new_data,
        config=dataset.config,
        concept_specs=dataset.concept_specs,
    )


# ---------------------------------------------------------------------------
# Cross-round fingerprint stability
# ---------------------------------------------------------------------------

def _compute_fingerprint_stability(
    dataset: DriftDataset,
    n_features: int,
) -> float:
    """Compute mean cosine similarity of same-client fingerprints across
    consecutive rounds.

    Builds a ConceptFingerprint for each (client, round) from the
    dataset's data, then measures how stable each client's fingerprint
    vector is from round t to round t+1.

    Parameters
    ----------
    dataset : DriftDataset
        Dataset (possibly noisy) to build fingerprints from.
    n_features : int
        Feature dimensionality.

    Returns
    -------
    float
        Mean cosine similarity in [0, 1].  Higher = more stable.
    """
    K, T = dataset.concept_matrix.shape

    # Infer n_classes from dataset labels
    all_labels: set[int] = set()
    for (_, _), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1 if all_labels else 2

    # Build fingerprint vectors per (client, round)
    fp_vectors: dict[tuple[int, int], np.ndarray] = {}
    for k in range(K):
        for t in range(T):
            X, y = dataset.data[(k, t)]
            fp = ConceptFingerprint(n_features, n_classes)
            fp.update(X, y)
            fp_vectors[(k, t)] = fp.to_vector()

    # Compute cosine similarity between consecutive rounds for each client
    cosine_sims: list[float] = []
    for k in range(K):
        for t in range(1, T):
            v_prev = fp_vectors[(k, t - 1)]
            v_curr = fp_vectors[(k, t)]
            norm_prev = np.linalg.norm(v_prev)
            norm_curr = np.linalg.norm(v_curr)
            if norm_prev < 1e-12 or norm_curr < 1e-12:
                continue
            cos_sim = float(np.dot(v_prev, v_curr) / (norm_prev * norm_curr))
            cosine_sims.append(cos_sim)

    return float(np.mean(cosine_sims)) if cosine_sims else 0.0


# ---------------------------------------------------------------------------
# Per-seed experiment
# ---------------------------------------------------------------------------

def run_seed(
    seed: int,
    noise_label: str,
    noise_scale: float,
    base_dataset: DriftDataset,
    cfg: CIFAR100RecurrenceConfig,
) -> dict:
    """Run FPT + FedAvg + Oracle for one seed at one noise level."""
    print(f"    [seed={seed}] Injecting noise: scale={noise_scale}")
    dataset = _add_concept_consistent_noise(base_dataset, noise_scale, seed)
    n_concepts = int(dataset.concept_matrix.max()) + 1
    n_features = next(iter(dataset.data.values()))[0].shape[1]

    # --- Fingerprint stability ---
    fp_stability = _compute_fingerprint_stability(dataset, n_features)
    print(f"    [seed={seed}] Fingerprint stability (cosine): {fp_stability:.4f}")

    # --- FedProTrack ---
    tp_config = TwoPhaseConfig(
        omega=2.0,
        kappa=0.7,
        novelty_threshold=0.25,
        loss_novelty_threshold=0.15,
        sticky_dampening=1.5,
        sticky_posterior_gate=0.35,
        merge_threshold=0.85,
        min_count=5.0,
        max_concepts=max(6, n_concepts + 3),
        merge_every=2,
        shrink_every=6,
    )
    runner = FedProTrackRunner(
        config=tp_config,
        seed=seed,
        federation_every=1,
        detector_name="ADWIN",
        lr=LR,
        n_epochs=N_EPOCHS,
        soft_aggregation=False,
        blend_alpha=0.0,
    )
    t0 = time.time()
    fpt_result = runner.run(dataset)
    fpt_elapsed = time.time() - t0

    fpt_log = fpt_result.to_experiment_log()
    fpt_metrics = compute_all_metrics(fpt_log, identity_capable=True)

    # --- FedAvg ---
    exp_cfg = ExperimentConfig(
        generator_config=dataset.config, federation_every=1,
    )
    t0 = time.time()
    fedavg_result = run_fedavg_baseline(
        exp_cfg, dataset=dataset, lr=LR, n_epochs=N_EPOCHS, seed=seed,
    )
    fedavg_elapsed = time.time() - t0

    # --- Oracle ---
    t0 = time.time()
    oracle_result = run_oracle_baseline(
        exp_cfg, dataset=dataset, lr=LR, n_epochs=N_EPOCHS, seed=seed,
    )
    oracle_elapsed = time.time() - t0

    # Build ExperimentLog for Oracle (uses ground-truth concept IDs)
    oracle_log = ExperimentLog(
        ground_truth=dataset.concept_matrix,
        predicted=oracle_result.predicted_concept_matrix,
        accuracy_curve=oracle_result.accuracy_matrix,
        total_bytes=oracle_result.total_bytes,
        method_name="Oracle",
    )
    oracle_metrics = compute_all_metrics(oracle_log, identity_capable=False)

    # Clustering error eta = 1 - concept_re_id_accuracy (None for non-identity methods)
    fpt_re_id = fpt_metrics.concept_re_id_accuracy
    fpt_eta = 1.0 - fpt_re_id if fpt_re_id is not None else None

    return {
        "seed": seed,
        "noise_label": noise_label,
        "noise_scale": noise_scale,
        # Accuracy
        "fpt_mean_acc": float(fpt_result.accuracy_matrix.mean()),
        "fpt_final_acc": float(fpt_result.accuracy_matrix[:, -1].mean()),
        "fedavg_mean_acc": float(fedavg_result.accuracy_matrix.mean()),
        "fedavg_final_acc": float(fedavg_result.accuracy_matrix[:, -1].mean()),
        "oracle_mean_acc": float(oracle_result.accuracy_matrix.mean()),
        "oracle_final_acc": float(oracle_result.accuracy_matrix[:, -1].mean()),
        # Clustering error eta (1 - re-ID accuracy)
        "fpt_re_id": float(fpt_re_id) if fpt_re_id is not None else None,
        "fpt_eta": float(fpt_eta) if fpt_eta is not None else None,
        # Fingerprint stability
        "fingerprint_stability": fp_stability,
        # Timing
        "fpt_elapsed_s": fpt_elapsed,
        "fedavg_elapsed_s": fedavg_elapsed,
        "oracle_elapsed_s": oracle_elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path("results_partial_finetune")
    out_dir.mkdir(exist_ok=True)

    # Build the base (frozen) dataset once; noise variants are derived from it.
    print("Preparing CIFAR-100 recurrence feature cache ...")
    base_cfg = CIFAR100RecurrenceConfig(
        K=CIFAR_K,
        T=CIFAR_T,
        n_samples=400,
        rho=CIFAR_RHO,
        alpha=0.75,
        delta=0.85,
        n_features=CIFAR_N_FEATURES,
        samples_per_coarse_class=120,
        batch_size=256,
        n_workers=0,
        data_root=".cifar100_cache",
        feature_cache_dir=".feature_cache",
        feature_seed=2718,
        seed=42,  # base seed for concept matrix; per-run seed overrides below
        label_split="none",
        min_group_size=1,
    )
    prepare_cifar100_recurrence_feature_cache(base_cfg)

    all_results: list[dict] = []

    for noise_cfg in NOISE_LEVELS:
        label = str(noise_cfg["label"])
        scale = float(noise_cfg["noise_scale"])  # type: ignore[arg-type]

        print(f"\n{'='*65}")
        print(f"  NOISE LEVEL: {label} (scale={scale})")
        print(f"{'='*65}")

        for seed in SEEDS:
            # Regenerate dataset per seed for the concept matrix randomness.
            per_seed_cfg = CIFAR100RecurrenceConfig(
                K=CIFAR_K,
                T=CIFAR_T,
                n_samples=400,
                rho=CIFAR_RHO,
                alpha=0.75,
                delta=0.85,
                n_features=CIFAR_N_FEATURES,
                samples_per_coarse_class=120,
                batch_size=256,
                n_workers=0,
                data_root=".cifar100_cache",
                feature_cache_dir=".feature_cache",
                feature_seed=2718,
                seed=seed,
                label_split="none",
                min_group_size=1,
            )
            base_dataset = generate_cifar100_recurrence_dataset(per_seed_cfg)

            result = run_seed(seed, label, scale, base_dataset, per_seed_cfg)
            all_results.append(result)

            eta_str = (
                f"{result['fpt_eta']:.4f}"
                if result["fpt_eta"] is not None
                else "N/A"
            )
            print(
                f"    [seed={seed}] "
                f"FPT={result['fpt_final_acc']:.4f}  "
                f"FedAvg={result['fedavg_final_acc']:.4f}  "
                f"Oracle={result['oracle_final_acc']:.4f}  "
                f"eta={eta_str}  "
                f"fp_stab={result['fingerprint_stability']:.4f}"
            )

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------
    with open(out_dir / "partial_finetune.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Aggregate and print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*75}")
    print("  PARTIAL FINE-TUNING ROBUSTNESS — AGGREGATE RESULTS")
    print(f"{'='*75}")
    header = (
        f"{'Noise Level':<25} "
        f"{'FPT Acc':>9} {'FedAvg':>9} {'Oracle':>9} "
        f"{'eta':>8} {'FP Stab':>9}"
    )
    print(header)
    print("-" * len(header))

    for noise_cfg in NOISE_LEVELS:
        label = str(noise_cfg["label"])
        subset = [r for r in all_results if r["noise_label"] == label]

        fpt_accs = [r["fpt_final_acc"] for r in subset]
        fedavg_accs = [r["fedavg_final_acc"] for r in subset]
        oracle_accs = [r["oracle_final_acc"] for r in subset]
        etas = [r["fpt_eta"] for r in subset if r["fpt_eta"] is not None]
        fp_stabs = [r["fingerprint_stability"] for r in subset]

        eta_str = f"{np.mean(etas):.4f}" if etas else "N/A   "
        print(
            f"{label:<25} "
            f"{np.mean(fpt_accs):.4f}+/-{np.std(fpt_accs):.3f} "
            f"{np.mean(fedavg_accs):.4f}+/-{np.std(fedavg_accs):.3f} "
            f"{np.mean(oracle_accs):.4f}+/-{np.std(oracle_accs):.3f} "
            f"{eta_str}   "
            f"{np.mean(fp_stabs):.4f}"
        )

    print(f"\nResults saved to {out_dir / 'partial_finetune.json'}")


if __name__ == "__main__":
    main()
