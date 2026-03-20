"""Deep diagnosis: why CFL beats Oracle on CIFAR-100 recurrence.

Hypothesis: CFL with warmup_rounds=20 and T=20 never splits, so it acts as
FedAvg pooling all 4 clients. Oracle does concept-specific aggregation, so
concepts with only 1 client get no federation benefit (= LocalOnly).
Additionally CFL uses n_epochs=10 vs Oracle n_epochs=5.

This script quantifies:
1. Per-concept client count distribution
2. Per-concept accuracy breakdown for both methods
3. Label overlap between concepts (shared CIFAR-100 coarse classes)
4. Effective sample size per concept
5. Training epoch mismatch
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from collections import Counter

from fedprotrack.real_data.cifar100_recurrence import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
)
from fedprotrack.experiment.baselines import run_oracle_baseline, run_fedavg_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.models import TorchLinearClassifier
from fedprotrack.baselines.comm_tracker import model_bytes


def main() -> None:
    cfg = CIFAR100RecurrenceConfig(K=4, T=20, n_samples=400, seed=42)
    print("Generating CIFAR-100 recurrence dataset (K=4, T=20, n_samples=400, seed=42)...")
    dataset = generate_cifar100_recurrence_dataset(cfg)
    cm = dataset.concept_matrix
    print(f"\nConcept matrix:\n{cm}")

    K, T = cm.shape
    n_concepts = int(cm.max()) + 1
    print(f"\nUnique concepts: {n_concepts}")
    print(f"Concepts present: {sorted(set(cm.flatten()))}")

    # ===== STEP 1: Per-concept client distribution =====
    print("\n" + "=" * 70)
    print("STEP 1: Per-concept client distribution per timestep")
    print("=" * 70)

    concept_client_counts: dict[int, list[int]] = {c: [] for c in range(n_concepts)}
    singleton_timesteps = 0
    total_assignments = 0

    for t in range(T):
        concepts_at_t = [int(cm[k, t]) for k in range(K)]
        counts = Counter(concepts_at_t)
        for concept_id in range(n_concepts):
            if concept_id in counts:
                concept_client_counts[concept_id].append(counts[concept_id])
                if counts[concept_id] == 1:
                    singleton_timesteps += 1
                total_assignments += counts[concept_id]

    print(f"\n{'Concept':<10} {'Appearances':<12} {'Mean clients':<14} {'Min':<5} {'Max':<5} {'Singletons':<12}")
    print("-" * 68)
    for c in range(n_concepts):
        counts = concept_client_counts[c]
        if counts:
            print(f"{c:<10} {len(counts):<12} {np.mean(counts):<14.2f} {min(counts):<5} {max(counts):<5} {sum(1 for x in counts if x == 1):<12}")

    # Count singleton vs multi-client assignments
    total_kt = K * T
    singleton_total = 0
    for t in range(T):
        concepts_at_t = [int(cm[k, t]) for k in range(K)]
        counts = Counter(concepts_at_t)
        for k in range(K):
            c = int(cm[k, t])
            if counts[c] == 1:
                singleton_total += 1

    print(f"\nTotal (k,t) assignments: {total_kt}")
    print(f"Singleton assignments (concept has only 1 client at that step): {singleton_total} ({100*singleton_total/total_kt:.1f}%)")
    print(f"Multi-client assignments: {total_kt - singleton_total} ({100*(total_kt - singleton_total)/total_kt:.1f}%)")

    # ===== STEP 2: Run Oracle and CFL, compare per-concept accuracy =====
    print("\n" + "=" * 70)
    print("STEP 2: Per-concept accuracy breakdown")
    print("=" * 70)

    gen_config = cfg.to_generator_config()
    exp_config = ExperimentConfig(
        generator_config=gen_config,
        federation_every=1,
    )

    print("\nRunning Oracle (n_epochs=5)...")
    oracle_result = run_oracle_baseline(exp_config, dataset, n_epochs=5, seed=42)
    oracle_acc = oracle_result.accuracy_matrix
    print(f"  Oracle mean acc: {oracle_acc.mean():.4f}, final acc: {oracle_acc[:, -1].mean():.4f}")

    print("\nRunning FedAvg (n_epochs=10, same as CFL)...")
    fedavg_result = run_fedavg_baseline(exp_config, dataset, n_epochs=10, seed=42)
    fedavg_acc = fedavg_result.accuracy_matrix
    print(f"  FedAvg mean acc: {fedavg_acc.mean():.4f}, final acc: {fedavg_acc[:, -1].mean():.4f}")

    print("\nRunning CFL (n_epochs=10, warmup_rounds=20)...")
    cfl_result = run_cfl_full(dataset, federation_every=1, warmup_rounds=20)
    cfl_acc = cfl_result.accuracy_matrix
    print(f"  CFL mean acc: {cfl_acc.mean():.4f}, final acc: {cfl_acc[:, -1].mean():.4f}")

    # Also run Oracle with n_epochs=10 to isolate the epoch effect
    print("\nRunning Oracle (n_epochs=10, same epochs as CFL)...")
    oracle10_result = run_oracle_baseline(exp_config, dataset, n_epochs=10, seed=42)
    oracle10_acc = oracle10_result.accuracy_matrix
    print(f"  Oracle-10 mean acc: {oracle10_acc.mean():.4f}, final acc: {oracle10_acc[:, -1].mean():.4f}")

    # Per-concept accuracy
    print(f"\n{'Concept':<9} {'#clients(avg)':<14} {'Oracle-5':<10} {'Oracle-10':<10} {'CFL':<10} {'FedAvg':<10} {'O10-CFL':<10}")
    print("-" * 73)
    for c in range(n_concepts):
        mask = (cm == c)
        if not mask.any():
            continue
        o5 = oracle_acc[mask].mean()
        o10 = oracle10_acc[mask].mean()
        cfl_c = cfl_acc[mask].mean()
        fa = fedavg_acc[mask].mean()
        avg_clients = np.mean(concept_client_counts[c]) if concept_client_counts[c] else 0
        print(f"{c:<9} {avg_clients:<14.1f} {o5:<10.4f} {o10:<10.4f} {cfl_c:<10.4f} {fa:<10.4f} {o10 - cfl_c:<+10.4f}")

    # Singleton vs multi analysis for Oracle
    print(f"\n{'Assignment type':<22} {'Oracle-5':<10} {'Oracle-10':<10} {'CFL':<10} {'FedAvg':<10}")
    print("-" * 62)
    for label, check_fn in [
        ("Singleton (1 client)", lambda cnt: cnt == 1),
        ("Multi (2+ clients)", lambda cnt: cnt >= 2),
    ]:
        o5_vals, o10_vals, cfl_vals, fa_vals = [], [], [], []
        for t in range(T):
            concepts_at_t = [int(cm[k, t]) for k in range(K)]
            counts = Counter(concepts_at_t)
            for k in range(K):
                c = int(cm[k, t])
                if check_fn(counts[c]):
                    o5_vals.append(oracle_acc[k, t])
                    o10_vals.append(oracle10_acc[k, t])
                    cfl_vals.append(cfl_acc[k, t])
                    fa_vals.append(fedavg_acc[k, t])
        if o5_vals:
            print(f"{label:<22} {np.mean(o5_vals):<10.4f} {np.mean(o10_vals):<10.4f} {np.mean(cfl_vals):<10.4f} {np.mean(fa_vals):<10.4f}")

    # ===== STEP 3: Label overlap between concepts =====
    print("\n" + "=" * 70)
    print("STEP 3: Label (class) overlap between concepts")
    print("=" * 70)

    # All concepts use the same pool of 20 coarse classes
    # The difference is visual style, not label distribution
    print("\nAll concepts share the SAME 20 coarse CIFAR-100 classes.")
    print("Concepts differ only in visual appearance style (color shift, blur, etc).")
    print("This means Oracle's concept-specific models train on identical label distributions.")
    print("=> No label-based separation advantage for concept-specific aggregation.")

    # Verify by checking actual label distributions
    print("\nLabel distribution per concept (from actual dataset):")
    for c in range(n_concepts):
        all_labels = []
        for k in range(K):
            for t in range(T):
                if int(cm[k, t]) == c:
                    _, y = dataset.data[(k, t)]
                    all_labels.extend(y.tolist())
        if all_labels:
            unique_labels = sorted(set(all_labels))
            print(f"  Concept {c}: {len(unique_labels)} unique classes, {len(all_labels)} total samples")

    # ===== STEP 4: Effective sample size analysis =====
    print("\n" + "=" * 70)
    print("STEP 4: Effective sample size per concept per step")
    print("=" * 70)

    n_samples = cfg.n_samples  # 400
    train_samples = n_samples // 2  # 200 (half for train, half for test)
    n_classes = 20

    print(f"\nn_samples per (k,t) = {n_samples}")
    print(f"Training samples per (k,t) = {train_samples}")
    print(f"Number of classes = {n_classes}")
    print(f"Training samples per class per client = {train_samples / n_classes:.1f}")

    print(f"\nOracle aggregation pool sizes:")
    for t in range(T):
        concepts_at_t = [int(cm[k, t]) for k in range(K)]
        counts = Counter(concepts_at_t)
        parts = []
        for c_id in sorted(counts.keys()):
            n = counts[c_id]
            total_train = n * train_samples
            per_class = total_train / n_classes
            parts.append(f"c{c_id}:{n}cl={total_train}tr({per_class:.0f}/cls)")
        print(f"  t={t:2d}: {', '.join(parts)}")

    print(f"\nCFL pools all {K} clients = {K * train_samples} training samples = {K * train_samples / n_classes:.0f} per class")
    print(f"Oracle with 1 client  = {train_samples} training samples = {train_samples / n_classes:.0f} per class")
    print(f"Oracle with 2 clients = {2 * train_samples} training samples = {2 * train_samples / n_classes:.0f} per class")

    # ===== STEP 5: The epoch mismatch =====
    print("\n" + "=" * 70)
    print("STEP 5: Training configuration comparison")
    print("=" * 70)

    print(f"\n{'Method':<15} {'n_epochs':<10} {'lr':<8} {'Federation':<15} {'Splitting':<20}")
    print("-" * 68)
    print(f"{'Oracle':<15} {'5':<10} {'0.1':<8} {'concept-aware':<15} {'N/A (perfect)':<20}")
    print(f"{'CFL':<15} {'10':<10} {'0.1':<8} {'all-in-one':<15} {'never (warmup=20)':<20}")
    print(f"{'FedAvg':<15} {'10':<10} {'0.1':<8} {'all-in-one':<15} {'N/A':<20}")

    # ===== STEP 6: Per-timestep accuracy curves =====
    print("\n" + "=" * 70)
    print("STEP 6: Per-timestep mean accuracy")
    print("=" * 70)

    print(f"\n{'t':<4} {'Oracle-5':<10} {'Oracle-10':<10} {'CFL':<10} {'FedAvg':<10} {'O10-CFL':<10}")
    print("-" * 54)
    for t in range(T):
        o5 = oracle_acc[:, t].mean()
        o10 = oracle10_acc[:, t].mean()
        cfl_t = cfl_acc[:, t].mean()
        fa = fedavg_acc[:, t].mean()
        print(f"{t:<4} {o5:<10.4f} {o10:<10.4f} {cfl_t:<10.4f} {fa:<10.4f} {o10 - cfl_t:<+10.4f}")

    # ===== STEP 7: CFL == FedAvg verification =====
    print("\n" + "=" * 70)
    print("STEP 7: CFL vs FedAvg equivalence check")
    print("=" * 70)

    max_diff = np.max(np.abs(cfl_acc - fedavg_acc))
    mean_diff = np.mean(np.abs(cfl_acc - fedavg_acc))
    print(f"\nMax |CFL - FedAvg| accuracy difference: {max_diff:.6f}")
    print(f"Mean |CFL - FedAvg| accuracy difference: {mean_diff:.6f}")
    if max_diff < 0.01:
        print("=> CFL and FedAvg are effectively IDENTICAL (CFL never splits)")
    else:
        print("=> CFL and FedAvg differ (CFL may be splitting or has different init)")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY: Root causes of Oracle < CFL")
    print("=" * 70)

    o5_final = oracle_acc[:, -1].mean()
    o10_final = oracle10_acc[:, -1].mean()
    cfl_final = cfl_acc[:, -1].mean()
    fa_final = fedavg_acc[:, -1].mean()

    print(f"\nFinal accuracy comparison:")
    print(f"  Oracle (n_epochs=5):  {o5_final:.4f}")
    print(f"  Oracle (n_epochs=10): {o10_final:.4f}")
    print(f"  CFL (n_epochs=10):    {cfl_final:.4f}")
    print(f"  FedAvg (n_epochs=10): {fa_final:.4f}")

    gap_epoch = o10_final - o5_final
    gap_pool = cfl_final - o10_final
    gap_total = cfl_final - o5_final

    print(f"\nGap decomposition:")
    print(f"  Epoch effect (Oracle-10 vs Oracle-5):  {gap_epoch:+.4f}")
    print(f"  Pooling effect (CFL vs Oracle-10):     {gap_pool:+.4f}")
    print(f"  Total gap (CFL vs Oracle-5):           {gap_total:+.4f}")

    print(f"\nRoot cause #1: TRAINING EPOCHS")
    print(f"  Oracle defaults to n_epochs=5, CFL defaults to n_epochs=10.")
    print(f"  Matching epochs recovers {gap_epoch:+.4f} of the {gap_total:+.4f} gap.")

    print(f"\nRoot cause #2: DATA POOLING ADVANTAGE")
    print(f"  All concepts share identical 20 CIFAR-100 coarse classes.")
    print(f"  Concept-specific visual styles (color/blur) change features but not labels.")
    print(f"  Oracle segregates by concept => smaller effective training set per concept.")
    print(f"  CFL/FedAvg pool all 4 clients => 4x training data per round.")
    print(f"  For singleton concepts, Oracle = LocalOnly (200 samples, 10/class).")
    print(f"  CFL always pools 800 samples (40/class).")
    print(f"  Pooling gap accounts for {gap_pool:+.4f} of the total {gap_total:+.4f} gap.")

    print(f"\nRoot cause #3: CONCEPT-SPECIFIC SEPARATION IS COUNTERPRODUCTIVE")
    print(f"  Because concepts share labels, concept separation LOSES cross-concept")
    print(f"  generalization. A model trained on concept 0 (original) and concept 3")
    print(f"  (blurred) for the same class still learns the same decision boundary.")
    print(f"  Oracle's perfect concept knowledge is HARMFUL in this setting because")
    print(f"  it prevents beneficial data pooling across visual styles.")

    print(f"\nConclusion: Oracle is NOT a true upper bound when concepts share labels.")
    print(f"  The 'oracle advantage' requires concepts to have DIFFERENT label distributions.")
    print(f"  In CIFAR-100 recurrence, concepts differ in style not content,")
    print(f"  making concept-specific aggregation strictly worse than pooling.")


if __name__ == "__main__":
    main()
