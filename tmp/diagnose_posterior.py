from __future__ import annotations

"""Diagnose posterior entropy collapse: trace exact values for linear vs adapter."""

import os
import sys
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.posterior.gibbs import GibbsPosterior, TransitionPrior
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig, TwoPhaseFedProTrack
from fedprotrack.posterior.fedprotrack_runner import (
    FedProTrackRunner,
    _build_fingerprint_features,
    _infer_n_classes,
    _infer_n_features,
    _make_model,
    _model_fit,
    _model_predict,
    _model_get_params,
    _model_set_params,
)
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

SEED = 42
K = 4
T = 20
N_SAMPLES = 400
N_FEATURES = 64
FED_EVERY = 5
LR = 0.01
N_EPOCHS = 30

print("=" * 80)
print("POSTERIOR ENTROPY DIAGNOSTIC")
print("=" * 80)

# --- Prepare dataset ---
cache_cfg = CIFAR100RecurrenceConfig(
    K=K, T=T, n_samples=N_SAMPLES, seed=SEED,
    n_features=N_FEATURES, samples_per_coarse_class=30,
)
prepare_cifar100_recurrence_feature_cache(cache_cfg)

dataset_cfg = CIFAR100RecurrenceConfig(
    K=K, T=T, n_samples=N_SAMPLES,
    rho=2.0, alpha=0.75, delta=0.9,
    n_features=N_FEATURES, samples_per_coarse_class=30,
    batch_size=128, n_workers=0, seed=SEED,
)
dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

n_classes = max(int(v) for _, y in dataset.data.values() for v in np.unique(y)) + 1
n_features = N_FEATURES

print(f"\nn_classes={n_classes}, n_features={n_features}")
print(f"K={K}, T={T}, federation_every={FED_EVERY}")
print(f"\nConcept matrix (ground truth):")
print(dataset.concept_matrix)
unique_concepts = sorted(set(int(c) for c in dataset.concept_matrix.flatten()))
print(f"Unique concepts: {unique_concepts} ({len(unique_concepts)} total)")

# --- Compute entropy=0.170 corresponds to what distribution? ---
print("\n" + "=" * 80)
print("WHAT DISTRIBUTION GIVES ENTROPY=0.170?")
print("=" * 80)
# For n concepts, entropy = -sum(p_i * log(p_i))
# If 1 concept: entropy = 0
# If 2 concepts, e.g. p=0.85, 1-p=0.15:
for n_concepts in range(1, 8):
    # Check: uniform distribution
    if n_concepts > 0:
        uniform_ent = np.log(n_concepts)
        print(f"  {n_concepts} concepts, uniform: H={uniform_ent:.4f}")

# Find distribution that gives H=0.170 for various support sizes
target = 0.170
for n_c in [2, 3, 4, 5, 6]:
    # Try: one dominant + rest equal
    for p_dom in np.arange(0.80, 1.00, 0.001):
        p_rest = (1.0 - p_dom) / max(n_c - 1, 1)
        probs = np.array([p_dom] + [p_rest] * (n_c - 1))
        probs = np.clip(probs, 1e-15, 1.0)
        H = -np.sum(probs * np.log(probs))
        if abs(H - target) < 0.002:
            print(f"  n_c={n_c}: p_dom={p_dom:.3f}, p_rest={p_rest:.6f} -> H={H:.4f}")
            break


def run_with_diagnostics(model_type: str, lr: float, n_epochs: int):
    """Run FedProTrack and print per-round posterior diagnostics."""
    print(f"\n{'=' * 80}")
    print(f"MODEL TYPE: {model_type}, lr={lr}, n_epochs={n_epochs}")
    print(f"{'=' * 80}")

    # Build models
    model_n_classes = n_classes
    models = [
        _make_model(
            model_type,
            n_features=n_features,
            n_classes=model_n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=SEED + k,
            hidden_dim=64,
            adapter_dim=16,
        )
        for k in range(K)
    ]

    # Build protocol
    cfg = TwoPhaseConfig(
        omega=1.0,
        kappa=0.8,
        novelty_threshold=0.3,
        loss_novelty_threshold=0.02,
        sticky_dampening=1.0,
        n_features=n_features,
        n_classes=n_classes,
    )
    protocol = TwoPhaseFedProTrack(cfg)
    gibbs = protocol.gibbs

    # State
    model_params = [{}] * K
    prev_assignments = None
    accuracy_matrix = np.zeros((K, T), dtype=np.float64)

    all_posteriors = []

    for t in range(T):
        # Build fingerprints and train
        step_fps = [ConceptFingerprint(n_features, n_classes) for _ in range(K)]

        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            current_slot = prev_assignments.get(k, 0) if prev_assignments else 0

            # Predict
            y_pred = _model_predict(models[k], X_test, slot_id=current_slot)
            acc = float(np.mean(y_pred == y_test)) if len(y_test) else 0.0
            accuracy_matrix[k, t] = acc

            # Train
            _model_fit(models[k], X_train, y_train, slot_id=current_slot)

            # Build fingerprint (raw_input)
            step_fps[k].update(X_train, y_train)

        # Federation?
        is_fed_step = (t + 1) % FED_EVERY == 0
        if is_fed_step:
            print(f"\n--- Federation at t={t} ---")
            print(f"  Accuracy: {[f'{accuracy_matrix[k,t]:.3f}' for k in range(K)]}")
            print(f"  True concepts: {[int(dataset.concept_matrix[k,t]) for k in range(K)]}")

            # Phase A
            client_fps = {k: step_fps[k] for k in range(K)}
            client_losses = {k: 1.0 - accuracy_matrix[k, t] for k in range(K)}

            # --- Manual posterior computation for diagnostics ---
            routing_library = protocol.memory_bank.routing_library
            n_lib = len(routing_library)
            print(f"  Library size: {n_lib}")
            if n_lib > 0:
                print(f"  Library concept IDs: {sorted(routing_library.keys())}")

            if n_lib > 0:
                print(f"\n  Per-client posterior details:")
                for k in range(K):
                    fp = client_fps[k]
                    prev_cid = prev_assignments.get(k) if prev_assignments else None

                    # Compute losses to each concept
                    concept_losses = {}
                    for cid, concept_fp in routing_library.items():
                        loss = gibbs.compute_loss(fp, concept_fp)
                        concept_losses[cid] = loss

                    # Compute raw similarities
                    concept_sims = {}
                    for cid, concept_fp in routing_library.items():
                        try:
                            sim = float(concept_fp.similarity(fp))
                        except Exception:
                            sim = float(fp.similarity(concept_fp))
                        concept_sims[cid] = sim

                    # Compute posterior
                    assignment = gibbs.compute_posterior_from_losses(
                        concept_losses, prev_concept_id=prev_cid,
                    )

                    print(f"    Client {k} (true={int(dataset.concept_matrix[k,t])}, "
                          f"prev={prev_cid}, acc={accuracy_matrix[k,t]:.3f}):")
                    print(f"      Similarities: {concept_sims}")
                    print(f"      Losses:       {concept_losses}")
                    print(f"      Posteriors:   {assignment.probabilities}")
                    print(f"      MAP={assignment.map_concept_id}, "
                          f"entropy={assignment.entropy:.6f}, "
                          f"novel={assignment.is_novel}")

                    # Check novelty via loss threshold
                    routed_loss = concept_losses[assignment.map_concept_id]
                    model_loss = client_losses[k]
                    effective_th = cfg.loss_novelty_threshold
                    print(f"      Routed loss={routed_loss:.6f}, "
                          f"eff_threshold={effective_th:.4f}, "
                          f"model_loss={model_loss:.4f}")
                    if routed_loss > effective_th:
                        print(f"      -> NOVEL by loss threshold!")

            # Actually run Phase A
            a_result = protocol.phase_a(
                client_fps,
                prev_assignments,
                client_losses,
            )
            prev_assignments = a_result.assignments
            all_posteriors.append((t, a_result.posteriors))

            print(f"\n  Phase A result:")
            print(f"    Assignments: {a_result.assignments}")
            print(f"    Spawned={a_result.spawned}, Merged={a_result.merged}")
            print(f"    Library size after: {protocol.memory_bank.n_concepts}")
            print(f"    Avg posterior entropy: {a_result.avg_posterior_entropy:.6f}")

            # Phase B (simplified -- just compute aggregate)
            # Skip actual model download to keep it fast

    # Compute final entropy from soft_assignments
    if all_posteriors:
        all_cids = set()
        for _, post_dict in all_posteriors:
            for pa in post_dict.values():
                all_cids.update(pa.probabilities.keys())
        if all_cids:
            cid_list = sorted(all_cids)
            cid_to_idx = {c: i for i, c in enumerate(cid_list)}
            C = len(cid_list)
            soft_asgn = np.zeros((K, T, C), dtype=np.float64)
            for t_step, post_dict in all_posteriors:
                for k, pa in post_dict.items():
                    for cid, prob in pa.probabilities.items():
                        if cid in cid_to_idx:
                            soft_asgn[k, t_step, cid_to_idx[cid]] = prob

            # Compute entropy the same way as metrics
            eps = 1e-12
            p = np.clip(soft_asgn, eps, None)
            H = -np.sum(p * np.log(p), axis=-1)  # (K, T)
            mean_H = float(H.mean())
            print(f"\n  FINAL assignment_entropy (mean over K*T): {mean_H:.6f}")

            # But wait: non-federation steps have soft_asgn = 0 for ALL concepts
            # That means p = [eps, eps, ..., eps], log(eps) ~ -27.6
            # H = -C * eps * log(eps) which is near zero
            # So the entropy is dominated by the federation steps
            fed_steps = [t_step for t_step, _ in all_posteriors]
            non_fed_steps = [t for t in range(T) if t not in fed_steps]

            if fed_steps:
                H_fed = H[:, fed_steps].mean()
                print(f"  Entropy on federation steps only: {H_fed:.6f}")
            if non_fed_steps:
                H_nonfed = H[:, non_fed_steps].mean()
                print(f"  Entropy on non-federation steps: {H_nonfed:.6f}")

            # THIS IS THE KEY: non-federation steps have all-zero probabilities
            # which get clipped to eps, producing near-zero entropy
            # The final metric averages over ALL K*T cells
            # So entropy = (n_fed_cells * H_fed + n_nonfed_cells * 0) / (K*T)
            n_fed_cells = K * len(fed_steps)
            n_nonfed_cells = K * len(non_fed_steps)
            n_total = K * T
            print(f"\n  Fed cells: {n_fed_cells}, Non-fed cells: {n_nonfed_cells}, Total: {n_total}")
            print(f"  Diluted entropy = {n_fed_cells} * {H_fed:.6f} / {n_total} = {n_fed_cells * H_fed / n_total:.6f}")

            # Print actual per-federation-step entropies
            print(f"\n  Per-federation-step mean entropy:")
            for t_step in fed_steps:
                step_H = float(H[:, t_step].mean())
                print(f"    t={t_step}: H={step_H:.6f}")


# Run both
run_with_diagnostics("linear", lr=LR, n_epochs=N_EPOCHS)
run_with_diagnostics("feature_adapter", lr=LR, n_epochs=N_EPOCHS)
