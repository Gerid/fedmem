# Prediction Bridge Design: Translating Concept Tracking into Accuracy

> Task #3 design document for FedProTrack NeurIPS 2026
> Date: 2026-03-20

---

## Problem Statement

FedProTrack achieves 2-3x better concept re-identification (re-ID) than IFCA on CIFAR-100 recurrence benchmarks, proving that its Gibbs posterior + memory bank machinery correctly tracks latent concept identities. However, this tracking advantage does **not** translate into prediction accuracy gains. CFL (which ignores concept identity entirely) beats all concept-aware methods.

Root cause analysis identified three structural issues:

1. **Label-homogeneous concepts**: The original CIFAR-100 recurrence dataset has all concepts sharing the same 20 coarse-class label distribution. Concepts only differ by visual style (grayscale, blur, hue). Concept-aware aggregation provides zero benefit when the optimal classifier is the same for all concepts.

2. **Singleton dominance**: ~55% of (client, timestep) positions are singletons -- a concept with exactly 1 client. For singletons, concept-aware aggregation degenerates to LocalOnly (no aggregation partner exists). The Oracle baseline (perfect concept knowledge) achieves nearly identical accuracy to CFL because aggregation within concept groups provides negligible benefit when most groups have size 1.

3. **No warm-start on recurrence**: While FedProTrack stores models in its memory bank (line 1343 of `two_phase_protocol.py`) and restores them on concept switches (lines 2061-2079 of `fedprotrack_runner.py`), this warm-start only fires when a Phase A re-assignment changes a client's concept ID. The benefit is limited when (a) concepts share labels, and (b) T is small so recurrence opportunities are few.

The `label_split` modes ("disjoint", "overlap") being added to `CIFAR100RecurrenceConfig` will fix issue #1. But issues #2 and #3 require mechanism-level changes. Below are three concrete, testable mechanisms.

---

## Mechanism 1: Memory-Assisted Warm-Start with Label-Split Concepts

### Hypothesis

**H_bridge_1**: When concepts have disjoint or partially-overlapping label distributions, FedProTrack's memory-bank warm-start on concept recurrence will yield >5 percentage points higher accuracy than methods without memory (FedAvg, CFL, IFCA), specifically at recurrence points (timesteps where a client returns to a previously-seen concept).

### Rationale

FedProTrack already implements concept-model warm-start (runner lines 2055-2079): when Phase A re-assigns a client from concept A to concept B, and concept B has a stored model in the memory bank, that model is loaded into the client. With label-split concepts, concept B's stored model will have a classifier tuned to concept B's label subset, giving an immediate accuracy advantage over:
- FedAvg: which maintains a single global model averaging over all label subsets
- CFL: which clusters by gradient similarity but has no concept memory
- IFCA: which maintains K fixed cluster models but does not explicitly store/retrieve per-concept snapshots across recurrences

The warm-start acts as a "free" adaptation step: the client begins predicting with a classifier that already knows concept B's label distribution, rather than starting from scratch or from a mismatched global model.

### What Already Works (No Code Changes Needed)

The mechanism is already implemented:
- `DynamicMemoryBank.store_model_params()` stores per-concept aggregated models after Phase B (line 1343-1344 of `two_phase_protocol.py`)
- `DynamicMemoryBank.get_model_params()` retrieves them (line 2070 of runner)
- On concept switch, the runner restores the stored model (lines 2061-2079)

### What Needs Validation

The warm-start's benefit has never been measured because the old dataset had label-homogeneous concepts. With `label_split="disjoint"`, the stored model for concept B will have a fundamentally different classifier head than concept A, making warm-start critical.

### Experiment Design

```
Script: run_cifar100_all_baselines_smoke.py (or a new minimal script)
Dataset: CIFAR100RecurrenceConfig(label_split="disjoint", K=6, T=20, n_samples=400)
Methods: FedProTrack, FedProTrack-NoMemory*, Oracle, FedAvg, IFCA, CFL
Seeds: 5
Key metrics:
  - overall_accuracy (mean over all K*T)
  - recurrence_accuracy (accuracy only at timesteps where concept recurs)
  - accuracy_at_switch (accuracy in the first step after a concept switch)
Budget: matched total bytes
```

*FedProTrack-NoMemory is an ablation that disables lines 2061-2079 (skip warm-start on switch). This can be achieved by setting `dormant_recall=False` and commenting out the warm-start block, or more cleanly by adding a `memory_warm_start: bool = True` flag to `FedProTrackRunner`.

### Validation Criteria

- **Support if**: FedProTrack with warm-start exceeds FedProTrack-NoMemory by >3pp on recurrence_accuracy, AND exceeds CFL by >2pp on overall_accuracy
- **Refute if**: warm-start provides <1pp improvement, or CFL still dominates

### Predicted Outcome

With disjoint labels (each concept gets 4 of 20 coarse classes when 5 concepts exist), a client switching from concept A to concept B faces a completely different classification problem. Without warm-start, the client's model will predict B's labels with random performance (~25% for 4-class). With warm-start, it immediately uses B's stored model (~60-70% accuracy). This should produce a dramatic improvement at switch points.

---

## Mechanism 2: Singleton-Aware Aggregation with Global Fallback

### Hypothesis

**H_bridge_2**: When concept groups contain only 1 client (singletons), falling back to cross-concept aggregation (blending with the global average or the nearest concept's model) will improve accuracy by >3 percentage points compared to pure concept-segregated aggregation, because singletons get no benefit from within-concept averaging.

### Rationale

The current Phase B aggregation (lines 1270-1336 of `two_phase_protocol.py`) averages client models within each concept group. When a concept group has size 1, the "aggregation" is a no-op: the client uploads its model and receives it back unchanged (minus communication cost). This is strictly worse than LocalOnly because it pays communication overhead for zero benefit.

With label-split concepts, pure global FedAvg is harmful (averaging classifiers for disjoint label sets destroys both). But a **selective** fallback strategy can help: when a concept group is singleton, blend the client's model with models from the most similar concept (measured by fingerprint similarity in the memory bank), weighted by similarity. This provides regularization from related concepts without the destructive interference of unrelated ones.

### Code Changes Required

**File**: `fedprotrack/posterior/two_phase_protocol.py`, method `phase_b()` (and `phase_b_soft()`)

Add logic after line 1270 (the concept group loop):

```python
for concept_id, client_ids in concept_groups.items():
    group_params = [client_params[cid] for cid in client_ids if cid in client_params]
    if not group_params:
        continue

    # NEW: Singleton fallback
    if len(group_params) == 1 and self.config.singleton_fallback:
        # Find most similar concept in memory bank
        current_fp = self.memory_bank.get_fingerprint(concept_id)
        if current_fp is not None:
            best_sim, best_cid = 0.0, None
            for other_cid in self.memory_bank.concept_library:
                if other_cid == concept_id:
                    continue
                other_fp = self.memory_bank.get_fingerprint(other_cid)
                if other_fp is not None:
                    sim = current_fp.similarity(other_fp)
                    if sim > best_sim:
                        best_sim, best_cid = sim, other_cid
            if best_cid is not None and best_sim > self.config.singleton_sim_threshold:
                other_model = self.memory_bank.get_model_params(best_cid)
                if other_model is not None:
                    # Blend: (1-alpha)*local + alpha*similar_concept
                    alpha = best_sim * self.config.singleton_blend_alpha
                    blended = {
                        k: (1-alpha)*group_params[0][k] + alpha*other_model[k]
                        for k in group_params[0]
                    }
                    group_params = [blended]
    # ... rest of aggregation
```

**Additional file**: `fedprotrack/posterior/two_phase_protocol.py`, class `TwoPhaseConfig`

Add three new fields:
- `singleton_fallback: bool = False`
- `singleton_sim_threshold: float = 0.5`
- `singleton_blend_alpha: float = 0.3`

### Experiment Design

```
Script: new minimal script or modification of run_cifar100_all_baselines_smoke.py
Dataset: CIFAR100RecurrenceConfig(label_split="overlap", n_classes_per_concept=10, K=6, T=20)
Methods:
  - FedProTrack (current, no singleton handling)
  - FedProTrack-SingletonFallback (with mechanism enabled)
  - Oracle
  - IFCA
  - CFL
  - FedAvg
Seeds: 5
Key metrics:
  - overall_accuracy
  - singleton_accuracy (accuracy at positions where concept group size = 1)
  - non_singleton_accuracy (accuracy at positions where group size >= 2)
  - singleton_group_ratio (from concept_metrics.py -- already implemented)
Budget: matched total bytes
```

### Validation Criteria

- **Support if**: FedProTrack-SingletonFallback improves singleton_accuracy by >3pp over FedProTrack, without degrading non_singleton_accuracy by >1pp
- **Refute if**: singleton_accuracy improvement <1pp, or non_singleton_accuracy degrades >2pp

### Predicted Outcome

With `label_split="overlap"` and `n_classes_per_concept=10`, neighboring concepts share ~50% of their classes. Singleton clients currently get zero aggregation benefit. By blending with the most similar concept's model (which shares half the label space), the singleton gets regularization on shared labels while retaining local specialization on unique labels. The improvement should be concentrated at singleton positions (55% of all K*T cells), producing ~2-4pp overall improvement.

### Risk: Misattribution

If concept similarity estimation is noisy, the fallback could blend with a dissimilar concept and hurt accuracy. The `singleton_sim_threshold` guards against this: only blend when similarity exceeds 0.5. This is exactly the kind of "memory helps vs. harms" question that FedProTrack's theoretical framing targets.

---

## Mechanism 3: Cross-Concept Knowledge Transfer via Concept Similarity Graph

### Hypothesis

**H_bridge_3**: Using concept identity to build a concept similarity graph (from fingerprint or model similarity in the memory bank) and performing similarity-weighted cross-concept model blending during Phase B download will improve accuracy by >2pp over hard concept-segregated aggregation, especially in the "overlap" label-split setting where related concepts share label subsets.

### Rationale

Current Phase B is binary: clients assigned to concept C receive only concept C's aggregated model. This is optimal when concepts are completely disjoint (no shared structure). But in realistic settings, concepts overlap:
- With `label_split="overlap"`, neighboring concepts share a sliding window of classes
- Even with `label_split="disjoint"`, PCA-compressed features carry shared structure from the ResNet backbone

A similarity graph enables **selective knowledge transfer**: concept C's model is blended with concept D's model proportional to their similarity. This is different from Mechanism 2 (which only handles singletons) -- Mechanism 3 applies to ALL clients, providing a softer form of aggregation that respects concept topology.

### Code Changes Required

**File**: `fedprotrack/posterior/fedprotrack_runner.py`, in the Phase B distribution section (lines 2178-2261)

After Phase B aggregation produces `b_result.aggregated_params`, add a cross-concept blending step before distributing to clients:

```python
# NEW: Cross-concept blending during download
if self.cross_concept_blend_alpha > 0.0:
    concept_similarities = {}
    library = protocol.memory_bank.concept_library
    concept_ids = list(b_result.aggregated_params.keys())
    for i, ci in enumerate(concept_ids):
        fp_i = library.get(ci)
        if fp_i is None:
            continue
        for cj in concept_ids:
            if ci == cj:
                continue
            fp_j = library.get(cj)
            if fp_j is not None:
                concept_similarities[(ci, cj)] = fp_i.similarity(fp_j)

    blended_params = {}
    for cid in concept_ids:
        base = b_result.aggregated_params[cid]
        # Weight contributions from other concepts by similarity
        blend_total = {k: arr.copy() * 1.0 for k, arr in base.items()}
        weight_sum = 1.0
        for other_cid in concept_ids:
            if other_cid == cid:
                continue
            sim = concept_similarities.get((cid, other_cid), 0.0)
            if sim > self.cross_concept_sim_threshold:
                w = sim * self.cross_concept_blend_alpha
                other = b_result.aggregated_params[other_cid]
                for k in blend_total:
                    blend_total[k] += w * other[k]
                weight_sum += w
        blended_params[cid] = {
            k: arr / weight_sum for k, arr in blend_total.items()
        }
    # Replace aggregated_params with blended versions
    for cid in blended_params:
        b_result.aggregated_params[cid] = blended_params[cid]
```

**File**: `fedprotrack/posterior/fedprotrack_runner.py`, class `FedProTrackRunner.__init__`

Add:
- `cross_concept_blend_alpha: float = 0.0`  (0.0 = disabled, matches current behavior)
- `cross_concept_sim_threshold: float = 0.3`

### Experiment Design

```
Script: new minimal script
Dataset: CIFAR100RecurrenceConfig(label_split="overlap", n_classes_per_concept=10, K=6, T=20)
Ablation grid:
  - cross_concept_blend_alpha in [0.0, 0.1, 0.2, 0.3, 0.5]
  - cross_concept_sim_threshold in [0.2, 0.4, 0.6]
Methods:
  - FedProTrack (alpha=0.0, current behavior)
  - FedProTrack-CrossBlend (alpha > 0)
  - Oracle
  - IFCA
  - CFL
Seeds: 5
Key metrics:
  - overall_accuracy
  - per_concept_accuracy (to detect if blending helps some concepts and hurts others)
  - re-ID accuracy (blending should not degrade concept tracking)
Budget: matched total bytes (blending adds no communication cost)
```

### Validation Criteria

- **Support if**: best (alpha, threshold) combination exceeds alpha=0.0 by >2pp overall_accuracy, with re-ID degradation <0.02
- **Refute if**: no (alpha, threshold) combination improves by >1pp, or re-ID degrades by >0.05

### Predicted Outcome

With `n_classes_per_concept=10` and 5 concepts over 20 classes, neighboring concepts share 50% of classes. Cross-concept blending effectively doubles the training data for shared-class classifier rows while leaving unique-class rows to local specialization. This should show:
- Monotonically increasing benefit as n_classes_per_concept increases (more overlap = more sharing value)
- Diminishing benefit beyond alpha ~0.3 (too much blending dilutes specialization)
- Strongest effect when combined with Mechanism 1 (warm-start provides a good base model, blending refines it)

### Key Advantage Over Existing Approaches

This mechanism is **zero communication cost** -- the blending happens server-side during model distribution, using already-computed Phase B aggregates and already-stored fingerprint similarities. No additional upload or download is needed. This makes it particularly attractive for the budget-matched comparison framework.

---

## Interaction Between Mechanisms

The three mechanisms are **complementary and can be composed**:

| Mechanism | When it helps | Communication cost |
|-----------|--------------|-------------------|
| M1: Memory warm-start | At concept switch points (recurrence) | Zero (already implemented) |
| M2: Singleton fallback | At singleton positions (55% of cells) | Zero (server-side blending) |
| M3: Cross-concept transfer | At all positions (soft knowledge sharing) | Zero (server-side blending) |

Recommended validation order:
1. **First**: Test M1 alone with `label_split="disjoint"` -- this is the lowest-risk test because the mechanism already exists in code, only the dataset changes
2. **Second**: Test M2 alone with `label_split="overlap"` -- requires small code changes
3. **Third**: Test M3 alone with `label_split="overlap"` -- requires moderate code changes
4. **Fourth**: Test M1+M2+M3 combined -- the full prediction bridge

---

## Pre-Experiment Checklist

Before running any experiments, verify:

1. **label_split works correctly**: Run a quick sanity check that `_concept_class_subsets()` produces expected partitions for 5 concepts with both "disjoint" and "overlap" modes
2. **Feature cache invalidation**: The cache file naming includes `_ls{split_tag}` (line 277-287 of `cifar100_recurrence.py`), so different label_split modes produce separate caches. Verify this.
3. **n_classes propagation**: With disjoint splits (4 classes per concept), each concept pool has different labels. The `_infer_n_classes()` function infers from the dataset globally, but individual concept models need to handle the label subset correctly. Verify that `TorchLinearClassifier(n_classes=20)` still works when a concept only presents 4 classes (it should -- unused output neurons just stay at initialization).
4. **Concept matrix generation**: Verify that `generate_concept_matrix()` with the existing (rho, alpha, delta) settings produces sufficient recurrence events in T=20 for M1 to have effect. At least 30% of K*T positions should be recurrence points.
5. **Singleton ratio**: Check that the singleton ratio with K=6 and T=20 is still ~55%. If the concept matrix has fewer concepts, singletons may decrease, reducing M2's impact.

---

## Expected Paper Impact

If validated, these mechanisms directly address the paper's core claim: **concept identity is useful for prediction, not just for tracking**. The narrative becomes:

1. FedProTrack tracks concepts correctly (re-ID 2-3x better than IFCA) [already established]
2. Correct tracking enables three prediction-boosting mechanisms:
   - **Memory warm-start**: instant adaptation on concept recurrence (M1)
   - **Singleton-aware aggregation**: avoiding degenerate no-op aggregation (M2)
   - **Cross-concept transfer**: leveraging concept topology for knowledge sharing (M3)
3. These mechanisms are communication-free (zero additional bytes), leveraging FedProTrack's existing memory bank infrastructure
4. The benefit scales with concept heterogeneity (label_split controls this experimentally)

This positions FedProTrack as a complete system: concept tracking is the *enabler*, and the three bridge mechanisms are the *payoff*. The label_split parameter provides a clean experimental dial for showing when concept-awareness matters (high heterogeneity) vs. when it is neutral (homogeneous labels).

---

## Files Referenced

- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\posterior\fedprotrack_runner.py` -- main runner, warm-start logic at lines 2055-2079, Phase B distribution at lines 2178-2261
- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\posterior\two_phase_protocol.py` -- Phase B aggregation at lines 1221-1350, model storage at line 1343
- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\posterior\memory_bank.py` -- memory bank with store/retrieve model params, fingerprint similarity
- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\real_data\cifar100_recurrence.py` -- CIFAR-100 dataset with label_split support, class subset logic at lines 120-168
- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\experiment\baselines.py` -- Oracle baseline at lines 203-307
- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\baselines\runners.py` -- CFL (310-376), IFCA (684-753) runners
- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\metrics\concept_metrics.py` -- singleton_group_ratio at line 199
- `E:\fedprotrack\.claude\worktrees\elegant-poitras\fedprotrack\federation\aggregator.py` -- FedAvg and ConceptAwareFedAvg aggregators
