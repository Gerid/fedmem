# H5 and Phase Diagram Results -- CORRECTED (cache bug fix)

Date: 2026-03-20
Cache fix: `cifar100_recurrence.py` cache key now includes `n_concepts`.
All previous disjoint-label results used stale 2-class-per-concept caches
(from 10-concept experiments). Now each concept correctly has 5 classes
(20 total classes / 4 concepts = 5 per concept for disjoint split).

## H5: FPT Advantage Experiment (disjoint labels, K=10, T=30, 5 seeds)

| Method      | Final Acc        | Re-ID  | AUC    | Bytes     |
|-------------|------------------|--------|--------|-----------|
| Oracle      | 0.7889 +/- 0.039 | 1.0000 | 21.674 | 2,889,600 |
| CFL         | 0.7875 +/- 0.040 | --     | 20.243 | 2,889,600 |
| FedAvg      | 0.7343 +/- 0.040 | --     | 18.590 | 2,889,600 |
| FedProTrack | 0.6821 +/- 0.034 | 0.7393 | 17.302 | 3,796,296 |
| IFCA        | 0.3303 +/- 0.071 | 0.9387 | 9.636  | 7,224,000 |

### Key observations (H5)

1. **CFL matches Oracle** on accuracy (0.788 vs 0.789) at identical budget.
   CFL's clustering-then-aggregate strategy works well when label splits
   create clear gradient-space clusters.

2. **FedProTrack lags CFL by ~10.5 pp** (0.682 vs 0.788). FedProTrack's
   concept tracking (re-ID = 0.739) is decent but its model accuracy
   underperforms simpler aggregation methods.

3. **FedAvg is surprisingly strong** at 0.734, only 5.4 pp below Oracle.
   This suggests that even naive global averaging partially works because
   with 5 classes per concept and 4 concepts, gradient interference is
   moderate.

4. **IFCA collapses** to 0.330 despite near-perfect re-ID (0.939). The
   issue is IFCA's fixed-K clustering combined with its model selection
   strategy; it achieves correct identity but poor model quality.

5. **FedProTrack spawns 31 concepts** (mean) for a 4-concept ground truth,
   indicating significant over-spawning that dilutes aggregation groups.

---

## Phase Diagram: Accuracy vs Label Heterogeneity (3 seeds)

### Final Accuracy

| Method      | shared | overlapping | disjoint | delta (disj-shared) |
|-------------|--------|-------------|----------|---------------------|
| Oracle      | 0.4844 | 0.6480      | 0.7955   | +0.3111             |
| CFL         | 0.5584 | 0.6468      | 0.7910   | +0.2325             |
| FedAvg      | 0.5067 | 0.5869      | 0.7387   | +0.2321             |
| FedProTrack | 0.4623 | 0.5405      | 0.6760   | +0.2137             |
| IFCA        | 0.2979 | 0.3243      | 0.3435   | +0.0456             |

### Re-ID Accuracy (identity-capable methods)

| Method      | shared | overlapping | disjoint |
|-------------|--------|-------------|----------|
| Oracle      | 1.0000 | 1.0000      | 1.0000   |
| FedProTrack | 0.7311 | 0.7156      | 0.7289   |
| IFCA        | 0.4944 | 0.7378      | 0.9022   |

### Key observations (Phase Diagram)

1. **All methods improve with disjoint labels.** This is counterintuitive:
   the classification task is EASIER with fewer classes per concept (5 vs 20).
   The "heterogeneity" actually makes each sub-task simpler.

2. **Oracle gains most** (+0.311 from shared to disjoint), confirming that
   concept-aware aggregation benefits most from label separation.

3. **CFL tracks Oracle closely** at all heterogeneity levels, showing that
   gradient-based clustering is an effective concept discovery proxy on
   CIFAR-100 features.

4. **FedProTrack's re-ID is stable** across heterogeneity levels (~0.72-0.73),
   meaning its concept tracking quality does not depend on label structure.
   However, this stable tracking does not translate to competitive accuracy.

5. **IFCA's re-ID improves dramatically** with disjoint labels (0.49 -> 0.90)
   because distinct label sets create clear loss-based cluster separation.
   Despite good re-ID, IFCA's accuracy remains poor (~0.30-0.34).

6. **The FPT-CFL gap is consistent** at ~10 pp across all heterogeneity levels.
   This suggests the gap is structural (over-spawning, model quality) rather
   than related to label heterogeneity.

---

## Comparison with stale-cache results

Previous (buggy) disjoint results used 2 classes per concept (from 10-concept
caches). The corrected results with 5 classes per concept show:

- **Higher accuracy across the board**: 5-class classification is easier than
  2-class when the classes are drawn from CIFAR-100 superclasses (more diverse
  training signal, better feature utilization).
- **Same qualitative ranking**: CFL >= Oracle > FedAvg > FedProTrack >> IFCA.
- **FPT-CFL gap unchanged**: The gap is structural, not an artifact of the
  cache bug.

---

## Commands used

```bash
# H5 experiment
cd E:/fedprotrack/.claude/worktrees/elegant-poitras
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 E:/anaconda3/python.exe run_fpt_advantage.py

# Phase Diagram
PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 E:/anaconda3/python.exe run_phase_diagram.py
```

## Result files

- `tmp/fpt_advantage/fpt_advantage_results.csv` -- per-seed H5 results
- `tmp/fpt_advantage/summary.json` -- H5 summary with raw data
- `tmp/phase_diagram/phase_diagram_results.csv` -- per-seed phase diagram results
- `tmp/phase_diagram/summary.json` -- phase diagram summary with raw data
