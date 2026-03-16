# Phase 3 SINE Gate Readout

**Date:** 2026-03-17
**Dataset:** SINE (canonical synthetic)
**Seeds:** 42, 123, 456, 789, 1024 (5 seeds)
**Grid:** 75 settings per seed = 375 total per method
**Methods:** 10 (FedProTrack, IFCA, FedDrift, TrackedSummary, Flash, FedAvg, CompressedFedAvg, FedProto, LocalOnly, Oracle)
**Results directory:** `E:\fedprotrack\results_phase3_v2_sine`

---

## 1. Artifact Completeness

**Status: COMPLETE.**

All 10 methods have exactly 375 settings each (confirmed in `claim_check.json`).
Outputs present:

- `summary.json`, `summary.csv` -- per-method aggregate statistics
- `tables/` -- `main_table.tex`, axis-sweep tables (`table_rho.tex`, `table_alpha.tex`, `table_delta.tex`), `appendix/` (overhead, oracle table)
- `figures/` -- phase diagrams (5 alpha slices x 5 comparison heatmaps), budget frontier, ablation bar charts, scalability
- `logs/` -- `claim_check.json`, `budget_points.json`, `alpha_diagnostics.json`, `diagnostic_summary.md`

No missing artifacts.

---

## 2. Gate Verdict

| Gate | Criterion | Result |
|------|-----------|--------|
| **E1 (Identity)** | FedProTrack re-ID >= IFCA re-ID | **FAIL** -- FedProTrack trails by 0.112 |
| **E4 (Budget)** | FedProTrack has >= 1 Pareto non-dominated budget point | **FAIL** -- zero non-dominated points |

**Overall gate: NOT PASSED.** Both identity and budget claims are unsupported on the SINE canonical benchmark.

---

## 3. Key Numbers

### 3a. Main Table (375 settings, 5-seed mean)

| Method | Re-ID Acc | Final Acc | AUC | Mean Rank | Win Rate |
|--------|-----------|-----------|-----|-----------|----------|
| Oracle | 1.000 | 0.646 | 11.80 | 4.79 | 0.293 |
| TrackedSummary | -- | 0.617 | 11.07 | 3.89 | 0.235 |
| Flash | -- | 0.617 | 11.07 | 4.89 | 0.235 |
| IFCA | 0.715 | 0.634 | 11.64 | 3.96 | 0.205 |
| FedDrift | 0.518 | 0.634 | 11.65 | 4.22 | 0.203 |
| FedProTrack | 0.604 | 0.616 | 11.34 | 5.87 | 0.035 |
| LocalOnly | -- | 0.606 | 11.03 | 6.83 | 0.000 |
| FedAvg | -- | 0.596 | 10.89 | 6.59 | 0.032 |
| CompressedFedAvg | -- | 0.570 | 10.53 | 7.26 | 0.045 |
| FedProto | -- | 0.555 | 10.47 | 6.70 | 0.056 |

**Note on "--" entries:** Methods without a concept identity tracking mechanism (LocalOnly, FedAvg, CompressedFedAvg, FedProto, TrackedSummary, Flash) do not produce re-ID metrics. These nulls are **by design** -- only FedProTrack, IFCA, and FedDrift maintain explicit concept assignments that can be evaluated against ground truth. This is not a data-collection bug.

### 3b. Identity-Capable Methods (head-to-head)

| Method | Re-ID Acc | Wrong-Memory Rate | Assignment Entropy |
|--------|-----------|--------------------|--------------------|
| IFCA | 0.715 | 0.285 | 0.483 |
| FedProTrack | 0.604 | 0.397 | 0.182 |
| FedDrift | 0.518 | 0.482 | 0.522 |

IFCA dominates re-ID across all alpha slices. FedProTrack's low assignment entropy (0.182) suggests it is making confident but often incorrect assignments.

### 3c. Budget Frontier (federation_every sweep: 1, 2, 5, 10)

| Method | fe=1 bytes | fe=1 AUC | fe=10 bytes | fe=10 AUC |
|--------|-----------|----------|------------|----------|
| FedProTrack | 12,000 | 10.53 | 1,200 | 10.30 |
| IFCA | 9,120 | 10.73 | 480 | 10.74 |
| FedDrift | 4,560 | 10.77 | 240 | 10.76 |
| CompressedFedAvg | 3,800 | 10.06 | 200 | 10.70 |
| FedAvg-Full | 4,560 | 10.31 | 240 | 10.73 |

FedProTrack spends 2.6x the bytes of IFCA (12,000 vs 9,120 at fe=1) for lower AUC (10.53 vs 10.73). At every budget level, another method achieves equal or better AUC at equal or lower cost. FedDrift is the Pareto leader: best AUC at moderate cost.

---

## 4. E1 Failure Analysis (Identity)

FedProTrack's re-ID accuracy (0.604) trails IFCA (0.715) by 11.2 percentage points.

**Likely causes:**

1. **Gibbs posterior overconfidence.** FedProTrack's assignment entropy is the lowest among identity methods (0.182 vs IFCA's 0.483). The soft posterior is collapsing to near-hard assignments without sufficient exploration, leading to sticky misassignments that never self-correct.

2. **Spawn/merge sensitivity.** Ablations show that disabling spawn/merge drops re-ID to 0.380, confirming it is load-bearing. But the current novelty_threshold/merge_threshold may be poorly calibrated for SINE's drift structure -- spawning too late and merging too aggressively.

3. **Two-phase overhead taxes accuracy.** Phase A prototype exchange consumes 60% of FedProTrack's bytes (7,200 of 12,000) but its fingerprint-based concept matching is less accurate than IFCA's loss-based cluster selection. The prototype step adds cost without proportional identity benefit.

4. **Alpha sensitivity.** At alpha=0.0 (synchronous drift), FedProTrack's final accuracy gap vs IFCA is small (0.651 vs 0.663). At alpha=1.0 (fully asynchronous), the gap widens (0.592 vs 0.608). The temporal prior may be miscalibrated for high-asynchrony regimes.

---

## 5. E4 Failure Analysis (Budget)

FedProTrack has zero Pareto non-dominated budget points.

**Root cause:** The two-phase protocol is structurally expensive.

- At fe=1, FedProTrack sends 12,000 bytes (Phase A: 7,200 + Phase B: 4,800). FedDrift achieves higher AUC at 4,560 bytes. IFCA achieves higher AUC at 9,120 bytes.
- Reducing communication frequency (fe=10) cuts FedProTrack to 1,200 bytes, but AUC drops to 10.30 -- worse than FedDrift's 10.76 at only 240 bytes.
- The prototype exchange in Phase A is the dominant cost driver. Without matching accuracy gains, it pushes every FedProTrack budget point below the Pareto frontier.

---

## 6. Strongest Methods in This Package

Ranking by complementary criteria:

1. **FedDrift** -- Best budget efficiency (Pareto dominant), second-best AUC, solid re-ID (0.518). Practical choice.
2. **IFCA** -- Best re-ID accuracy (0.715), close to FedDrift on AUC, moderate cost (9,120 bytes at fe=1). Best identity tracker.
3. **TrackedSummary** -- Best mean rank (3.89), strong final accuracy (0.617), no identity tracking. Best non-identity method.
4. **Oracle** -- Highest final accuracy (0.646) and AUC (11.80), but is an upper-bound reference, not a deployable method.

FedProTrack ranks 6th of 10 by mean rank (5.87) and has the lowest win rate (0.035) among federated methods.

---

## 7. Ablation Highlights

From the module ablation (default SINE setting, single seed):

| Configuration | Re-ID | Wrong-Mem | Entropy |
|---------------|-------|-----------|---------|
| Full FedProTrack | 0.650 | 0.350 | 0.511 |
| No temporal prior | 0.395 | 0.605 | 0.001 |
| Hard assignment (omega=100) | 0.650 | 0.350 | 0.001 |
| No spawn/merge | 0.380 | 0.620 | 0.000 |
| Phase A only (no aggregation) | 0.380 | 0.620 | 0.000 |

**Critical modules:** Temporal prior and spawn/merge are essential (removing either halves re-ID). Post-spawn merge, sticky dampening, and model-loss gate have no measurable effect on this benchmark -- they may be redundant or require harder drift scenarios to activate.

---

## 8. Recommended Next Steps (Diagnostics Only)

These are scoped as diagnostic investigations, not full re-runs.

1. **Posterior temperature sweep.** The low assignment entropy (0.182) suggests omega is too high. Run a 1D sweep of omega in {0.1, 0.5, 1.0, 2.0, 5.0} on the default SINE setting (single seed) and plot re-ID vs entropy. Goal: find the entropy sweet spot where re-ID peaks.

2. **Phase A cost audit.** Log per-round Phase A vs Phase B byte counts. Determine whether prototype exchange can be made sparser (e.g., send prototypes every 2nd or 3rd federation round) without degrading identity accuracy.

3. **Spawn/merge threshold scan.** Sweep novelty_threshold in {0.1, 0.2, 0.3, 0.4} and merge_threshold in {0.5, 0.7, 0.9} on the default setting. The current defaults may be spawning too conservatively for SINE's abrupt drift pattern.

4. **Per-alpha breakdown of re-ID.** The alpha diagnostics show FedProTrack trails IFCA at every alpha level. Check whether the gap is uniform or concentrated in specific (rho, delta) cells. If concentrated, it suggests a structural weakness in specific drift regimes rather than a global calibration issue.

5. **IFCA loss-based selection as oracle comparison.** Instrument IFCA's cluster-selection accuracy per round and compare with FedProTrack's posterior assignment accuracy per round. This will pinpoint whether the gap is in initial assignment, drift detection, or post-drift re-identification.

None of these require algorithm changes or new baselines -- they are analysis scripts over existing logged data or single-setting re-runs.

---

## 9. Summary for PI

The Phase 3 canonical SINE gate has been fully executed: 10 methods, 375 settings each, all artifacts present. **Both gates failed.** FedProTrack's concept re-identification accuracy (0.604) falls 11 points below IFCA (0.715), and its communication cost places it off the Pareto frontier at every budget level. The two-phase protocol's prototype exchange is the primary cost driver without matching accuracy returns.

FedDrift and IFCA are the strongest methods in the current package. FedProTrack's core Gibbs posterior machinery (temporal prior + spawn/merge) is load-bearing per ablations, but appears miscalibrated: assignments are overconfident and the budget overhead is not justified by accuracy gains.

Before investing in a full multi-generator package, the recommended path is targeted diagnostics on posterior temperature (omega), Phase A communication frequency, and spawn/merge thresholds -- all achievable with single-setting scripts and no algorithm redesign.
