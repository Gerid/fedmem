# Three-Fix Ablation: Only Fix 1a Matters

**Date**: 2026-04-11
**Status**: Validated (2 configs × 3 seeds × 4 conditions = 24 runs)
**Related**:
- Multi-seed breakthrough: finding `20260410-235219`
- TT-Protocol provability: finding `20260411-023058`
- TT-Protocol empirical confirmation: finding `20260411-023407`
- Cluster-family confound floor: finding `20260411-030623`

---

## TL;DR

**The narrative of "three cascading fixes" that produced the OT breakthrough is wrong.** It is actually **one critical fix + one minor fix + one dead fix**:

| Fix | Description | Effect at K/C=5 | Effect at K/C=13.3 | Status |
|-----|-------------|------------------|---------------------|--------|
| **1a** | last_significant eigengap heuristic | −98 pp gap closed | −101 pp gap closed | **DECISIVE** |
| 1b | k-NN local bandwidth | −8 pp gap closed | 0 pp (no effect) | minor |
| 2 | DRCT SVD participation-ratio d_eff | 0 pp (bit-identical) | 0 pp (bit-identical) | **DEAD** |

Fix 1a alone accounts for the entire breakthrough. Fix 2 (DRCT shrinkage) is inactive at the current implementation and contributes nothing in practice. This is a surprising negative result that must be reflected honestly in the paper.

## Setup

- Dataset: CIFAR-100 recurrence (frozen ResNet-18 features)
- Configs: K=20 / ρ=25 (C=4, K/C=5.0) and K=40 / ρ=33 (C=3, K/C=13.3)
- Methods: FPT-OT baseline (all fixes ON) + 3 ablation variants (each with exactly one fix disabled)
- Seeds: 42, 43, 44 (3 per condition)
- Budget: T=100 rounds, n_epochs=5, lr=0.05, n=400 samples/client/round
- Ablation flags introduced (all default to current post-fix behaviour, so adding them is backward-compatible):
  - `ot_affinity_scale: "local" | "median"` (default `local` = fix 1b on)
  - `ot_eigengap_method: "last_significant" | "argmax"` (default `last_significant` = fix 1a on)
  - `drct_force_ambient_d_eff: bool` (default `False` = fix 2 on)

## Results

### Config 1: K=20, ρ=25, C=4, K/C=5.0

| Condition | Acc | Re-ID | Gap closed | Δ acc |
|-----------|-----|-------|------------|-------|
| Baseline (all 3 fixes ON) | 0.6116 ± 0.0339 | 0.9872 ± 0.0051 | **97.7%** | — |
| minus fix 1a (argmax eigengap) | 0.5657 ± 0.0374 | 0.3655 ± 0.0457 | **−0.2%** | −0.0459 |
| minus fix 1b (median bandwidth) | 0.6077 ± 0.0307 | 0.9015 ± 0.0168 | **89.5%** | −0.0038 |
| minus fix 2 (ambient d_eff) | 0.6116 ± 0.0339 | 0.9872 ± 0.0051 | **97.7%** | **+0.0000** |

Reference: FedAvg = 0.5658, Oracle = 0.6127.

### Config 2: K=40, ρ=33, C=3, K/C=13.3

| Condition | Acc | Re-ID | Gap closed | Δ acc |
|-----------|-----|-------|------------|-------|
| Baseline (all 3 fixes ON) | 0.7161 ± 0.0171 | 1.0000 ± 0.0000 | **101.3%** | — |
| minus fix 1a (argmax eigengap) | 0.6840 ± 0.0215 | 0.4472 ± 0.0393 | **0.4%** | −0.0321 |
| minus fix 1b (median bandwidth) | 0.7161 ± 0.0171 | 0.9987 ± 0.0010 | **101.4%** | **+0.0000** |
| minus fix 2 (ambient d_eff) | 0.7161 ± 0.0171 | 1.0000 ± 0.0000 | **101.3%** | **+0.0000** |

Reference: FedAvg = 0.6839, Oracle = 0.7157.

## Analysis

### Fix 1a is the entire breakthrough

Turning off the last_significant-gap heuristic (reverting to classical argmax-gap) **completely collapses** FPT-OT's performance:
- Re-ID drops from 0.99 to 0.37 (K/C=5.0) and from 1.00 to 0.45 (K/C=13.3).
- Gap closed drops from 97.7% to −0.2% and from 101.3% to 0.4%.
- Accuracy drops ~3–5 percentage points, landing essentially back at FedAvg.

The argmax-gap heuristic is mis-estimating the number of concepts. With ρ=25 the ground truth C=4, but argmax-gap tends to pick C=2 or C=3 (under-counting), because the eigenvalue gap is dominated by the transition from the first structural eigenvalue to the rest, not by the last-to-plateau transition. The last_significant-gap heuristic explicitly looks for the transition from structural eigenvalues to the plateau, which is robust to heterogeneous concept separation.

**Paper implication**: The eigengap heuristic is the single key algorithmic insight. Everything else is supporting infrastructure.

### Fix 1b is a small rescue at low K/C

Replacing the k-NN local bandwidth with the median bandwidth costs ~8 percentage points of gap closed at K/C=5.0 (97.7% → 89.5%) but has no measurable effect at K/C=13.3 (101.3% → 101.4%, within noise). This matches the intuition: at low K/C the concept clusters are small and their dispersions vary more, so a per-point adaptive bandwidth is needed; at high K/C the overall structure is clean enough that even a single median bandwidth works.

**Paper implication**: Fix 1b is a minor robustness fix that matters only near the K/C=5 boundary. Mention it in the appendix, not the main claim.

### Fix 2 is dead

Forcing the DRCT shrinkage to use the ambient `d_eff = d × 0.9` (instead of the SVD participation-ratio estimate) produces **bit-identical** results in both configurations — same accuracy, same Re-ID, same per-seed values. This means either (a) the DRCT code path is not being exercised (e.g., too few concept groups per round for DRCT to fire), or (b) the DRCT shrinkage is being computed but the resulting λ is effectively zero regardless of d_eff, because sigma_B² dominates.

Either way, **DRCT as currently implemented contributes nothing to FPT-OT's breakthrough**. The earlier narrative that "DRCT fixing d_eff was critical to unlock Oracle-level performance" is wrong.

Possible explanations (not verified, out of scope for this ablation):
1. `drct_min_concepts=2` is met, but `shrink_every=6` means DRCT only fires every 6 rounds, and on those rounds the shrinkage amount is negligible.
2. With Scheme C early OT, the concept assignments are so clean that within-concept dispersion is tiny and σ² → 0, making `variance_term / (variance_term + σ_B²)` → 0 regardless of d_eff.
3. The aggregation path in OT mode bypasses the DRCT shrinkage entirely in some code path I have not audited.

**Paper implication**: Either (a) remove the DRCT shrinkage claim entirely, (b) investigate and fix the dead code, or (c) keep it but explicitly note that ablation shows no measurable contribution in the current setting. Option (a) is the cleanest.

## Updated cascading fixes narrative

**Before (incorrect)**: "Three cascading fixes: last_significant eigengap + k-NN local bandwidth + SVD participation-ratio d_eff were all necessary for the OT breakthrough."

**After (validated)**: "The OT breakthrough is driven by a single algorithmic fix: the last_significant eigengap heuristic for concept count estimation (Fix 1a). The k-NN local bandwidth (Fix 1b) is a minor robustness improvement that adds 8 percentage points of gap closed near the K/C=5 boundary but has no effect beyond it. The DRCT SVD d_eff (Fix 2) has no measurable effect in the current implementation — it is either inactive or computes a shrinkage coefficient that is negligibly small."

## Paper table for ablation section

```
Ablation                           | K=20, rho=25 (K/C=5) | K=40, rho=33 (K/C=13.3)
                                   | Acc    Re-ID         | Acc    Re-ID
Baseline (all fixes on)            | 0.612  0.987         | 0.716  1.000
minus eigengap last_significant    | 0.566  0.366         | 0.684  0.447
minus k-NN local bandwidth         | 0.608  0.902         | 0.716  0.999
minus SVD d_eff                    | 0.612  0.987         | 0.716  1.000
```

**Paper interpretation to include:** Only the eigengap heuristic matters; the other two are either marginal or inactive in our setting.

## Risk: reviewer attack on the narrative

A careful reviewer will ask: "You introduced three fixes but the ablation shows only one of them matters. Why did you introduce the other two?" The answer must be honest:

1. The fixes were introduced iteratively during debugging, each motivated by a specific failure mode we observed at the time. Only after the full pipeline was working did we have the budget to run a proper ablation.
2. The eigengap heuristic is the only one that matters for the breakthrough. The other two are in the code but the paper should not claim they are necessary.
3. We flag this openly in the paper as a minor methodological lesson: ablation studies should be run early, and it is surprisingly easy for an implementation to carry "dead" components that contribute nothing.

## Decision

- **Rewrite the addendum** to lead with "Fix 1a is the critical algorithmic fix" and demote Fix 1b and Fix 2 to supporting roles.
- **Update `ot_concept_discovery.py` docstring** to explicitly call out that the last_significant-gap heuristic is the critical choice.
- **Consider removing DRCT from the OT mode entirely** in a future cleanup, or investigating why it is inactive and fixing it. Not urgent for the paper.
- **Save this ablation as a validated finding** in the ledger.
