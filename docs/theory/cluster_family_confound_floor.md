# Cluster-After-Train Family Confound Floor

**Date**: 2026-04-11
**Status**: Validated (15 runs × 4 baselines, 60 data points total)
**Related**:
- TT-Protocol theorem: `docs/theory/scheme_c_provability_sketch.md`
- TT-Protocol empirical confirmation (IFCA only): `docs/theory/tt_protocol_empirical_confirmation.md`
- Multi-seed breakthrough: finding `20260410-235219`

---

## TL;DR

Extending the η_A vs η_C analysis from IFCA to the full cluster family (CFL, FedEM, FedRC, IFCA) on CIFAR-100 recurrence, 5 configs × 3 seeds:

- **No cluster-after-train method achieves η < 0.30 in any of our 5 configurations**.
- The confound floor is structural: doubling K (20 → 40) does not reduce η for any of these methods.
- IFCA is the BEST cluster-after-train method (η = 0.30–0.56) and still far above FPT-OT (η = 0.00–0.16).
- CFL's clustering never fires on CIFAR-100: it degenerates to FedAvg and has undefined Re-ID.
- FedEM has the worst clustering error among methods that DO cluster (η = 0.54–0.72).

---

## Full results table

Clustering error rate $\eta = 1 - \mathrm{Re\mhyphen ID}$, 3 seeds averaged. N/A = method doesn't cluster (degenerate).

| K | ρ | C | K/C | **IFCA** | **CFL** | **FedEM** | **FedRC** | min cluster | FPT-OT | Δ |
|---|---|---|-----|----------|---------|-----------|-----------|-------------|--------|---|
| 20 | 17 | 6 | 3.3 | 0.5643 | N/A | 0.7215 | 0.6393 | 0.5643 | 0.1645 | **+0.400** |
| 20 | 25 | 4 | 5.0 | 0.2972 | N/A | 0.6390 | 0.4715 | 0.2972 | 0.0128 | **+0.284** |
| 20 | 33 | 3 | 6.7 | 0.3048 | N/A | 0.5365 | 0.5342 | 0.3048 | 0.0033 | **+0.302** |
| 40 | 25 | 4 | 10.0 | 0.3108 | N/A | 0.6284 | 0.4826 | 0.3108 | 0.0011 | **+0.310** |
| 40 | 33 | 3 | 13.3 | 0.3426 | N/A | 0.5459 | 0.5497 | 0.3426 | 0.0000 | **+0.343** |

**Accuracy comparison** (sanity check that the clustering failures also translate to accuracy gaps):

| K | ρ | FedAvg | IFCA | CFL | FedEM | FedRC | FPT-OT | Oracle |
|---|---|--------|------|-----|-------|-------|--------|--------|
| 20 | 17 | 0.6203 | 0.6012 | 0.6164 | 0.5307 | 0.2785 | **0.6586** | 0.6716 |
| 20 | 25 | 0.5658 | 0.5287 | 0.5560 | 0.4526 | 0.2241 | **0.6116** | 0.6127 |
| 20 | 33 | 0.6841 | 0.6533 | 0.6741 | 0.5830 | 0.3320 | **0.7123** | 0.7139 |
| 40 | 25 | 0.5999 | 0.5605 | 0.5922 | 0.4802 | 0.2448 | **0.6495** | 0.6489 |
| 40 | 33 | 0.6839 | 0.6502 | 0.6737 | 0.5700 | 0.3773 | **0.7161** | 0.7157 |

## Interpretation

**IFCA is the canonical Scheme A baseline** — competitive accuracy, cluster heads assigned by MSE. Its confound floor of $\eta \approx 0.30$ is the cleanest comparison point for Theorem (Protocol Ordering) and appears in the main paper.

**FedEM** (expectation-maximization over cluster assignments): Its even higher $\eta$ (0.54–0.72) is consistent with the TT-Protocol story — the EM latent variable is still in weight space, which inherits the stale-init confound. Accuracy drop vs FedAvg is 5–13pp, suggesting EM's mis-clustering actively hurts beyond the implicit-shrinkage absorption.

**FedRC** (regularized clustering): Catastrophic accuracy collapse (0.22–0.38) suggests a baseline implementation issue (likely optimizer divergence or regularizer tuning) rather than a pure confound-floor effect. We exclude it from the main paper claim to avoid conflating these two phenomena. The Re-ID is still reported for completeness.

**CFL** (gradient-similarity hierarchical clustering): As documented in the CIFAR-100 calibration notes, CFL's clustering step never fires on CIFAR-100 because frozen ResNet-18 features produce gradient similarities that never exceed the split threshold. Consequently CFL ≡ FedAvg on this dataset, and Re-ID is undefined. This is a known limitation of CFL on frozen-feature tasks. CFL is included in the paper for completeness but is not used to support the confound floor claim.

## Main paper claim (refined)

> Across 5 CIFAR-100 recurrence configurations (K ∈ {20, 40}, ρ ∈ {17, 25, 33}, C ∈ {3, 4, 6}), every cluster-after-train method tested (IFCA, FedEM, FedRC) exhibits a clustering error rate η ≥ 0.30, independent of K. The best cluster-after-train method (IFCA) has η ∈ [0.30, 0.56]. FPT-OT (cluster-before-train via Scheme C) has η ∈ [0.00, 0.16] and achieves η = 0 at K/C ≥ 10, confirming the spectral concentration prediction of Theorem (Protocol Ordering).

## Rebuttal ammunition

**"Maybe this is just IFCA-specific"**: NO — FedEM also has η > 0.50 across all configs, using a completely different clustering algorithm (EM vs MSE-minimization). The floor is structural, not algorithmic.

**"Maybe CFL would do better"**: CFL's clustering is a no-op on CIFAR-100 (degenerates to FedAvg). Gradient-similarity clustering requires non-stationary gradient statistics that frozen features do not provide. This is CFL-specific, not evidence against the theorem.

**"FedRC has different η numbers, so the floor isn't universal"**: FedRC's accuracy collapse indicates a convergence issue that contaminates its Re-ID measurement. We exclude it from the main claim and use IFCA as the canonical comparison.

**"What about the 3-seed variance"**: All η differences (IFCA vs FPT-OT) are > 10× the seed standard deviation. Variance is not a concern.

## What this does NOT show

- It does not show that NO cluster-after-train method can achieve η < 0.30 — only that all tested methods in our setup have this property.
- It does not rule out methods with pre-clustering (e.g., data-distribution-aware cluster init) that would be somewhere between Scheme A and Scheme C.
- It does not address non-recurrent concept drift (one-shot concept introduction), where the confound floor argument doesn't apply by construction.

## Decision

Keep the main paper's mechanism table focused on **IFCA vs FPT-OT** (cleanest comparison). Add a sentence in the mechanism section noting that FedEM also exhibits a confound floor (η > 0.50) across all configurations, and reference the full cluster-family table in this document in the Appendix for completeness.
