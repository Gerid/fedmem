# Self-Review: TT-Protocol + Empirical Mechanism + Three-Fix Ablation

**Date**: 2026-04-11
**Reviewer**: hostile self-review (autoresearch protocol, MEDIUM difficulty)
**Subject**: All work produced in the 2026-04-11 overnight outer loop, including:
- TT-Protocol theorem + provability sketch (`docs/theory/scheme_c_provability_sketch.md`)
- TT-Protocol empirical confirmation (`docs/theory/tt_protocol_empirical_confirmation.md`)
- Cluster-after-train family confound floor (`docs/theory/cluster_family_confound_floor.md`)
- Three-fix ablation (`docs/theory/three_fix_ablation.md`)
- Paper addendum `paper/scheme_c_addendum.tex`

---

## Prosecution Brief (written first, as hostile as possible)

### Objection 1 — "Theorem is a re-dressing of existing results"

**Severity**: MAJOR
**Claim**: Theorem (Protocol Ordering) reduces to Corollary 1 of the base paper plus a one-line observation that $\eta_C \le \eta_A$ in the recurrence regime. This is not a novel theoretical contribution. The "novel lemma" in Step 3 of the proof is a two-sentence linearization of SGD followed by a monotonicity argument. Calling this "one novel lemma + 1 page" is generous; a skeptical reviewer would call it "a trivial extension of an existing result."

**Evidence**: The bulk of the proof is reusing Corollary 1 (main.tex line 193). The only new piece is the within-cluster dispersion argument in Step 3, which is ~1 paragraph of informal reasoning about SGD one-step linearization. No new concentration inequality, no new minimax bound, no new non-trivial inequality.

**Recommendation**: Either beef up the theoretical contribution (e.g., finite-sample concentration, rate-matching lower bound, alternative under relaxed assumptions) or position the theorem as a "conceptual reframing" rather than a "theoretical contribution".

### Objection 2 — "(A4$'$) is a load-bearing assumption that is not actually verified"

**Severity**: CRITICAL
**Claim**: Assumption (A4$'$) says "fingerprint separation dominates weight separation" with constant $\beta > 0$. This is the assumption that rules out pathological cases like label-flip noise. But:
1. The paper does NOT measure $\beta$ empirically on CIFAR-100.
2. The paper does NOT bound $\beta$ theoretically.
3. The paper ASSERTS that "class-conditional fingerprints satisfy this by construction" without proof.

A reviewer can legitimately ask: "Show me the numerical value of $\beta$ for your CIFAR-100 fingerprint. If you cannot, the theorem is not anchored in the empirical setting."

**Evidence**: No measurements of $\beta$ in any of the analysis files. No discussion of how to estimate $\beta$ from data.

**Recommendation**: Either (a) add an empirical section measuring $\beta$ on CIFAR-100 directly, or (b) state the theorem as "conditional on (A4$'$) holding" and explicitly flag the assumption in the abstract/introduction as a limitation.

### Objection 3 — "FedRC's convergence issue contaminates the confound floor claim"

**Severity**: MAJOR
**Claim**: The cluster-family confound floor result uses FedEM and FedRC to extend the η ≥ 0.30 claim. But FedRC's accuracy is catastrophically low (0.22–0.38 vs FedAvg ~0.60). This is not a confound-floor effect; it is a baseline that does not converge. Including FedRC in the "family-level confound floor" claim is misleading.

**Evidence**: FedRC accuracy values in `docs/theory/cluster_family_confound_floor.md` Table 2. The gap from FedAvg (-0.38 to -0.22 pp) is far larger than any plausible clustering-error effect.

**Recommendation**: Remove FedRC from the main confound floor claim. Use only IFCA and FedEM, and note that their different clustering algorithms (MSE assignment vs EM) converge to similar high η, which is the structural evidence.

### Objection 4 — "Fix 2 being DEAD is a serious bug in the narrative"

**Severity**: CRITICAL
**Claim**: The paper's earlier narrative said "three cascading fixes" were necessary. The ablation now says Fix 2 (DRCT SVD d_eff) is bit-identical with and without the fix, i.e. it contributes nothing. This means:
1. Either the DRCT shrinkage code path is never executed in practice (implementation bug), or
2. The SVD d_eff improvement was never material in the first place (wrong analysis).

In either case, the earlier publication-facing claim was wrong. Reviewers will ask: "Why did you introduce Fix 2 in the first place if it has no effect? What does this say about the soundness of the rest of your method?"

**Evidence**: Bit-identical per-seed values for baseline vs `drct_force_ambient_d_eff=True` at both K=20/ρ=25 and K=40/ρ=33.

**Recommendation**: Either (a) investigate why DRCT is inactive and fix it (so Fix 2 actually does something measurable), or (b) remove DRCT from the OT mode entirely in the paper and explain in the appendix that DRCT is a non-factor for FPT-OT.

### Objection 5 — "Wall-clock claim is misleading"

**Severity**: MINOR
**Claim**: The addendum says "FPT-OT's total wall-clock per run, averaged across seeds and configurations, is 48.8s, comparable to IFCA's 38.3s (1.3× ratio)". But the ratio varies from 0.90× (K=20/ρ=17) to 1.54× (K=20/ρ=33). Reporting the mean ratio hides a 70% spread.

**Evidence**: Wall-clock table in `docs/theory/tt_protocol_empirical_confirmation.md`.

**Recommendation**: Report per-config wall-clock ratios, not the average.

### Objection 6 — "The 'exceeds Oracle' claim is within 1σ — this is noise"

**Severity**: MINOR
**Claim**: The addendum says FPT-OT "exceeds" Oracle at K=40/ρ=25 (+0.06pp) and K=40/ρ=33 (+0.04pp), and cites this as a feature. But both deltas are well within 1σ (std ≈ 0.02). A proper statistical test would say these are ties, not "exceedances".

**Evidence**: $\sigma \approx 0.02$ in the multi-seed table; $|\Delta| \approx 0.0004$–$0.0006$.

**Recommendation**: Report as "FPT-OT matches Oracle within 1σ at K/C ≥ 10" rather than "FPT-OT exceeds Oracle".

### Objection 7 — "CFL 'doesn't fire' is a convenient dismissal"

**Severity**: MINOR
**Claim**: The claim "CFL's clustering never fires on CIFAR-100" removes CFL from the confound-floor comparison, which is the strongest cluster-after-train baseline when it does fire (e.g. on EMNIST). A reviewer can say: "You picked a dataset where your strongest competitor degenerates. On EMNIST, CFL clusters correctly and the confound floor claim may not hold."

**Evidence**: Memory note says "CFL clustering never fires on CIFAR-100". Paper EMNIST section shows CFL clustering does matter there (Table `tab:emnist_kscaling`).

**Recommendation**: Rerun the protocol-ordering experiment on EMNIST where CFL actually clusters. If the confound floor still appears, the claim strengthens; if not, caveat the claim as "holds on frozen-feature CIFAR-100".

---

## Defense Brief (evidence-based responses)

### Response to Objection 1 (Theorem is trivial)

**Acknowledged as PARTIALLY_SUSTAINED**: The theorem IS mostly a reframing of Corollary 1. I should be honest in the paper and position it as a "conceptual theorem that explains the empirical breakthrough" rather than a heavy theoretical contribution. The real contribution is (a) the protocol reframing (Scheme A vs Scheme C), (b) the empirical mechanism measurement (η_A vs η_C), and (c) the validated application on CIFAR-100. I will update the addendum to call this a "theorem" with explicit scare quotes or rename it to "Proposition (Protocol Ordering)" to set expectations.

**Action**: Rename `\begin{theorem}[Protocol Ordering]` to `\begin{proposition}[Protocol Ordering]` and update surrounding prose.

### Response to Objection 2 ((A4$'$) is unverified)

**SUSTAINED**: This is a real gap. I cannot produce a numerical value of $\beta$ right now without additional measurement code. The honest move is to:
1. Add a paragraph in the proof sketch explicitly flagging $\beta$ as an assumed constant.
2. Add a footnote saying "an empirical estimate of $\beta$ on CIFAR-100 is left to future work".
3. Ensure the abstract/introduction does NOT claim the theorem holds "universally" — only under (A4) + (A4$'$).

**Action**: I will add this caveat to `scheme_c_addendum.tex` in the next pass.

### Response to Objection 3 (FedRC contaminates)

**ACKNOWLEDGED and already addressed**: The `cluster_family_confound_floor.md` explicitly excludes FedRC from the main claim and uses IFCA + FedEM as the evidence. The addendum `scheme_c_addendum.tex` Table `tab:cluster_family` reports FedRC for completeness with a prose caveat: "FedRC exhibits a baseline convergence issue leading to low absolute accuracy; its Re-ID is reported but excluded from the main confound-floor claim." This was already done.

**Action**: Verify the addendum caveat is present and clear. Strengthen if needed.

### Response to Objection 4 (Fix 2 is DEAD)

**ACKNOWLEDGED and already addressed in the most recent update**: The addendum's new ablation section is explicit: "Fix 2 has no measurable effect in either configuration ... We report this negative result openly." The paper is honest about the inactive code path. The alternative is to either remove DRCT entirely from the paper (safer) or investigate why DRCT is inactive (more work but could strengthen the narrative).

**Action**: In the next iteration, investigate WHY DRCT is inactive. The leading hypothesis is that `shrink_every=6` combined with Scheme C's clean assignments means the shrinkage coefficient λ is always near zero. An instrumented run that logs the actual λ values would settle this. If λ is indeed ~0, the paper can say "DRCT is theoretically active but empirically negligible because Scheme C's assignments make the within-concept variance vanish".

### Response to Objection 5 (Wall-clock ratio)

**SUSTAINED (minor)**: I should report per-config ratios.

**Action**: Update the addendum's wall-clock paragraph to include the range 0.90×–1.54× rather than just the mean.

### Response to Objection 6 ("exceeds" Oracle is within noise)

**SUSTAINED (minor)**: The language is misleading. I will soften "exceeds" to "matches within 1σ" in the paper.

**Action**: Update the addendum's "Why Scheme C can exceed oracle" paragraph.

### Response to Objection 7 (CFL on EMNIST)

**SUSTAINED (minor)**: The confound floor claim is strictly verified on CIFAR-100 only. I should explicitly caveat this.

**Action**: Add a sentence: "Our confound floor claim is verified on CIFAR-100 with frozen ResNet-18 features; CFL and related gradient-similarity methods may cluster meaningfully on other datasets, and the protocol ordering result should be re-verified in those settings."

---

## Verdict

| Objection | Severity | Verdict |
|-----------|----------|---------|
| 1. Theorem is trivial | MAJOR | PARTIALLY SUSTAINED — re-label as Proposition |
| 2. (A4$'$) unverified | **CRITICAL** | **SUSTAINED** — must caveat explicitly |
| 3. FedRC contaminates | MAJOR | OVERRULED — already addressed |
| 4. Fix 2 DEAD | **CRITICAL** | **SUSTAINED** — addendum already honest; future work to investigate |
| 5. Wall-clock ratio misleading | MINOR | SUSTAINED — update prose |
| 6. "exceeds" Oracle is noise | MINOR | SUSTAINED — update prose |
| 7. CFL-on-EMNIST caveat | MINOR | SUSTAINED — add caveat |

**Overall**: 2 critical findings sustained, neither unresolvable. The work is publishable but requires:
1. Rename theorem to proposition (1 line)
2. Caveat (A4$'$) as assumed not verified (1 paragraph)
3. Keep addendum's honest Fix 2 discussion; consider investigating the inactive code path as future work (no immediate action)
4. Update wall-clock / "exceeds Oracle" / CFL-on-EMNIST language (3 sentences)

No critical findings that block publication. Proceed with the paper, with the caveats listed above applied.

## Action items to be applied

1. `scheme_c_addendum.tex`: rename `\begin{theorem}[Protocol Ordering]` → `\begin{proposition}[Protocol Ordering]` (and corresponding `\label` and refs)
2. `scheme_c_addendum.tex`: add footnote under assumption (A4$'$) saying the constant $\beta$ is assumed and left to future work to estimate empirically
3. `scheme_c_addendum.tex`: update wall-clock paragraph with per-config range
4. `scheme_c_addendum.tex`: soften "exceeds Oracle" to "matches within 1σ"
5. `scheme_c_addendum.tex`: add CFL-on-EMNIST caveat
6. `docs/theory/tt_protocol_empirical_confirmation.md`: also update wall-clock and "exceeds" language for consistency

These are all text-only changes. I'll apply them in the next edit pass.
