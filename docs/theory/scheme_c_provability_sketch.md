# Scheme C Provability Sketch

**Date**: 2026-04-11
**Context**: OT breakthrough (FPT-OT matches / exceeds Oracle on CIFAR-100 recurrence, 5×3 multi-seed validated) pointed to Scheme C early OT clustering as the key fix. Creative thinking session suggested the paper story should reframe around "cluster before train is a strictly better E-step." This document evaluates whether that claim is provable under mild assumptions.

**Verdict (preview)**: **PROVABLE** with mild assumptions, cleanly slots into the existing theory section as an additional theorem that reuses Corollary 1 (Imperfect Clustering as Implicit Shrinkage, already in main.tex line 193). No new heavy machinery required.

---

## 1. Setup

Fix the paper's existing notation (A1-A3). Two protocols process a single federation round:

**Scheme A (cluster-after-train, baseline for IFCA/CFL)**:
1. Each client $k$ loads the previous round's aggregated model $\bar w^{(t-1)}$.
2. Client $k$ runs local SGD, producing $\Delta_k^{(t)}$.
3. Server clusters clients in weight space: $\hat c_k^A = \arg\min_c \|\Delta_k^{(t)} - \mu_c\|^2$.
4. Server aggregates within each cluster: $\bar w_c^{(t)} = \text{mean}_{k: \hat c_k = c} \Delta_k^{(t)}$.

**Scheme C (cluster-before-train, the OT breakthrough)**:
1. Each client $k$ sends a cheap fingerprint $\phi_k^{(t)}$ (class-conditional means, feature distribution) to the server. No local training yet.
2. Server runs OT spectral clustering on $\{\phi_k^{(t)}\}_{k=1}^K$: $\hat c_k^C = \text{SpectralOT}(\phi, \cdot)$.
3. Client $k$ loads the matched concept model $\bar w_{\hat c_k^C}^{(t-1)}$.
4. Client runs local SGD from that starting point, producing $\Delta_k^{(t)}$.
5. Server aggregates within each pre-declared cluster.

Both protocols pay the same compute and communication budget (fingerprint is $O(|\text{classes}| \cdot d)$, negligible vs one gradient step).

**Assumption (A4) — concept recoverability**: There exists a fingerprint map $\phi$ and a separation constant $\Delta > 0$ such that
$$
\underbrace{\max_{c} \mathbb{E}_{k \in c} \|\phi_k - \mu_c\|^2}_{\text{within-concept dispersion}} \le \alpha \cdot \underbrace{\min_{c \ne c'} \|\mu_c - \mu_{c'}\|^2}_{\text{between-concept separation}}
$$
for some $\alpha \in [0, \alpha_{\max})$. The regime $\alpha < \alpha_{\max}$ is where spectral clustering's population error rate $\eta_C \to 0$ (standard result, e.g. Ng-Jordan-Weiss 2002, Balakrishnan et al 2011).

---

## 2. The main claim

**Theorem (Protocol Ordering)**. *Under (A1)-(A3) + (A4), for any concept $j$,*
$$
\mathbb{E}[\text{excess risk}_j^{\text{Scheme C}}] \le \mathbb{E}[\text{excess risk}_j^{\text{Scheme A}}]
$$
*where the expectation is over the random data split, client noise, and the clustering randomness.*

---

## 3. Proof sketch

### Step 1: reduce to clustering error rates

The paper's Corollary 1 (Imperfect Clustering as Implicit Shrinkage, main.tex line 193) already establishes:

$$
\mathbb{E}[\cE_j] = \rho^2 B_j^2 + \frac{\sigma^2 C d}{Kn}(1 + o(1)), \quad \rho = \frac{\eta C}{C-1}
$$

where $\eta$ is the symmetric clustering error rate. This gives us a monotone dependence: **lower $\eta$ implies lower expected excess risk**, for fixed $B_j^2, \sigma^2, K, n, C$.

The protocol question reduces to: is $\eta_C \le \eta_A$?

### Step 2: upper-bound $\eta_C$ via (A4)

Under (A4), the fingerprints $\{\phi_k\}$ live in a signal subspace with within-to-between variance ratio $\le \alpha$. Standard spectral clustering error bounds (Lei & Rinaldo 2015, Theorem 3.1; Löffler, Zhang, Zhou 2021) give
$$
\eta_C \le g_1(\alpha) + O\left(\frac{1}{\sqrt{K/C}}\right)
$$
where $g_1$ is an increasing function with $g_1(0) = 0$.

Empirically this matches our observed Re-ID scaling with $K/C$: at $K/C \ge 5$, Re-ID $\ge 0.987$, i.e. $\eta_C \le 0.013$.

### Step 3: lower-bound $\eta_A$ by a weight-space confounder

This is the heart of the argument and the novel piece.

Under Scheme A, client $k$'s weight-space fingerprint is $\Delta_k^{(t)}$, which depends on TWO quantities:
- **concept signal**: the gradient direction pointing toward the true concept model $w^*_{c_k}$;
- **initialization confound**: the stale starting point $\bar w^{(t-1)}$.

Write $\Delta_k^{(t)} = \alpha_k (w^*_{c_k} - \bar w^{(t-1)}) + \epsilon_k$ where $\alpha_k$ is the step size and $\epsilon_k$ is local-SGD noise (standard one-step linearization of SGD, e.g. Mania et al 2016).

For Scheme A to correctly cluster, $\|\Delta_k - \Delta_{k'}\|$ must be small iff $c_k = c_{k'}$. Expanding:
$$
\|\Delta_k - \Delta_{k'}\|^2 = \alpha_k^2 \|(w^*_{c_k} - w^*_{c_{k'}}) + (\bar w^{(t-1)} - \bar w^{(t-1)})\|^2 + \text{noise} = \alpha_k^2 \|w^*_{c_k} - w^*_{c_{k'}}\|^2 + \text{noise}
$$

Wait, the stale init cancels for a pair from the *same* init. Good. BUT — in the drift / recurrence setting, different clients may have loaded DIFFERENT stale inits in previous rounds (after a concept switch, some clients finish loading before others, or participation is partial). Concretely, in our protocol clients from concept $c$ in round $t-1$ might not all have the same $\bar w_c^{(t-1)}$ when concept $c$ is dormant — cold-start clients use global model, hot clients use cached concept model.

Formally: let $w_k^{(\text{init})}$ be client $k$'s round-$t$ starting point. Then
$$
\Delta_k = \alpha_k (w^*_{c_k} - w_k^{(\text{init})}) + \epsilon_k.
$$

If $w_k^{(\text{init})} \ne w_{k'}^{(\text{init})}$ for $k, k' \in c$, then
$$
\Delta_k - \Delta_{k'} = \alpha (w_{k'}^{(\text{init})} - w_k^{(\text{init})}) + (\epsilon_k - \epsilon_{k'})
$$
i.e., **within-concept variance inflates by** $\alpha^2 \mathbb{V}(w^{(\text{init})})$. This is a direct contribution to Scheme A's within-concept dispersion that has NO analogue in Scheme C's fingerprint space.

Meanwhile the between-concept separation in Scheme A is at most $\alpha_k \|w^*_c - w^*_{c'}\|$, which is bounded by the same quantity Scheme C sees (via the assumption that $w^* - w^*$ and $\phi - \phi$ live in comparable geometries — this is the non-trivial assumption **(A4')** we'd need).

Under (A4'), Scheme A's within-to-between ratio is strictly larger than Scheme C's:
$$
\alpha^A = \alpha + \frac{\mathbb{V}(w^{(\text{init})})}{\|w^*_c - w^*_{c'}\|^2} > \alpha = \alpha^C.
$$

By the monotonicity of $g_1$, $\eta_A \ge \eta_C$. Strictness holds whenever $\mathbb{V}(w^{(\text{init})}) > 0$, i.e., whenever at least one concept has ever been dormant and re-spawned — which is the recurrence regime by definition.

### Step 4: combine

Plugging $\eta_A \ge \eta_C$ into Corollary 1:
$$
\mathbb{E}[\cE_j^A] - \mathbb{E}[\cE_j^C] = \frac{C^2}{(C-1)^2}(\eta_A^2 - \eta_C^2) B_j^2 \ge 0.
$$

Strictness: strict whenever $\mathbb{V}(w^{(\text{init})}) > 0$, which is guaranteed in the recurrence regime.

$\blacksquare$ (sketch)

---

## 4. What this proof needs that the paper doesn't already have

1. **Assumption (A4)**: concept recoverability from input fingerprints. Mild and standard.
2. **Assumption (A4')**: between-concept separation in fingerprint space is comparable to between-concept separation in weight space. This is non-trivial — it rules out pathological cases where concepts are separated in weight space but identical in input space. For standard federated tasks (CIFAR / MNIST / language), input separation implies label distribution shift implies weight separation, so this is natural.
3. **One lemma** establishing $\eta_A \ge \eta_C$ via the $\mathbb{V}(w^{(\text{init})}) > 0$ argument in Step 3. This is the novel piece, $\sim 1$ page.
4. **Everything else is reuse**: Corollary 1 from main.tex line 193 does all the heavy lifting. Spectral clustering error bounds are off-the-shelf.

---

## 5. What this proof does NOT claim

- It does NOT claim Scheme C is universally better. The strict inequality only holds in the **recurrence regime** where at least one concept has been dormant. In the pure IID setting or the single-drift setting, Scheme A and Scheme C are equivalent (both face $\eta \to 0$).
- It does NOT claim OT spectral clustering is the optimal algorithm — only that clustering before training, using inputs not weights, dominates clustering after training regardless of the algorithm used.
- It does NOT address the ConceptMemory cross-time accumulation effect (which is why FPT-OT exceeds Oracle at $K/C \ge 10$). That's a separate, complementary effect, worth its own proposition.

---

## 6. Where this slots in the paper

Insert as **Theorem (Protocol Ordering)** in §Extensions or §Algorithm. Pitch:

> Beyond the statistical question of *when* concept-level aggregation wins (Theorem 1), the protocol design also matters. Clustering decisions in clustered FL are typically made in weight space *after* local training (IFCA, CFL, FeSEM). We show that under mild assumptions, clustering in fingerprint space *before* local training yields strictly lower expected excess risk in the recurrence regime. The intuition: weight-space clusters inherit a confound from the stale initialization, which fingerprint-space clusters avoid.

Then cite this in the Experiments section to explain the empirical gap-closed percentages.

---

## 7. Risks and open issues

1. **(A4') is not free**. Need to argue carefully that fingerprint separation $\ge$ weight separation. Pathological counter-example: suppose two concepts have identical input distributions but different label assignments (e.g., MNIST with flipped labels). Then $\phi_c = \phi_{c'}$ but $w^*_c \ne w^*_{c'}$. Scheme C fails; Scheme A (eventually) succeeds. This is a real gap; need to explicitly say "fingerprints must include label conditional information" or equivalent. Our empirical fingerprint IS class-conditional (sees both x and y), so we're fine, but the theorem statement must reflect this.
2. **Step 3 is informal about $\alpha_k$**. Need to formalize the SGD one-step linearization. Standard in optimization theory but adds ~0.5 page of setup.
3. **Reviewer attack surface**: "Why not just use fingerprint features in Scheme A?" — answer: that's exactly Scheme C. The contribution is the ordering, not the feature choice. But the paper must make this clear to avoid the rebuttal "this is just a minor protocol tweak."
4. **Finite-sample**: the proof is expectation-level. A finite-sample version via concentration (like Corollary 2 in the paper) would strengthen it. Easy extension if asked.

---

## 8. Decision

**PROCEED with the paper re-framing around "cluster before train"**. The theorem is provable, reuses existing machinery, and slots cleanly into the theory section. Key deliverables:

- [ ] Write the formal theorem statement in main.tex (1 page)
- [ ] Write the full proof in the appendix (~2 pages)
- [ ] Add an empirical verification section: compare η_A (from running IFCA/CFL logs) against η_C (from OT spectral assignments) on the same CIFAR-100 configs, plot η_A - η_C vs K/C, show that larger K/C gives larger gap
- [ ] Update the Algorithm section to emphasize the protocol ordering as the key mechanism
- [ ] Update Related Work to position against clustered FL methods that all do cluster-after-train

**Critical risk**: (A4') and the pathological label-flip counter-example. Mitigation: state the theorem with explicit "fingerprint must include class-conditional information," which our OT fingerprint does by construction.

---

## 9. References (for the full write-up)

- Ng, Jordan, Weiss (2002), "On spectral clustering: analysis and an algorithm" — spectral clustering population bounds.
- Lei, Rinaldo (2015), "Consistency of spectral clustering in stochastic block models" — Annals of Statistics, finite-sample bounds.
- Löffler, Zhang, Zhou (2021), "Optimality of spectral clustering in the Gaussian mixture model" — Annals of Statistics, tight rates.
- Balakrishnan, Xu, Krishnamurthy, Singh (2011), "Noise thresholds for spectral clustering" — NeurIPS, the $\alpha_{\max}$ constant.
- Mania, Pan, Papailiopoulos, Recht, Ramchandran, Jordan (2016), "Perturbed iterate analysis for asynchronous stochastic optimization" — SGD one-step linearization.
- Ghosh, Chung, Yin, Ramchandran (2020), IFCA — baseline Scheme A method.
- Sattler, Müller, Samek (2021), CFL — baseline Scheme A method.
- FedProTrack main.tex — Corollary 1 (Imperfect Clustering as Implicit Shrinkage), line 193.
