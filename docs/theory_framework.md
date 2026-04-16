# Aggregation Granularity Tradeoff under Federated Concept Drift

## A Theoretical Framework for NeurIPS 2026

---

## 1. Problem Setting

We study supervised learning in a federated system with $K$ clients over
$T$ discrete communication rounds. At each round $t$, client $k$ is assigned
to exactly one of $C$ latent concepts via a (possibly time-varying) assignment
function $c: [K] \times [T] \to [C]$. Let

$$
\mathcal{C}_j(t) := \{k \in [K] : c(k,t) = j\}
$$

denote the set of clients assigned to concept $j$ at time $t$, and write
$K_j(t) := |\mathcal{C}_j(t)|$.

**Data model.** Client $k$ at round $t$ draws $n$ i.i.d. samples
$(x_i, y_i)_{i=1}^{n}$ from a distribution $P_{c(k,t)}$. We work in a
Gaussian discriminative model: for concept $j$,

$$
P_j: \quad x \sim \mathcal{N}(\mu_j, \Sigma), \qquad y = \langle w_j^*, x \rangle + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2),
$$

where $\mu_j \in \mathbb{R}^d$ is the concept-specific mean, $\Sigma \succ 0$
is a shared covariance, $w_j^* \in \mathbb{R}^d$ is the concept-specific
Bayes-optimal linear predictor, and $\sigma^2$ is the noise variance. We write
$\kappa := \lambda_{\max}(\Sigma) / \lambda_{\min}(\Sigma)$ for the condition
number.

**Concept separation.** Define the minimum pairwise concept distance and the
maximum concept radius:

$$
\Delta := \min_{i \neq j} \|w_i^* - w_j^*\|_\Sigma, \qquad
R := \max_j \|w_j^* - \bar{w}^*\|_\Sigma,
$$

where $\bar{w}^* := \frac{1}{C} \sum_{j=1}^{C} w_j^*$ is the concept centroid
and $\|v\|_\Sigma := \sqrt{v^\top \Sigma\, v}$.

**Notation.** $\lesssim$ hides universal constants. $\tilde{O}(\cdot)$ hides
logarithmic factors in $d$, $K$, $n$, and $1/\delta$.

---

## 2. Two Aggregation Strategies

We compare two one-shot estimators for a target concept $j$.

**Global estimator.** Pool all $Kn$ samples and compute the ordinary
least-squares (OLS) estimator:

$$
\hat{w}_{\mathrm{glob}} := \arg\min_{w} \frac{1}{Kn} \sum_{k=1}^{K} \sum_{i=1}^{n} (y_{k,i} - \langle w, x_{k,i} \rangle)^2.
$$

**Concept-level estimator.** Pool only the $K_j n$ samples from clients in
$\mathcal{C}_j$ and compute OLS:

$$
\hat{w}_j := \arg\min_{w} \frac{1}{K_j n} \sum_{k \in \mathcal{C}_j} \sum_{i=1}^{n} (y_{k,i} - \langle w, x_{k,i} \rangle)^2.
$$

The excess risk for concept $j$ under estimator $\hat{w}$ is

$$
\mathcal{E}_j(\hat{w}) := \mathbb{E}_{P_j}[\ell(\hat{w}; x, y)] - \mathbb{E}_{P_j}[\ell(w_j^*; x, y)]
= \|\hat{w} - w_j^*\|_\Sigma^2.
$$

---

## 3. Excess Risk Decomposition (Proposition 1)

**Assumptions.**
- (A1) All concepts have equal client share: $K_j = K/C$ for all $j$.
- (A2) The design matrix per client satisfies standard sub-Gaussian concentration (implied by the Gaussian model above).
- (A3) $n \ge \Omega(d \log d)$ so that sample covariance is well-conditioned within each client.

**Proposition 1** (Bias-Variance Decomposition). *Under (A1)--(A3), the
expected excess risks satisfy:*

$$
\mathbb{E}[\mathcal{E}_j(\hat{w}_{\mathrm{glob}})]
= \underbrace{\left\| w_j^* - \frac{1}{C}\sum_{\ell=1}^{C} w_\ell^* \right\|_\Sigma^2
  + \frac{C-1}{C^2} \sum_{\ell \neq j} \|w_j^* - w_\ell^*\|_\Sigma^2 \cdot \frac{K_\ell^2}{K^2}}_{\text{interference bias}^2}
\;+\; \underbrace{\frac{\sigma^2 d}{Kn} (1 + o(1))}_{\text{estimation variance}},
$$

$$
\mathbb{E}[\mathcal{E}_j(\hat{w}_j)]
= \underbrace{0}_{\text{no interference}}
\;+\; \underbrace{\frac{\sigma^2 d}{(K/C) n} (1 + o(1))}_{\text{estimation variance}}.
$$

*Proof sketch.* The global OLS estimator converges to the population minimiser
of the mixture $\frac{1}{C}\sum_j P_j$. Because each $P_j$ has a different
optimal predictor $w_j^*$, the population minimiser of the mixture is a
weighted average of the $w_j^*$, biased away from any individual $w_j^*$. The
bias term is the squared $\Sigma$-norm of this displacement. The variance term
follows from standard OLS theory with $Kn$ (resp. $(K/C)n$) effective samples.
The $o(1)$ remainder is $O(\kappa d / (Kn))$ under (A3). $\square$

**Remark.** Under the balanced assumption (A1), the interference bias
simplifies to

$$
B_j^2 := \|w_j^* - \bar{w}^*\|_\Sigma^2
= \frac{1}{C^2} \left\|\sum_{\ell \neq j}(w_j^* - w_\ell^*)\right\|_\Sigma^2.
$$

In the worst case (two concepts, $C=2$), $B_j^2 = \Delta^2/4$.

---

## 4. Crossover Condition (Theorem 1)

**Theorem 1** (Concept-Level Crossover). *Under (A1)--(A3), the concept-level
estimator $\hat{w}_j$ achieves lower expected excess risk than the global
estimator $\hat{w}_{\mathrm{glob}}$ for concept $j$ if and only if:*

$$
B_j^2 > \frac{\sigma^2 d (C - 1)}{K n} (1 + \rho_{d,n,K}),
\tag{1}
$$

*where $|\rho_{d,n,K}| \le c_0 \kappa d / \min(Kn, (K/C)n)$ for a universal
constant $c_0 > 0$.*

*Equivalently, up to the remainder $\rho$, concept-level aggregation is
preferable when*

$$
\frac{Kn \cdot B_j^2}{\sigma^2 d} > C - 1.
\tag{2}
$$

*Proof.* The concept-level estimator wins when the global interference bias
exceeds the additional variance cost of using fewer samples:

$$
B_j^2 + \frac{\sigma^2 d}{Kn}(1+o(1))
> \frac{\sigma^2 d}{(K/C)n}(1+o(1)).
$$

Rearranging and noting
$\frac{1}{K/C} - \frac{1}{K} = \frac{C-1}{K}$
gives (1). $\square$

**Interpretation.** The crossover is controlled by the
*signal-to-noise ratio of concept separation*:

$$
\mathrm{SNR}_{\mathrm{concept}} := \frac{K n \cdot B_j^2}{\sigma^2 d}.
$$

When $\mathrm{SNR}_{\mathrm{concept}} > C - 1$, concepts are sufficiently
separated relative to statistical uncertainty, and concept-level aggregation
wins. When $\mathrm{SNR}_{\mathrm{concept}} < C - 1$ (e.g., early rounds with
small $n$, or high noise), global aggregation is preferable despite its bias.

**Comparison with Ghosh et al. (2020).** The IFCA convergence analysis
(Ghosh et al., 2020) establishes conditions under which alternating
minimisation recovers the correct cluster assignments and the corresponding
estimators converge. Their focus is on the *convergence rate* and
*identifiability* of the clustering procedure itself. Our result is
complementary: we characterise *when clustering is worth doing at all*,
regardless of whether cluster recovery succeeds. In regimes where
$\mathrm{SNR}_{\mathrm{concept}} < C - 1$, even an oracle that knows the
true concept assignments cannot outperform global pooling -- the variance
penalty of splitting dominates the bias reduction.

---

## 5. Non-Stationary Extension (Theorem 2)

We now allow the concept assignment $c(k,t)$ to change over time. We model
this as piecewise-stationary drift: concept assignments are constant for
*stability periods* of length $\tau$ rounds, after which a client may switch
to a different concept.

**Assumption (A4)** (Piecewise stationarity). There exists a partition of
$[T]$ into contiguous intervals $I_1, I_2, \ldots$ of length at most $\tau$
such that $c(k, \cdot)$ is constant on each $I_m$ for every client $k$.

**Assumption (A5)** (Exponential forgetting). The server uses a sliding window
or exponential weighting with effective window size $W \le \tau$ rounds.

**Theorem 2** (Time-Varying Crossover). *Under (A1)--(A5), consider a client
at time $t$ that is $s$ rounds into its current stability period ($1 \le s
\le \tau$). The effective sample size available to the concept-level estimator
is*

$$
N_{\mathrm{eff}}^{(j)}(s) = \frac{K}{C} \cdot n \cdot \min(s, W).
$$

*The concept-level estimator has lower excess risk than the global estimator
if and only if*

$$
B_j^2 > \frac{\sigma^2 d (C-1)}{K \cdot n \cdot \min(s, W)} (1 + \rho'),
\tag{3}
$$

*where $\rho'$ is a remainder of the same order as in Theorem 1.*

*In particular, immediately after a concept switch ($s = 1$), condition (3)
requires*

$$
B_j^2 > \frac{\sigma^2 d (C-1)}{K n},
$$

*which is $\min(s,W)$ times harder to satisfy than in steady state. There
exists a transient period*

$$
s^* = \left\lceil \frac{\sigma^2 d (C-1)}{K n B_j^2} \right\rceil
$$

*during which global aggregation has lower risk even under concept
heterogeneity. For $s > s^*$, concept-level aggregation becomes preferable.*

**Corollary 1** (Drift-Rate Crossover). *If $\tau < s^*$, i.e., concepts
switch faster than the transient recovery period, then global aggregation
dominates throughout. Concept-level aggregation is only beneficial when*

$$
\tau > s^* = \Theta\!\left(\frac{\sigma^2 d C}{K n \Delta^2}\right).
$$

**Interpretation.** This formalises the intuition that under rapid drift,
there is insufficient time for the concept-level estimator to accumulate
enough samples to overcome its variance disadvantage. The optimal granularity
is *time-varying*: global immediately after a switch, transitioning to
concept-level after $s^*$ rounds.

---

## 6. Shrinkage Estimator (Theorem 3)

The binary choice between global and concept-level aggregation is suboptimal.
A principled interpolation arises naturally from the James-Stein / empirical
Bayes framework.

**Setup.** Treat the concept-level OLS estimates $\hat{w}_1, \ldots,
\hat{w}_C$ as noisy observations of the true parameters:

$$
\hat{w}_j = w_j^* + \xi_j, \qquad
\xi_j \sim \mathcal{N}\!\left(0, \frac{\sigma^2}{(K/C)n} \Sigma^{-1}\right).
$$

Place an empirical prior on the $w_j^*$:

$$
w_j^* \sim \mathcal{N}(\bar{w}^*, \sigma_B^2 I_d),
$$

where $\sigma_B^2$ captures the between-concept dispersion.

**Theorem 3** (Shrinkage Estimator). *The posterior mean under the above model
is the James-Stein shrinkage estimator:*

$$
\hat{w}_j^{\mathrm{shrink}} = (1 - \lambda_j)\,\hat{w}_j + \lambda_j\,\bar{\hat{w}},
$$

*where $\bar{\hat{w}} := \frac{1}{C}\sum_j \hat{w}_j$ and the shrinkage
coefficient is*

$$
\lambda_j = \frac{\sigma^2 / ((K/C) n \cdot \lambda_{\min}(\Sigma))}
{\sigma^2 / ((K/C) n \cdot \lambda_{\min}(\Sigma)) + \sigma_B^2}.
\tag{4}
$$

*This estimator interpolates adaptively between the two minimax-optimal
regimes and achieves near-oracle excess risk. Specifically:*

- *When $\sigma_B^2 \gg \sigma^2 / ((K/C)n\,\lambda_{\min}(\Sigma))$
  (concepts are well-separated relative to estimation noise), $\lambda_j \to 0$
  and $\hat{w}_j^{\mathrm{shrink}} \to \hat{w}_j$ (concept-level).*
- *When $\sigma_B^2 \ll \sigma^2 / ((K/C)n\,\lambda_{\min}(\Sigma))$
  (concepts are close or samples are scarce), $\lambda_j \to 1$ and
  $\hat{w}_j^{\mathrm{shrink}} \to \bar{\hat{w}}$ (global).*

**Practical estimation of $\sigma_B^2$.** The between-concept variance can be
estimated from the observed dispersion of concept-level estimators:

$$
\hat{\sigma}_B^2 = \max\!\left(
  \frac{1}{(C-1) d} \sum_{j=1}^{C} \|\hat{w}_j - \bar{\hat{w}}\|^2
  - \frac{\sigma^2}{(K/C) n \cdot \lambda_{\min}(\Sigma)},\;
  0
\right).
$$

This makes the shrinkage coefficient *data-adaptive*, requiring no
hyperparameter tuning beyond what the empirical Bayes framework provides.

**Remark (relation to FedProTrack soft aggregation).** The two-phase protocol
in FedProTrack performs soft aggregation with posterior-weighted mixing across
concept slots. This can be understood as a structured variant of the shrinkage
estimator above, where the shrinkage target is not the global mean but a
posterior-weighted combination of memory-bank entries. The shrinkage
coefficient is implicitly determined by the Gibbs posterior concentration, which
plays the role of $\hat{\sigma}_B^2$ estimation.

---

## 7. Information-Theoretic Lower Bounds (Theorem 4)

We establish that the crossover phenomenon is *fundamental*: no estimator can
simultaneously avoid the interference cost and the variance cost.

**Theorem 4** (Minimax Lower Bounds). *Consider the class $\mathcal{F}(C, \Delta)$ of
federated learning problems with $C$ concepts satisfying
$\min_{i\neq j}\|w_i^*-w_j^*\|_\Sigma \ge \Delta$ and noise level $\sigma^2$.
For any estimator $\hat{w}$ that does not use concept labels:*

$$
\sup_{P \in \mathcal{F}(C,\Delta)}
\mathbb{E}\!\left[\frac{1}{C}\sum_{j=1}^{C} \|\hat{w} - w_j^*\|_\Sigma^2\right]
\;\ge\; \Omega\!\left(\frac{\Delta^2}{C}\right).
\tag{5}
$$

*For any estimator $\hat{w}_j$ that uses concept labels but is restricted to
the $K/C$ clients in concept $j$:*

$$
\sup_{P \in \mathcal{F}(C,\Delta)}
\mathbb{E}\!\left[\|\hat{w}_j - w_j^*\|_\Sigma^2\right]
\;\ge\; \Omega\!\left(\frac{\sigma^2 d}{(K/C)\,n}\right).
\tag{6}
$$

*Consequently, any aggregation strategy pays at least*

$$
\max\!\left(
  \Omega\!\left(\frac{\Delta^2}{C}\right)(1 - p_{\mathrm{cluster}}),\;
  \Omega\!\left(\frac{\sigma^2 d}{(K/C)\, n}\right) p_{\mathrm{cluster}}
\right),
\tag{7}
$$

*where $p_{\mathrm{cluster}} \in [0,1]$ is the effective degree of
concept-specific aggregation.*

*Proof sketch.*

*Lower bound (5):* We apply the standard hypothesis-testing reduction. Consider
$C=2$ concepts with $w_1^* = \bar{w}^* + (\Delta/2)\,e_1$ and
$w_2^* = \bar{w}^* - (\Delta/2)\,e_1$ for an arbitrary unit vector $e_1$. Any
estimator that does not distinguish concepts must target a single $\hat{w}$.
By the triangle inequality,
$\|\hat{w}-w_1^*\|_\Sigma^2 + \|\hat{w}-w_2^*\|_\Sigma^2 \ge \Delta^2/2$.
Averaging over concepts gives (5).

*Lower bound (6):* This is the standard minimax rate for $d$-dimensional linear
regression with $(K/C)\,n$ Gaussian samples and noise variance $\sigma^2$
(see, e.g., Tsybakov, 2009, Theorem 2.5). $\square$

**Corollary 2** (Fundamental Crossover). *No estimator achieves risk below
$\Omega(\Delta^2/C)$ without concept-level information, and no concept-level
estimator achieves risk below $\Omega(\sigma^2 d C / (Kn))$ without access to
other concepts' data. The two lower bounds cross at*

$$
\frac{Kn \Delta^2}{\sigma^2 d} = \Theta(C^2),
$$

*confirming that the crossover condition in Theorem 1 is tight up to the
factor $C$ vs. $C-1$ and constant factors.*

---

## 8. Summary of Conditions and Regimes

| Regime | Condition | Optimal Strategy |
|--------|-----------|-----------------|
| Low separation / early rounds | $Kn B_j^2 / (\sigma^2 d) < C-1$ | Global aggregation |
| High separation / steady state | $Kn B_j^2 / (\sigma^2 d) > C-1$ | Concept-level aggregation |
| Moderate / uncertain | Intermediate $\mathrm{SNR}_{\mathrm{concept}}$ | Shrinkage toward global mean |
| Post-switch transient ($s < s^*$) | $s < \sigma^2 d(C-1)/(Kn B_j^2)$ | Global (temporarily) |
| Rapid drift ($\tau < s^*$) | $\tau < \sigma^2 dC/(Kn\Delta^2)$ | Global (permanently) |

---

## 9. Assumptions and Limitations

1. **Gaussian / linear model.** The closed-form decomposition relies on
   Gaussianity and linearity. However, the qualitative bias-variance tradeoff
   extends to generalised linear models and, under appropriate regularity
   conditions, to overparameterised models via the effective dimension
   $d_{\mathrm{eff}}$ replacing $d$ (Bartlett et al., 2020).

2. **Balanced concepts.** The equal-share assumption (A1) simplifies
   exposition. For unbalanced concepts, $K/C$ is replaced by $K_j$ throughout,
   and the crossover condition becomes concept-specific.

3. **Known concept assignments.** Theorems 1--3 assume oracle concept labels.
   In practice, concept labels must be inferred (e.g., via IFCA-style
   alternating minimisation or FedProTrack's posterior inference). Mis-assignment
   probability $\delta^{\mathrm{id}}$ adds an $O(\delta^{\mathrm{id}}
   \Delta^2)$ term to the concept-level excess risk (cf. Proposition 1 in
   the companion identifiability analysis).

4. **One-shot vs. iterative.** Our results are stated for one-shot OLS to
   isolate the aggregation tradeoff. In iterative federated optimisation
   (e.g., FedAvg with multiple communication rounds), the per-round analysis
   applies with $n$ replaced by the effective per-round sample count and
   additional terms for client drift (Li et al., 2020).

5. **No privacy constraints.** We do not model differential privacy noise.
   Under $(\varepsilon, \delta)$-DP, the variance terms increase by
   $O(d^2 / (Kn\varepsilon^2))$, shifting the crossover toward concept-level
   aggregation (the bias term is unaffected by DP noise).

---

## 10. Connection to FedProTrack Experiments

The theoretical framework directly predicts the empirical findings of the
diagnostic experiments:

- **Oracle intervention (6.0pp gap).** The oracle knows true concept labels,
  eliminating interference bias entirely. The observed 6.0pp accuracy gap
  between oracle and global aggregation on CIFAR-100 is a direct measurement
  of $B_j^2$ in that setting.

- **Identity tracking contributes only 0.1pp.** Consistent with the theory:
  once concept-level aggregation is achieved (removing the $B_j^2$ bias),
  further improvements from better identity tracking are second-order,
  bounded by $O(\delta^{\mathrm{id}} \Delta^2)$ with small $\delta^{\mathrm{id}}$.

- **Post-switch transient.** The observed initial accuracy dip after concept
  switches, followed by recovery, matches the time-varying crossover in
  Theorem 2 with the predicted recovery period $s^*$.

---

## References

- Bartlett, P. L., Long, P. M., Lugosi, G., & Tsigler, A. (2020). Benign overfitting in linear regression. *PNAS*, 117(48), 30063--30070.
- Ghosh, A., Chung, J., Yin, D., & Ramchandran, K. (2020). An efficient framework for clustered federated learning. *NeurIPS*, 33, 19586--19597.
- Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). Federated optimization in heterogeneous networks. *MLSys*, 2, 429--450.
- Tsybakov, A. B. (2009). *Introduction to Nonparametric Estimation*. Springer.
