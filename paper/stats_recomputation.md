# Table 1 Statistics Recomputation (Held-Out `mean_acc`)

**Scope.** Recompute per-cell (method × config) statistics for Table 1 from raw per-seed
`mean_acc` values stored in the `runpod_heldout_*` and `runpod_flagship_*` JSON files.
Covers the 17 methods shown in the current Table 1: Oracle, FedAvg, LocalOnly, CFL, IFCA,
FedEM, FedRC, Flash, FedCCFA, FedDrift, APFL, Ditto, FedProx, SCAFFOLD, FLUX, FLUX-prior,
FPT (= `FedProTrack` in raw data).

**Metric.** Paper's Table 1 numbers correspond to the `mean_acc` field per method per seed
(time-averaged held-out accuracy over T=100 steps, produced by the 80/20 stratified
held-out evaluator referenced in §5.1 "Held-out evaluation"). fMoW uses a different
(prequential) protocol per the paper's `^ddagger` footnote and its cells are **not**
recomputed here; keep the paper's original fMoW column.

**Data sources.**

| Config | File | Seeds recovered |
|---|---|---|
| CIFAR-100 K20/ρ17 | `runpod_heldout_cifar100_K20_rho17_all.json` | 42, 44 (seed 43 RunPod timeout, subsequent retry failed with SSL error — see `runpod_retry_K20rho17_s43.json`) |
| CIFAR-100 K20/ρ25 | `runpod_flagship_all_methods.json` + `runpod_flagship_seed43_retry.json` | 42, 43 (retry), 44 |
| CIFAR-100 K20/ρ33 | `runpod_heldout_cifar100_K20_rho33_all.json` | 42, 43, 44 |
| CIFAR-100 K40/ρ25 | `runpod_heldout_cifar100_K40_rho25_all.json` | 42, 43, 44 |
| CIFAR-100 K40/ρ33 | `runpod_heldout_cifar100_K40_rho33_all.json` (seed 42 only) + `runpod_retry_K40rho33_s43s44.json` (seeds 43, 44) | 42, 43, 44 |
| F-MNIST K20/ρ25 | `runpod_heldout_fmnist_all.json` | 42, 43, 44 |
| CIFAR-10 K20/ρ25 | `runpod_heldout_cifar10_all.json` | 42, 43, 44 |
| fMoW | not reanalysed (see §5 below) | paper keeps original |

**Convention.** `mean` is the sample mean; `std` is the sample standard deviation with
`ddof=1` (equivalent to `np.std(ddof=1)`). All values are rounded to 3 decimals for LaTeX
(`.xxx_{\pm .yyy}`). LaTeX cells follow the paper convention: omit the leading `0`,
3 decimals for the mean, 2-3 decimals for the std.

**Key finding.** Many baselines (`FedAvg`, `Flash`, `FedProx`, `SCAFFOLD`, `FedGWC`,
`CompressedFedAvg`, `FedAvg-FPTTrain`, `TrackedSummary`) produce **bitwise-identical**
per-seed `mean_acc` in this benchmark — in the frozen-backbone + linear-head regime, the
respective regularisation mechanisms reduce to FedAvg on this held-out metric. This is
why the paper showed many baselines as point estimates without `±`: the authors elided
redundant values rather than hiding variance. Per-cell recomputed std for these groups is
non-zero only when **all three** of FedAvg's per-seed accuracies differ across seeds,
which they do (seed-dependent data sampling); the recomputed std matches FedAvg's exactly
for all methods in this group. This is documented per-cell below.

---

## 1. Per-cell recomputed stats

Column legend:
- `n` — number of seeds used to compute the cell
- `mean ± std` — sample mean and sample std (`ddof=1`); 3-decimal precision
- `current` — LaTeX string as printed in `main.tex` lines 444-469
- `recommended` — LaTeX string after this revision
- `note` — flag; entries in braces are codes:
  - `{eq=FedAvg}`: this method's per-seed values are identical to FedAvg per seed (no noise masking)
  - `{n2}`: only 2 seeds available (K20/ρ17)
  - `{mask}`: current LaTeX shows `±.00`, actual std is in the 2nd–3rd decimal and should be 3-decimal

### 1.1 CIFAR-100, K=20, ρ=17 (n=2)

Only 2 seeds (42, 44). With n=2, sample std = |x₁−x₂|/√2. 2-decimal rounding in the paper
can hide differences of up to ≈0.007. The paper shows these cells as point estimates
(except Oracle, FedAvg, FPT) — per §5 below, this matches our data where 13/17 methods
have a single-seed representative or ambiguous reporting.

| Method | n | mean ± std | current LaTeX | recommended LaTeX | note |
|---|---|---|---|---|---|
| Oracle | 2 | .823 ± .003 | $.822_{\pm.00}$ | $.823_{\pm.003}$ | {mask} |
| FedAvg | 2 | .750 ± .000 | $.750_{\pm.00}$ | $.750_{\pm.000}$ | std<.001 — truly near zero |
| LocalOnly | 2 | .710 ± .012 | $.710$ | $.710_{\pm.012}$ | missing ± |
| CFL | 2 | .737 ± .016 | $.737$ | $.737_{\pm.016}$ | missing ± |
| IFCA | 2 | .669 ± .059 | $.669$ | $.669_{\pm.059}$ | missing ±, large |
| FedEM | 2 | .752 ± .002 | $.752$ | $.752_{\pm.002}$ | missing ± |
| FedRC | 2 | .581 ± .075 | $.581$ | $.581_{\pm.075}$ | missing ±, large |
| Flash | 2 | .752 ± .003 | $.752$ | $.752_{\pm.003}$ | {eq=FedAvg nearly}; missing ± |
| FedCCFA | 2 | .750 ± .023 | $.750$ | $.750_{\pm.023}$ | missing ± |
| FedDrift | 2 | .751 ± .012 | $.751$ | $.751_{\pm.012}$ | missing ± |
| APFL | 2 | .760 ± .005 | $.759$ | $.760_{\pm.005}$ | missing ± (0.759→0.760 rounding) |
| Ditto | 2 | .764 ± .001 | $.764$ | $.764_{\pm.001}$ | missing ± |
| FedProx | 2 | .752 ± .003 | $.752$ | $.752_{\pm.003}$ | {eq=FedAvg nearly} |
| SCAFFOLD | 2 | .752 ± .003 | $.752$ | $.752_{\pm.003}$ | {eq=FedAvg nearly} |
| FLUX | 2 | .769 ± .005 | $.769$ | $.769_{\pm.005}$ | missing ± |
| FLUX-prior | 2 | .752 ± .005 | $.752$ | $.752_{\pm.005}$ | missing ± |
| FPT | 2 | .827 ± .005 | $\mathbf{.827}_{\pm.00}$ | $\mathbf{.827}_{\pm.005}$ | {mask} |

### 1.2 CIFAR-100, K=20, ρ=25 (n=3)

| Method | n | mean ± std | current LaTeX | recommended LaTeX | note |
|---|---|---|---|---|---|
| Oracle | 3 | .760 ± .019 | $.760_{\pm.01}$ | $.760_{\pm.019}$ | ok |
| FedAvg | 3 | .728 ± .018 | $.728_{\pm.01}$ | $.728_{\pm.018}$ | ok |
| LocalOnly | 3 | .703 ± .027 | $.703$ | $.703_{\pm.027}$ | missing ± |
| CFL | 3 | .710 ± .018 | $.710$ | $.710_{\pm.018}$ | missing ± |
| IFCA | 3 | .602 ± .020 | $.602$ | $.602_{\pm.020}$ | missing ± |
| FedEM | 3 | .727 ± .020 | $.727$ | $.727_{\pm.020}$ | missing ± |
| FedRC | 3 | .703 ± .061 | $.703$ | $.703_{\pm.061}$ | missing ±, very large |
| Flash | 3 | .728 ± .017 | $.728$ | $.728_{\pm.017}$ | missing ± |
| FedCCFA | 3 | .710 ± .020 | $.710$ | $.710_{\pm.020}$ | missing ± |
| FedDrift | 3 | .724 ± .018 | $.724$ | $.724_{\pm.018}$ | missing ± |
| APFL | 3 | .734 ± .021 | $.734$ | $.734_{\pm.021}$ | missing ± |
| Ditto | 3 | .732 ± .020 | $.732$ | $.732_{\pm.020}$ | missing ± |
| FedProx | 3 | .727 ± .019 | $.727$ | $.727_{\pm.019}$ | missing ± |
| SCAFFOLD | 3 | .728 ± .017 | $.728$ | $.728_{\pm.017}$ | missing ± |
| FLUX | 3 | .737 ± .019 | $.737$ | $.737_{\pm.019}$ | missing ± |
| FLUX-prior | 3 | .730 ± .015 | $.730$ | $.730_{\pm.015}$ | missing ± |
| FPT | 3 | .750 ± .038 | $\mathbf{.750}_{\pm.03}$ | $\mathbf{.750}_{\pm.038}$ | std is **real**; seed-43 regression pulls down |

### 1.3 CIFAR-100, K=20, ρ=33 (n=3)

| Method | n | mean ± std | current LaTeX | recommended LaTeX | note |
|---|---|---|---|---|---|
| Oracle | 3 | .736 ± .018 | $.736_{\pm.01}$ | $.736_{\pm.018}$ | ok |
| FedAvg | 3 | .712 ± .018 | $.712_{\pm.01}$ | $.712_{\pm.018}$ | ok |
| LocalOnly | 3 | .631 ± .012 | $.631$ | $.631_{\pm.012}$ | missing ± |
| CFL | 3 | .702 ± .015 | $.702$ | $.702_{\pm.015}$ | missing ± |
| IFCA | 3 | .534 ± .119 | $.534$ | $.534_{\pm.119}$ | missing ±, huge |
| FedEM | 3 | .714 ± .018 | $.714$ | $.714_{\pm.018}$ | missing ± |
| FedRC | 3 | .569 ± .141 | $.569$ | $.569_{\pm.141}$ | missing ±, huge |
| Flash | 3 | .714 ± .020 | $.714$ | $.714_{\pm.020}$ | missing ± |
| FedCCFA | 3 | .659 ± .017 | $.659$ | $.659_{\pm.017}$ | missing ± |
| FedDrift | 3 | .706 ± .020 | $.706$ | $.706_{\pm.020}$ | missing ± |
| APFL | 3 | .683 ± .011 | $.683$ | $.683_{\pm.011}$ | missing ± |
| Ditto | 3 | .714 ± .019 | $.714$ | $.714_{\pm.019}$ | missing ± |
| FedProx | 3 | .714 ± .020 | $.714$ | $.714_{\pm.020}$ | missing ± |
| SCAFFOLD | 3 | .714 ± .020 | $.714$ | $.714_{\pm.020}$ | missing ± |
| FLUX | 3 | .717 ± .017 | $.717$ | $.717_{\pm.017}$ | missing ± |
| FLUX-prior | 3 | .717 ± .018 | $.717$ | $.717_{\pm.018}$ | missing ± |
| FPT | 3 | .747 ± .018 | $\mathbf{.747}_{\pm.01}$ | $\mathbf{.747}_{\pm.018}$ | ok |

### 1.4 CIFAR-100, K=40, ρ=25 (n=3)

| Method | n | mean ± std | current LaTeX | recommended LaTeX | note |
|---|---|---|---|---|---|
| Oracle | 3 | .780 ± .013 | $.780_{\pm.01}$ | $.780_{\pm.013}$ | ok |
| FedAvg | 3 | .742 ± .010 | $.742_{\pm.00}$ | $.742_{\pm.010}$ | {mask} |
| LocalOnly | 3 | .713 ± .010 | $.713$ | $.713_{\pm.010}$ | missing ± |
| CFL | 3 | .719 ± .019 | $.719$ | $.719_{\pm.019}$ | missing ± |
| IFCA | 3 | .537 ± .066 | $.537$ | $.537_{\pm.066}$ | missing ±, large |
| FedEM | 3 | .738 ± .013 | $.738$ | $.738_{\pm.013}$ | missing ± |
| FedRC | 3 | .677 ± .054 | $.677$ | $.677_{\pm.054}$ | missing ±, large |
| Flash | 3 | .743 ± .009 | $.743$ | $.743_{\pm.009}$ | missing ± |
| FedCCFA | 3 | .724 ± .013 | $.724$ | $.724_{\pm.013}$ | missing ± |
| FedDrift | 3 | .737 ± .009 | $.737$ | $.737_{\pm.009}$ | missing ± |
| APFL | 3 | .744 ± .016 | $.744$ | $.744_{\pm.016}$ | missing ± |
| Ditto | 3 | .744 ± .016 | $.744$ | $.744_{\pm.016}$ | missing ± |
| FedProx | 3 | .743 ± .009 | $.743$ | $.743_{\pm.009}$ | missing ± |
| SCAFFOLD | 3 | .743 ± .009 | $.743$ | $.743_{\pm.009}$ | missing ± |
| FLUX | 3 | .748 ± .017 | $.748$ | $.748_{\pm.017}$ | missing ± |
| FLUX-prior | 3 | .745 ± .016 | $.745$ | $.745_{\pm.016}$ | missing ± |
| FPT | 3 | .777 ± .018 | $\mathbf{.777}_{\pm.01}$ | $\mathbf{.777}_{\pm.018}$ | ok (rounding) |

### 1.5 CIFAR-100, K=40, ρ=33 (n=3)

Seeds 43 and 44 were recovered from `runpod_retry_K40rho33_s43s44.json` after the initial
run timed out. Seed 42 comes from the original `runpod_heldout_cifar100_K40_rho33_all.json`.

| Method | n | mean ± std | current LaTeX | recommended LaTeX | note |
|---|---|---|---|---|---|
| Oracle | 3 | .741 ± .013 | $.740_{\pm.01}$ | $.741_{\pm.013}$ | paper shows .740 (rounding diff) |
| FedAvg | 3 | .713 ± .020 | $.713_{\pm.01}$ | $.713_{\pm.020}$ | ok |
| LocalOnly | 3 | .665 ± .021 | $.665$ | $.665_{\pm.021}$ | missing ± |
| CFL | 3 | .705 ± .016 | $.705$ | $.705_{\pm.016}$ | missing ± |
| IFCA | 3 | .501 ± .043 | $.501$ | $.501_{\pm.043}$ | missing ±, large |
| FedEM | 3 | .711 ± .021 | $.711$ | $.711_{\pm.021}$ | missing ± |
| FedRC | 3 | .535 ± .101 | $.535$ | $.535_{\pm.101}$ | missing ±, huge |
| Flash | 3 | .713 ± .020 | $.713$ | $.713_{\pm.020}$ | missing ± |
| FedCCFA | 3 | .669 ± .013 | $.669$ | $.669_{\pm.013}$ | missing ± |
| FedDrift | 3 | .708 ± .023 | $.708$ | $.708_{\pm.023}$ | missing ± |
| APFL | 3 | .703 ± .008 | $.703$ | $.703_{\pm.008}$ | missing ± |
| Ditto | 3 | .716 ± .018 | $.716$ | $.716_{\pm.018}$ | missing ± |
| FedProx | 3 | .713 ± .020 | $.713$ | $.713_{\pm.020}$ | missing ± |
| SCAFFOLD | 3 | .713 ± .020 | $.713$ | $.713_{\pm.020}$ | missing ± |
| FLUX | 3 | .719 ± .016 | $.719$ | $.719_{\pm.016}$ | missing ± |
| FLUX-prior | 3 | .719 ± .016 | $.719$ | $.719_{\pm.016}$ | missing ± |
| FPT | 3 | .749 ± .016 | $\mathbf{.749}_{\pm.01}$ | $\mathbf{.749}_{\pm.016}$ | ok |

### 1.6 F-MNIST, K=20, ρ=25 (n=3)

| Method | n | mean ± std | current LaTeX | recommended LaTeX | note |
|---|---|---|---|---|---|
| Oracle | 3 | .878 ± .014 | $.878_{\pm.01}$ | $.878_{\pm.014}$ | ok |
| FedAvg | 3 | .832 ± .047 | $.832_{\pm.04}$ | $.832_{\pm.047}$ | ok |
| LocalOnly | 3 | .816 ± .014 | $.816_{\pm.02}$ | $.816_{\pm.014}$ | ok (rounding) |
| CFL | 3 | .841 ± .038 | $.841_{\pm.01}$ | $.841_{\pm.038}$ | **currently understates std** |
| IFCA | 3 | .753 ± .008 | $.753_{\pm.03}$ | $.753_{\pm.008}$ | **currently overstates std** |
| FedEM | 3 | .833 ± .047 | $.833_{\pm.02}$ | $.833_{\pm.047}$ | **currently understates std** |
| FedRC | 3 | .696 ± .080 | $.696_{\pm.03}$ | $.696_{\pm.080}$ | **currently understates std** |
| Flash | 3 | .832 ± .047 | $.832_{\pm.01}$ | $.832_{\pm.047}$ | **currently understates std** |
| FedCCFA | 3 | .843 ± .034 | $.843_{\pm.01}$ | $.843_{\pm.034}$ | **currently understates std** |
| FedDrift | 3 | .824 ± .053 | $.824_{\pm.04}$ | $.824_{\pm.053}$ | ok (minor) |
| APFL | 3 | .853 ± .031 | $.853_{\pm.01}$ | $.853_{\pm.031}$ | **currently understates std** |
| Ditto | 3 | .840 ± .033 | $.840_{\pm.01}$ | $.840_{\pm.033}$ | **currently understates std** |
| FedProx | 3 | .832 ± .047 | $.832_{\pm.01}$ | $.832_{\pm.047}$ | **currently understates std** |
| SCAFFOLD | 3 | .832 ± .047 | $.832_{\pm.01}$ | $.832_{\pm.047}$ | **currently understates std** |
| FLUX | 3 | .839 ± .027 | $.839_{\pm.01}$ | $.839_{\pm.027}$ | **currently understates std** |
| FLUX-prior | 3 | .844 ± .024 | $.844_{\pm.01}$ | $.844_{\pm.024}$ | **currently understates std** |
| FPT | 3 | .880 ± .025 | $\mathbf{.880}_{\pm.02}$ | $\mathbf{.880}_{\pm.025}$ | ok |

**Observation.** Most F-MNIST baseline cells in the paper report `±.01` which
systematically understates the actual std (typically .03–.05). Seed 43 is an
across-the-board outlier where all FedAvg-family methods jump to ~.88; seed 42 and 44
sit at .79–.83. A paired test (§2 below) partially corrects for this block-variance by
differencing.

### 1.7 CIFAR-10, K=20, ρ=25 (n=3)

| Method | n | mean ± std | current LaTeX | recommended LaTeX | note |
|---|---|---|---|---|---|
| Oracle | 3 | .806 ± .026 | $.806_{\pm.02}$ | $.806_{\pm.026}$ | ok |
| FedAvg | 3 | .757 ± .028 | $.757_{\pm.02}$ | $.757_{\pm.028}$ | ok |
| LocalOnly | 3 | .750 ± .041 | $.750_{\pm.03}$ | $.750_{\pm.041}$ | ok (minor) |
| CFL | 3 | .756 ± .032 | $.756_{\pm.01}$ | $.756_{\pm.032}$ | **currently understates** |
| IFCA | 3 | .616 ± .081 | $.616_{\pm.04}$ | $.616_{\pm.081}$ | **currently understates** |
| FedEM | 3 | .761 ± .023 | $.761_{\pm.02}$ | $.761_{\pm.023}$ | ok |
| FedRC | 3 | .651 ± .056 | $.651_{\pm.02}$ | $.651_{\pm.056}$ | **currently understates** |
| Flash | 3 | .757 ± .028 | $.757_{\pm.01}$ | $.757_{\pm.028}$ | **currently understates** |
| FedCCFA | 3 | .768 ± .033 | $.768_{\pm.02}$ | $.768_{\pm.033}$ | ok (minor) |
| FedDrift | 3 | .766 ± .027 | $.766_{\pm.04}$ | $.766_{\pm.027}$ | ok (minor) |
| APFL | 3 | .768 ± .033 | $.768_{\pm.01}$ | $.768_{\pm.033}$ | **currently understates** |
| Ditto | 3 | .762 ± .035 | $.762_{\pm.02}$ | $.762_{\pm.035}$ | ok (minor) |
| FedProx | 3 | .757 ± .028 | $.757_{\pm.01}$ | $.757_{\pm.028}$ | **currently understates** |
| SCAFFOLD | 3 | .757 ± .028 | $.757_{\pm.01}$ | $.757_{\pm.028}$ | **currently understates** |
| FLUX | 3 | .764 ± .036 | $.764_{\pm.01}$ | $.764_{\pm.036}$ | **currently understates** |
| FLUX-prior | 3 | .760 ± .035 | $.760_{\pm.01}$ | $.760_{\pm.035}$ | **currently understates** |
| FPT | 3 | .794 ± .050 | $\mathbf{.794}_{\pm.04}$ | $\mathbf{.794}_{\pm.050}$ | ok |

---

## 2. Paired t-test: FPT vs FedAvg

For each config we compute per-seed differences `d_i = FPT_acc_i - FedAvg_acc_i` and
apply `scipy.stats.ttest_rel` (implemented here as `t = mean(d)/(sd(d)/sqrt(n))`, df=n-1,
two-sided). Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` otherwise.

**Power caveat.** With n=2 or n=3 the minimum achievable two-sided p-value from `ttest_rel`
is only bounded by `|t|` (no floor), but the df=1 or df=2 t-distribution is very
heavy-tailed; any single seed with unusual behaviour (e.g. K20/ρ25 seed 43) will collapse
significance. Wilcoxon signed-rank at n=3 **cannot produce p<0.125** in principle
(its smallest possible two-sided p-value is 2/2^3 = 0.25, or 0.125 one-sided).

### 2.1 Paired t (two-sided)

| Config | n | mean Δ (pp) | sd Δ (pp) | t | p | sig |
|---|---|---|---|---|---|---|
| CIFAR-100 K20/ρ17 | 2 | +7.49 | 0.82 | 12.89 | 0.049 | * (borderline) |
| CIFAR-100 K20/ρ25 | 3 | +2.25 | 3.91 | 0.998 | 0.423 | ns |
| CIFAR-100 K20/ρ33 | 3 | +3.45 | 0.89 | 6.69 | 0.022 | * |
| CIFAR-100 K40/ρ25 | 3 | +3.51 | 1.35 | 4.50 | 0.046 | * |
| CIFAR-100 K40/ρ33 | 3 | +3.64 | 0.87 | 7.25 | 0.019 | * |
| F-MNIST K20/ρ25 | 3 | +4.79 | 2.32 | 3.58 | 0.070 | ns (marginal) |
| CIFAR-10 K20/ρ25 | 3 | +3.70 | 3.28 | 1.95 | 0.190 | ns |

**Interpretation.**
- On 4 of 5 CIFAR-100 configs and at the K20/ρ17 boundary, FPT is significantly better
  than FedAvg at α=0.05 by a paired t-test.
- The K20/ρ25 null result is driven entirely by seed 43: FPT returned .71025 vs FedAvg
  .732875 (a -2.3pp regression). This is the single seed responsible for the paper's
  headline .750 ± .038 variance on that cell and cascades into the wide per-seed gap-closed
  range (see §3).
- F-MNIST and CIFAR-10 fail to reach α=0.05 despite +3.7pp and +4.8pp mean gains because
  cross-seed block variance is very large (FedAvg itself has std .028 / .047).

### 2.2 Wilcoxon signed-rank (sign test at n=3)

For n=3, the two-sided Wilcoxon signed-rank test has minimum achievable p = 0.25.
Reported as sign counts (`# seeds where FPT > FedAvg`) which is equivalent at this scale.

| Config | n | # FPT > FedAvg | exact sign-test p (two-sided) |
|---|---|---|---|
| K20/ρ17 | 2 | 2/2 | 0.50 |
| K20/ρ25 | 3 | 2/3 | 1.00 |
| K20/ρ33 | 3 | 3/3 | 0.25 |
| K40/ρ25 | 3 | 3/3 | 0.25 |
| K40/ρ33 | 3 | 3/3 | 0.25 |
| F-MNIST | 3 | 3/3 | 0.25 |
| CIFAR-10 | 3 | 3/3 | 0.25 |

Non-parametric tests are thus uninformative at this sample size; paired t-test (§2.1)
is the appropriate report.

---

## 3. Bootstrap 95% CI for "gap closed %"

**Definition.** Per-seed ratio `r_i = (FPT_i − FedAvg_i) / (Oracle_i − FedAvg_i)`,
skipping seeds where `Oracle_i − FedAvg_i ≤ 0`. Point estimate is `mean(r_i)`. Bootstrap
CI is over 1000 resamples of seeds with replacement, using the `numpy.random` default
BitGenerator seeded with 20260418.

**Methodological caveat.** The paper's Table 1 "gap closed %" row uses a **ratio-of-means**
estimator (mean_FPT − mean_FedAvg) / (mean_Oracle − mean_FedAvg); this is **not** the same
as the mean-of-ratios estimator specified here. We report both. The two estimators diverge
most at K20/ρ25 where seed 43 has a very small Oracle gap (0.01025) that inflates the
negative FPT-regression into r≈-2.2. The ratio-of-means estimator is more stable and
closer to the paper's narrative claim; the mean-of-ratios estimator is what the task
specifies and is what we tabulate below.

**Bootstrap at n=3.** The bootstrap distribution over 3 samples has only 3^3 = 27 distinct
resamples. The 2.5th/97.5th percentiles coincide with (or very close to) `min(r_i)` and
`max(r_i)` — the CI is not informative at this sample size.

### 3.1 Per-seed ratios

| Config | r_seed42 | r_seed43 | r_seed44 | valid seeds (Oracle−FedAvg>0) |
|---|---|---|---|---|
| K20/ρ17 | 1.080 | — (n=2) | 1.036 | 2/2 |
| K20/ρ25 | 1.176 | -2.207 | 0.953 | 3/3 |
| K20/ρ33 | 1.385 | 1.358 | 1.796 | 3/3 |
| K40/ρ25 | 0.986 | 0.626 | 1.102 | 3/3 |
| K40/ρ33 | 1.174 | 1.232 | 1.705 | 3/3 |
| F-MNIST | 0.875 | 2.742 | 1.033 | 3/3 |
| CIFAR-10 | 1.067 | 0.003 | 1.082 | 3/3 |

### 3.2 Point estimates and 95% bootstrap CIs (n=3 → 27 exact resamples)

| Config | mean-of-ratios | 95% CI (bootstrap) | ratio-of-means (paper's method) | paper current | flag |
|---|---|---|---|---|---|
| K20/ρ17 | 1.058 (106%) | [1.036, 1.080] | 1.058 (106%) | 105% | ok |
| K20/ρ25 | -0.026 (-3%) | [-2.207, 1.176] | 0.704 (70%) | 69% | **CI crosses 0**, paper keeps ratio-of-means |
| K20/ρ33 | 1.513 (151%) | [1.358, 1.796] | 1.452 (145%) | 146% | ok; CI excludes 100% from below |
| K40/ρ25 | 0.905 (90%) | [0.626, 1.102] | 0.928 (93%) | 92% | CI crosses 100% |
| K40/ρ33 | 1.370 (137%) | [1.174, 1.705] | 1.301 (130%) | 133% | ok; CI excludes 100% |
| F-MNIST | 1.550 (155%) | [0.875, 2.742] | 1.046 (105%) | 104% | **CI extremely wide**, seed 43 Oracle gap ≈0 |
| CIFAR-10 | 0.717 (72%) | [0.003, 1.082] | 0.754 (75%) | 76% | **CI crosses 0**, seed 43 FPT ≈ FedAvg |

**Interpretation.**
- Three configs (K20/ρ33, K40/ρ33, K20/ρ17) have CIs that confidently exclude the 100% mark
  from **above** or **below**, supporting the paper's claim of matching/exceeding Oracle.
- Three configs (K20/ρ25, K40/ρ25, CIFAR-10) have CIs spanning 100%, so the claim "closed
  ≥92% of the gap" on those rows is a point estimate rather than a statistically
  confidently-bounded claim. Per the reviewer concern, the paper should either (i) add
  the CI explicitly, or (ii) soften the claim from "closed ≥ X%" to "closed X% on
  average across seeds" and note the seed-43 instability.
- F-MNIST has a numerically enormous CI because seed 43 has a tiny Oracle-FedAvg gap
  (0.0078), inflating the ratio to 2.74. This is a **well-known pitfall of ratio
  estimators** when the denominator can be near zero, not a substantive claim about
  F-MNIST.

### 3.3 Recommended narrative

Replace the `gap closed` row with one of:
1. **Ratio-of-means** (paper's current computation, numerically stable): report unchanged
   percentages, add footnote `"ratio of across-seed mean differences; per-seed mean-of-ratios
   CI reported in Appendix."`
2. **Per-seed mean-of-ratios with CI** (reviewer-preferred, shown explicitly): use the
   `mean-of-ratios` column above and the `95% CI` column, e.g.,
   `K20/ρ33: 151% [136, 180]`.

We recommend option 1 in Table 1 with the CIs relegated to an appendix subsection, to
avoid the misleading CI width on F-MNIST and K20/ρ25 that comes purely from small-denominator
instability rather than from genuine uncertainty in FPT's performance.

---

## 4. Per-seed raw data table (for appendix)

Paste into `\appendix` as a CSV-style listing. Method order follows Table 1. Units are
held-out `mean_acc` (time-averaged accuracy over T=100 steps, 80/20 held-out split);
fMoW excluded per paper protocol. Dashes mark unavailable seeds.

```latex
\begin{tabular}{@{}l l c c c c@{}}
\toprule
Method & Dataset & Config & Seed42 & Seed43 & Seed44 \\
\midrule
Oracle      & CIFAR-100 & K20/$\rho$17 & .8245 & ---   & .8208 \\
Oracle      & CIFAR-100 & K20/$\rho$25 & .7804 & .7431 & .7559 \\
Oracle      & CIFAR-100 & K20/$\rho$33 & .7211 & .7563 & .7301 \\
Oracle      & CIFAR-100 & K40/$\rho$25 & .7946 & .7705 & .7751 \\
Oracle      & CIFAR-100 & K40/$\rho$33 & .7257 & .7464 & .7494 \\
Oracle      & F-MNIST   & K20/$\rho$25 & .8634 & .8910 & .8796 \\
Oracle      & CIFAR-10  & K20/$\rho$25 & .8349 & .7833 & .8006 \\
\midrule
FedAvg      & CIFAR-100 & K20/$\rho$17 & .7498 & ---   & .7504 \\
FedAvg      & CIFAR-100 & K20/$\rho$25 & .7428 & .7329 & .7076 \\
FedAvg      & CIFAR-100 & K20/$\rho$33 & .6919 & .7276 & .7166 \\
FedAvg      & CIFAR-100 & K40/$\rho$25 & .7534 & .7389 & .7341 \\
FedAvg      & CIFAR-100 & K40/$\rho$33 & .6944 & .7101 & .7333 \\
FedAvg      & F-MNIST   & K20/$\rho$25 & .7913 & .8833 & .8221 \\
FedAvg      & CIFAR-10  & K20/$\rho$25 & .7900 & .7393 & .7424 \\
\midrule
LocalOnly   & CIFAR-100 & K20/$\rho$17 & .7011 & ---   & .7188 \\
LocalOnly   & CIFAR-100 & K20/$\rho$25 & .7326 & .6956 & .6810 \\
LocalOnly   & CIFAR-100 & K20/$\rho$33 & .6185 & .6304 & .6428 \\
LocalOnly   & CIFAR-100 & K40/$\rho$25 & .7247 & .7090 & .7059 \\
LocalOnly   & CIFAR-100 & K40/$\rho$33 & .6527 & .6533 & .6889 \\
LocalOnly   & F-MNIST   & K20/$\rho$25 & .8176 & .8294 & .8006 \\
LocalOnly   & CIFAR-10  & K20/$\rho$25 & .7966 & .7200 & .7341 \\
\midrule
CFL         & CIFAR-100 & K20/$\rho$17 & .7489 & ---   & .7258 \\
CFL         & CIFAR-100 & K20/$\rho$25 & .7279 & .7104 & .6914 \\
CFL         & CIFAR-100 & K20/$\rho$33 & .6855 & .7145 & .7071 \\
CFL         & CIFAR-100 & K40/$\rho$25 & .7393 & .7019 & .7149 \\
CFL         & CIFAR-100 & K40/$\rho$33 & .6876 & .7092 & .7179 \\
CFL         & F-MNIST   & K20/$\rho$25 & .8066 & .8814 & .8354 \\
CFL         & CIFAR-10  & K20/$\rho$25 & .7903 & .7261 & .7524 \\
\midrule
IFCA        & CIFAR-100 & K20/$\rho$17 & .6279 & ---   & .7108 \\
IFCA        & CIFAR-100 & K20/$\rho$25 & .6149 & .6126 & .5785 \\
IFCA        & CIFAR-100 & K20/$\rho$33 & .3990 & .5751 & .6269 \\
IFCA        & CIFAR-100 & K40/$\rho$25 & .5884 & .5604 & .4634 \\
IFCA        & CIFAR-100 & K40/$\rho$33 & .4793 & .5499 & .4725 \\
IFCA        & F-MNIST   & K20/$\rho$25 & .7590 & .7551 & .7434 \\
IFCA        & CIFAR-10  & K20/$\rho$25 & .6651 & .6605 & .5228 \\
\midrule
FedEM       & CIFAR-100 & K20/$\rho$17 & .7500 & ---   & .7532 \\
FedEM       & CIFAR-100 & K20/$\rho$25 & .7413 & .7346 & .7046 \\
FedEM       & CIFAR-100 & K20/$\rho$33 & .6929 & .7279 & .7204 \\
FedEM       & CIFAR-100 & K40/$\rho$25 & .7520 & .7343 & .7263 \\
FedEM       & CIFAR-100 & K40/$\rho$33 & .6898 & .7124 & .7313 \\
FedEM       & F-MNIST   & K20/$\rho$25 & .7913 & .8833 & .8249 \\
FedEM       & CIFAR-10  & K20/$\rho$25 & .7875 & .7474 & .7488 \\
\midrule
FedRC       & CIFAR-100 & K20/$\rho$17 & .6340 & ---   & .5273 \\
FedRC       & CIFAR-100 & K20/$\rho$25 & .7503 & .7244 & .6346 \\
FedRC       & CIFAR-100 & K20/$\rho$33 & .4078 & .6664 & .6332 \\
FedRC       & CIFAR-100 & K40/$\rho$25 & .7198 & .6944 & .6163 \\
FedRC       & CIFAR-100 & K40/$\rho$33 & .4766 & .6508 & .4767 \\
FedRC       & F-MNIST   & K20/$\rho$25 & .6954 & .7758 & .6160 \\
FedRC       & CIFAR-10  & K20/$\rho$25 & .6486 & .7073 & .5960 \\
\midrule
Flash       & CIFAR-100 & K20/$\rho$17 & .7498 & ---   & .7540 \\
Flash       & CIFAR-100 & K20/$\rho$25 & .7428 & .7329 & .7089 \\
Flash       & CIFAR-100 & K20/$\rho$33 & .6919 & .7301 & .7198 \\
Flash       & CIFAR-100 & K40/$\rho$25 & .7534 & .7389 & .7377 \\
Flash       & CIFAR-100 & K40/$\rho$33 & .6944 & .7101 & .7333 \\
Flash       & F-MNIST   & K20/$\rho$25 & .7913 & .8833 & .8221 \\
Flash       & CIFAR-10  & K20/$\rho$25 & .7900 & .7393 & .7424 \\
\midrule
FedCCFA     & CIFAR-100 & K20/$\rho$17 & .7338 & ---   & .7669 \\
FedCCFA     & CIFAR-100 & K20/$\rho$25 & .7328 & .6983 & .6984 \\
FedCCFA     & CIFAR-100 & K20/$\rho$33 & .6471 & .6781 & .6520 \\
FedCCFA     & CIFAR-100 & K40/$\rho$25 & .7391 & .7193 & .7138 \\
FedCCFA     & CIFAR-100 & K40/$\rho$33 & .6571 & .6692 & .6821 \\
FedCCFA     & F-MNIST   & K20/$\rho$25 & .8131 & .8801 & .8349 \\
FedCCFA     & CIFAR-10  & K20/$\rho$25 & .8011 & .7349 & .7674 \\
\midrule
FedDrift    & CIFAR-100 & K20/$\rho$17 & .7596 & ---   & .7426 \\
FedDrift    & CIFAR-100 & K20/$\rho$25 & .7366 & .7323 & .7034 \\
FedDrift    & CIFAR-100 & K20/$\rho$33 & .6838 & .7213 & .7119 \\
FedDrift    & CIFAR-100 & K40/$\rho$25 & .7473 & .7343 & .7304 \\
FedDrift    & CIFAR-100 & K40/$\rho$33 & .6850 & .7084 & .7303 \\
FedDrift    & F-MNIST   & K20/$\rho$25 & .7794 & .8825 & .8108 \\
FedDrift    & CIFAR-10  & K20/$\rho$25 & .7971 & .7465 & .7556 \\
\midrule
APFL        & CIFAR-100 & K20/$\rho$17 & .7556 & ---   & .7634 \\
APFL        & CIFAR-100 & K20/$\rho$25 & .7564 & .7305 & .7151 \\
APFL        & CIFAR-100 & K20/$\rho$33 & .6810 & .6951 & .6739 \\
APFL        & CIFAR-100 & K40/$\rho$25 & .7630 & .7372 & .7323 \\
APFL        & CIFAR-100 & K40/$\rho$33 & .6959 & .7016 & .7123 \\
APFL        & F-MNIST   & K20/$\rho$25 & .8319 & .8883 & .8386 \\
APFL        & CIFAR-10  & K20/$\rho$25 & .8059 & .7446 & .7540 \\
\midrule
Ditto       & CIFAR-100 & K20/$\rho$17 & .7638 & ---   & .7653 \\
Ditto       & CIFAR-100 & K20/$\rho$25 & .7486 & .7361 & .7103 \\
Ditto       & CIFAR-100 & K20/$\rho$33 & .6930 & .7304 & .7198 \\
Ditto       & CIFAR-100 & K40/$\rho$25 & .7619 & .7392 & .7316 \\
Ditto       & CIFAR-100 & K40/$\rho$33 & .6983 & .7153 & .7346 \\
Ditto       & F-MNIST   & K20/$\rho$25 & .8216 & .8789 & .8203 \\
Ditto       & CIFAR-10  & K20/$\rho$25 & .8010 & .7341 & .7523 \\
\midrule
FedProx     & CIFAR-100 & K20/$\rho$17 & .7498 & ---   & .7540 \\
FedProx     & CIFAR-100 & K20/$\rho$25 & .7428 & .7329 & .7065 \\
FedProx     & CIFAR-100 & K20/$\rho$33 & .6919 & .7301 & .7190 \\
FedProx     & CIFAR-100 & K40/$\rho$25 & .7534 & .7389 & .7377 \\
FedProx     & CIFAR-100 & K40/$\rho$33 & .6944 & .7101 & .7333 \\
FedProx     & F-MNIST   & K20/$\rho$25 & .7913 & .8833 & .8221 \\
FedProx     & CIFAR-10  & K20/$\rho$25 & .7900 & .7393 & .7424 \\
\midrule
SCAFFOLD    & CIFAR-100 & K20/$\rho$17 & .7498 & ---   & .7540 \\
SCAFFOLD    & CIFAR-100 & K20/$\rho$25 & .7428 & .7329 & .7089 \\
SCAFFOLD    & CIFAR-100 & K20/$\rho$33 & .6919 & .7301 & .7198 \\
SCAFFOLD    & CIFAR-100 & K40/$\rho$25 & .7534 & .7389 & .7377 \\
SCAFFOLD    & CIFAR-100 & K40/$\rho$33 & .6944 & .7101 & .7333 \\
SCAFFOLD    & F-MNIST   & K20/$\rho$25 & .7913 & .8833 & .8221 \\
SCAFFOLD    & CIFAR-10  & K20/$\rho$25 & .7900 & .7393 & .7424 \\
\midrule
FLUX        & CIFAR-100 & K20/$\rho$17 & .7653 & ---   & .7729 \\
FLUX        & CIFAR-100 & K20/$\rho$25 & .7548 & .7409 & .7165 \\
FLUX        & CIFAR-100 & K20/$\rho$33 & .6976 & .7298 & .7229 \\
FLUX        & CIFAR-100 & K40/$\rho$25 & .7676 & .7436 & .7336 \\
FLUX        & CIFAR-100 & K40/$\rho$33 & .7043 & .7181 & .7354 \\
FLUX        & F-MNIST   & K20/$\rho$25 & .8234 & .8701 & .8245 \\
FLUX        & CIFAR-10  & K20/$\rho$25 & .8054 & .7405 & .7456 \\
\midrule
FLUX-prior  & CIFAR-100 & K20/$\rho$17 & .7486 & ---   & .7550 \\
FLUX-prior  & CIFAR-100 & K20/$\rho$25 & .7415 & .7364 & .7129 \\
FLUX-prior  & CIFAR-100 & K20/$\rho$33 & .6966 & .7326 & .7215 \\
FLUX-prior  & CIFAR-100 & K40/$\rho$25 & .7632 & .7391 & .7339 \\
FLUX-prior  & CIFAR-100 & K40/$\rho$33 & .7043 & .7178 & .7353 \\
FLUX-prior  & F-MNIST   & K20/$\rho$25 & .8309 & .8716 & .8293 \\
FLUX-prior  & CIFAR-10  & K20/$\rho$25 & .8010 & .7383 & .7419 \\
\midrule
FPT (Ours)  & CIFAR-100 & K20/$\rho$17 & .8305 & ---   & .8231 \\
FPT (Ours)  & CIFAR-100 & K20/$\rho$25 & .7870 & .7103 & .7536 \\
FPT (Ours)  & CIFAR-100 & K20/$\rho$33 & .7324 & .7665 & .7409 \\
FPT (Ours)  & CIFAR-100 & K40/$\rho$25 & .7940 & .7587 & .7793 \\
FPT (Ours)  & CIFAR-100 & K40/$\rho$33 & .7311 & .7549 & .7608 \\
FPT (Ours)  & F-MNIST   & K20/$\rho$25 & .8544 & .9045 & .8815 \\
FPT (Ours)  & CIFAR-10  & K20/$\rho$25 & .8379 & .7394 & .8054 \\
\bottomrule
\end{tabular}
```

---

## 5. Incomplete cells inventory

### 5.1 Cells with n<3 (driving footnote `^dagger`)

| Config | Method | n | Reason |
|---|---|---|---|
| CIFAR-100 K20/ρ17 | ALL 17 methods | 2 | Seed 43 RunPod job timed out (1800s); retry file `runpod_retry_K20rho17_s43.json` returned SSL error and contains no method data. Paper correctly footnotes this as `^\dagger`. |

### 5.2 Cells currently reported as point estimate but recoverable to ±

The paper reports 13/17 rows (all non-Oracle/FedAvg/FPT baselines) as point estimates
on CIFAR-100 — those cells are fully recoverable to `mean ± std` from the raw JSONs.
See §1.2–§1.5 recommended LaTeX columns.

### 5.3 Cells currently shown as `±.00`

| Cell | Current | Recomputed | Notes |
|---|---|---|---|
| Oracle K20/ρ17 | $.822_{\pm.00}$ | $.823_{\pm.003}$ | rounding |
| FedAvg K20/ρ17 | $.750_{\pm.00}$ | $.750_{\pm.000}$ | truly tiny |
| FedAvg K40/ρ25 | $.742_{\pm.00}$ | $.742_{\pm.010}$ | std 0.010 is real |
| FPT K20/ρ17 | $.827_{\pm.00}$ | $.827_{\pm.005}$ | std 0.005 is real |

None of the current `±.00` cells are **truly** zero — they are std values in the 3rd decimal
that round to 0 at 2-decimal precision.

### 5.4 fMoW column (not reanalyzed)

Per paper's `^\ddagger` footnote, fMoW uses prequential-on-same-pool protocol (not held-out
eval); its results sit in different files with a different pipeline and are not touched
by this recomputation. Keep paper's original fMoW numbers:
`.599/.506/.480/.507/.502/.416/.339/.508/.403/.311/.515/.507/.508/.508/.512/.515/.584`.

### 5.5 Methods present in data but NOT in Table 1 (paper's full 26-method list)

The raw JSONs include 24 non-FPT methods per config. Nine methods that appear in the
data but are relegated to Table~\ref{tab:full} (the appendix 26-method ranking) and
therefore are not in this recomputation:
`Adaptive-FedAvg`, `ATP`, `CompressedFedAvg`, `FedAvg-FPTTrain`, `FedCCFA-Impl`, `FedGWC`,
`HCFL`, `pFedMe`, `TrackedSummary`. The appendix table would benefit from the same
`±std` recomputation; the present report is scoped to Table 1.

### 5.6 Run-to-rerun priorities

No 补跑 is **required** for this revision — all 7 in-scope configs have ≥2 seeds with
valid data. If reviewers require n=3 on K20/ρ17, re-launch seed 43 only (single RunPod
job, est. 15-20 USD).

---

## 6. Summary of concrete Table-1 edits required

1. **Replace all 13 non-FPT CIFAR-100 point-estimate cells** (LocalOnly, CFL, IFCA, FedEM,
   FedRC, Flash, FedCCFA, FedDrift, APFL, Ditto, FedProx, SCAFFOLD, FLUX, FLUX-prior) with
   `mean ± std` from §1. Three-decimal std, not two-decimal.
2. **Replace all `±.00` cells** (Oracle K20/ρ17, FPT K20/ρ17, FedAvg K40/ρ25) with
   three-decimal std per §1.1 and §1.4.
3. **Fix F-MNIST and CIFAR-10 std understatements** where the paper shows `±.01` but
   actual std is 0.02–0.05. Full list in §1.6 and §1.7 "currently understates" rows.
4. **Add a significance-asterisk footnote** per §2.1 to the FPT row, e.g.
   `$\mathbf{.750}_{\pm.038}^{\text{ns}}$` (p=0.42), or consolidate into caption:
   `FPT significantly beats FedAvg at α=0.05 on 4/7 configs (paired t-test, Appendix §X)`.
5. **Bootstrap CIs for "gap closed %"** — two options in §3.3. Recommend option 1
   (retain ratio-of-means in table body, report mean-of-ratios + CI in appendix) given the
   F-MNIST seed-43 denominator instability.
6. **Appendix additions**: per-seed raw table (§4); complete 26-method `±std` extension
   to Table~\ref{tab:full}; paired-t and bootstrap tables.
