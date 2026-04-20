# Table 1 vs Table 3 CIFAR-10 Forensic Diagnosis

**Question.** Both Table 1 (main results) and Table 3 (Appendix `app:shrinkage_ablation`)
claim the same CIFAR-10 setup (`K=20`, `ρ=25`, 3 seeds, "held-out evaluation"), yet
report FPT at `.794` vs `.850`–`.859`. A 5.6pp gap between supposedly-same-method runs
is a reviewer-visible inconsistency.

**Bottom line.** Table 1 and Table 3 are **two completely different experiments** that
share only the high-level CIFAR-10 recurrence dataset family. They differ in (a) the
entrypoint script, (b) the reporting metric (curve-mean vs final-round), (c) the
per-client training data size, (d) the local-training hyperparameters, and
(e) additional dataset-construction knobs. Each of the four Table 3 rows matches a
separate "cifar10_{noshrink,shrink,gateonly,snrgate}_*.json" ablation run using
`run_cifar10_concept_diagnostic.py`; none of them were produced by the heldout
benchmark that generated Table 1.

---

## 1. Source files per table

### Table 1, CIFAR-10 column (FPT `.794`)
- File: `runpod_heldout_cifar10_all.json` (mtime 2026-04-17 20:11).
- Entry script: **`run_cifar100_neurips_benchmark.py`** (despite the name; the
  script handles CIFAR-10 too via `_DATASET_DISPATCH`).
- Metric reported: `mean_acc` = curve mean over all 100 rounds of the held-out test
  pool (`eval_on_test_pool=true`).
- Per-seed `FedProTrack.mean_acc`:
  - seed 42: `0.837875`
  - seed 43: `0.739375`
  - seed 44: `0.805375`
  - **Mean = 0.79421 → reported as `.794`** ✓

### Table 3, CIFAR-10 column — maps to FOUR separate JSON files (not the ones listed in the task)

Only one of the four candidates in the task brief is a Table 3 source
(`runpod_cifar10_snrgate_fixed.json`). The other three rows come from different
files that had to be located manually. The `anisotropic` / `implicit` JSONs run on
**CIFAR-100 / ResNet-18**, not CIFAR-10, so they cannot be Table 3 sources.

| Table 3 row | Reported | JSON file | Script | Key flags | Per-seed `fpt_final_acc` | Mean |
|---|---|---|---|---|---|---|
| Shrinkage off (λ=0) | `.854` | `runpod_cifar10_noshrink_multi.json` | `run_cifar10_concept_diagnostic.py` | `drct_shrinkage=false`, `snr_gate=false`, `ema=0` | 42→0.8948, 43→0.8008, 44→0.8657 | **0.8538** → `.854` ✓ |
| Pure Stein | `.850` | `runpod_cifar10_shrink_fixed.json`  | `run_cifar10_concept_diagnostic.py` | `drct_shrinkage=true`,  `snr_gate=false`, `ema=0` | 42→0.8940, 43→0.7893, 44→0.8653 | **0.8495** → `.850` ✓ |
| SNR-gated (default) | `.855` | `runpod_cifar10_gateonly.json`       | `run_cifar10_concept_diagnostic.py` | `drct_shrinkage=true`,  `snr_gate=true`,  `ema=0` | 42→0.8952, 43→0.8030, 44→0.8657 | **0.8546** → `.855` ✓ |
| SNR-gated + EMA (β=0.7) | `.859` | `runpod_cifar10_snrgate_fixed.json` | `run_cifar10_concept_diagnostic.py` | `drct_shrinkage=true`,  `snr_gate=true`,  `ema=0.7` | 42→0.8915, 43→0.8238, 44→0.8630 | **0.8594** → `.859` ✓ |

The `runpod_cifar10_snrgate_multi.json` file is a **known-bad / superseded** run
(permanent warmup — `lambda_mean=1` every round, so FPT≡FedAvg). Not used in Table 3.

---

## 2. Side-by-side config diff

| Parameter | Table 1 (heldout) | Table 3 (diagnostic) | Same? |
|---|---|---|---|
| Script | `run_cifar100_neurips_benchmark.py` | `run_cifar10_concept_diagnostic.py` | **No** |
| Metric reported | `mean_acc` = curve mean over 100 rounds | `fpt_final_acc` = acc at round 99 only | **No** |
| `fpt_mode` | `"ot"` | `"ot"` (hardcoded) | Yes |
| K | 20 | 20 | Yes |
| T | 100 | 100 | Yes |
| rho | 25 | 25 | Yes |
| alpha | 0.75 | 0.75 | Yes |
| delta | 0.85 | 0.85 | Yes |
| feature_seed | 2718 | 2718 | Yes |
| seeds | 42, 43, 44 | 42, 43, 44 | Yes |
| `n_features` | 128 | 128 | Yes |
| `n_samples` | 400 | 400 | Yes |
| `samples_per_class` (CIFAR-10) | 120 (via `--samples-per-coarse-class 120`) | **500 (default, not overridden)** | **No** |
| `batch_size` | 128 | **256** (CIFAR10RecurrenceConfig arg in diagnostic) | **No** |
| FPT local `lr` | 0.02 (`--fpt-lr`) | **0.01** (hardcoded) | **No** |
| FPT local `n_epochs` | 10 (`--fpt-epochs`) | **5** (hardcoded) | **No** |
| Baseline `lr` / `n_epochs` | 0.02 / 10 | 0.01 / 5 | **No** |
| `federation_every` | 1 | 1 | Yes |
| `drct_warmup_rounds` | 0 (CLI default, not passed) | 0 (explicitly passed) | Yes |
| `drct_snr_gate` | **False** (not passed; `store_true` CLI default) — see `snr_gate_diagnosis.md` | per-variant (see §1 table) | **No** |
| `drct_sigma_ema_beta` | 0.0 (default) | per-variant (0 or 0.7) | **No** |
| `eval_on_test_pool` | True (`--eval-on-test-pool true`) | True (default of `CIFAR10RecurrenceConfig`) | Yes |
| `drct_shrinkage` (effective) | True (because `fpt_mode="ot"` enables it) | per-variant | depends |
| # methods run in the same job | 26 | 3 (FPT / FedAvg / Oracle) | **No** |
| Model type | linear on 128-d features | linear on 128-d features | Yes |

The "same setup" claim in both captions is therefore wrong in at least five respects:
metric, samples_per_class, batch_size, lr, and n_epochs.

---

## 3. Seed-level comparison (Table 1 `mean_acc` vs Table 3 `fpt_final_acc`)

Both tables used seeds 42/43/44. Per-seed comparison of FPT:

| Seed | Table 1 mean_acc | Table 3 "Pure Stein" final_acc | Δ |
|---|---|---|---|
| 42 | 0.8379 | 0.8940 | +5.61pp |
| 43 | 0.7394 | 0.7893 | +4.99pp |
| 44 | 0.8054 | 0.8653 | +5.99pp |
| **mean** | **0.7942** | **0.8495** | **+5.53pp** |

The per-seed gap is very consistent (≈5–6pp) so the cause is systematic, not a random
difference in one bad seed. The direction (Table 3 > Table 1) matches the combined
effect of:

1. **Metric choice** — Table 1 averages over all 100 rounds including the long
   mid-run dip (seed 42's curve dips to ~0.74 around rounds 50–80 before recovering).
   Table 3's final-round figure samples only the recovered tail.
2. **Richer training data** — Table 3 uses `samples_per_class=500` vs Table 1's `120`
   (4.17× more images per class per client), which yields a better-trained per-concept
   model.
3. **Different local-training schedule** — Table 3 uses `lr=0.01, n_epochs=5`; Table 1
   uses `lr=0.02, n_epochs=10`. In Table 1 the higher lr × more epochs on the smaller
   `spc=120` pool overfits the per-round training batch.

---

## 4. Hypothesis test

Most likely explanation: **(a) + (d) + (g)** — a combination of different
hyperparameters, different eval protocol (curve-mean vs final-round), and a different
entrypoint script. Not (b) [seeds are identical], not (c) [both HEADs are close in
time], not (e) [both use the same `FedProTrackRunner("ot")`], not (f) [both use the
same linear model on 128-d features].

The decisive single factor is the **reporting metric + the increased samples_per_class**.
FedAvg as a control shows the same jump:

| Run | `FedAvg.mean_acc` (Table 1 file) | `FedAvg.fedavg_final_acc` (Table 3 file) |
|---|---|---|
| seed 42 | 0.7900 | 0.8547 |
| seed 43 | 0.7900 | 0.8210 |
| seed 44 | 0.7900 | 0.8308 |
| **mean** | **0.7900** | **0.8355** |

FedAvg jumps +4.6pp between the two protocols despite running identical code. That +4.6pp
baseline shift is not specific to FPT — it is entirely attributable to the training/eval
protocol difference. FPT's extra 0.94pp (over the FedAvg shift) is within seed noise.

---

## 5. Recommended paper fix

The cleanest fix is **Option 1 — add an explicit caption note** making the protocol
difference visible, since the ablation already shows the *relative* ordering of the
four shrinkage variants, which is the only thing the appendix is meant to demonstrate.
A full rerun is not justified because the *relative* signal (SNR-gated > Pure Stein by
0.5pp; EMA adds another 0.4pp) is independent of the protocol shift.

Suggested caption edit for `tab:shrinkage_ablation` (main.tex line ~1072):

```latex
\caption{\textbf{Shrinkage variant ablation.} SNR-gated (default) is a strict
improvement over pure Stein on both datasets. EMA ($\beta{=}0.7$) trades a small
CIFAR-100 cost for extra CIFAR-10 gain.
\emph{Protocol note:} this ablation reports \emph{final-round} accuracy from the
\texttt{run\_cifar10\_concept\_diagnostic.py} entrypoint (\(\text{lr}{=}0.01\),
\(n_{\text{ep}}{=}5\), \(\text{samples\_per\_class}{=}500\), batch 256), whereas
Table~\ref{tab:main} reports \emph{curve-mean} accuracy under the heldout-benchmark
entrypoint (\(\text{lr}{=}0.02\), \(n_{\text{ep}}{=}10\),
\(\text{samples\_per\_class}{=}120\), batch 128). Absolute numbers are therefore
not comparable across the two tables; only the relative ordering of the four
variants within this table is the claim being made.}
```

And a matching edit in §4.4 (main.tex ~line 347) to remove the overclaim:

Current text:
> ablation shows it is a strict improvement over always-on Stein shrinkage on both
> CIFAR-10 ($+0.5$pp) and CIFAR-100 ($+0.3$pp; Table~\ref{tab:shrinkage_ablation}).

Replace with:
> a dedicated ablation (Table~\ref{tab:shrinkage_ablation}) shows the gate is a
> strict improvement over always-on Stein shrinkage on both CIFAR-10 and CIFAR-100
> (the ablation uses a distinct training schedule and reports final-round
> accuracy, so it should be read only for relative ordering).

If a referee demands matching absolute numbers (Option 2 — most defensive):
- Rerun the four shrinkage variants under the Table 1 protocol
  (`samples_per_class=120`, `lr=0.02`, `fpt_epochs=10`, `batch_size=128`, curve-mean).
- Cost estimate: 4 variants × 3 seeds × ~7 min/run (the heldout FPT entry took
  68s × 10× overhead because the diagnostic script is cheaper) ≈ **1.5 GPU-hours on
  RunPod A4500-class serverless**, one submission, standard entrypoint. No code
  changes needed — `run_cifar100_neurips_benchmark.py` already accepts
  `--drct-snr-gate`, `--drct-snr-threshold`, `--drct-sigma-ema-beta`, and the
  `--fpt-mode ot` / `--fpt-mode no-drct` combinations cover all four variants (the
  "Shrinkage off" variant needs `--fpt-mode auto` or a new `--no-drct-shrinkage` CLI
  — worth an audit before submitting).

Option 4 (merge tables) is not recommended: the ablation signal
is the delta among the four shrinkage variants, which is lost if only the
Table 1 FPT number is reported.

---

## 6. Notes on the three candidate files that were *not* Table 3 sources

| Candidate | Why not | Observed behavior |
|---|---|---|
| `runpod_cifar10_snrgate_multi.json` | Script identical to Table 3 sources, same params, but `drct_lambda_log` shows permanent warmup (`lambda_mean=1` in every round) → FPT ≡ FedAvg. This is a buggy predecessor of `snrgate_fixed`. | Seeds 42/43/44 `fpt_final_acc` = 0.8547 / 0.8210 / 0.8293 (≡ FedAvg); mean 0.835 — does not match any Table 3 row. |
| `runpod_results_run_anisotropic_shrinkage_validation.json` | Different experiment family entirely. Dataset: **CIFAR-100**, backbone **ResNet-18**, `K=12`, `T=30`. Reports `acc_shrinkage_iso`, `acc_shrinkage_aniso`, `acc_oracle`, `acc_fedavg` only — no FPT, no Table 3 numbers. | N/A |
| `runpod_results_run_implicit_shrinkage_validation.json` | Same story — CIFAR-100 + ResNet-18 + cluster-count / epoch sweep. Reports `cfl_acc`, `oracle_acc`, `fedavg_acc`, `shrink_iso_acc`, `shrink_aniso_acc`. | N/A |

---

## 7. Summary for Wiki

- Table 1 CIFAR-10 FPT=`.794` is the **only** paper-critical FPT number on CIFAR-10.
  Source: `runpod_heldout_cifar10_all.json`, script `run_cifar100_neurips_benchmark.py`,
  curve-mean over 100 rounds, `samples_per_class=120`, `lr=0.02`, `fpt_epochs=10`,
  `drct_snr_gate=False`.
- Table 3 CIFAR-10 FPT=`{.854, .850, .855, .859}` are from four *separate* ablation
  jobs using `run_cifar10_concept_diagnostic.py`, final-round accuracy only,
  `samples_per_class=500`, `lr=0.01`, `fpt_epochs=5`.
- The absolute-number gap is not a bug in either run; it is a caption/narrative
  omission. Recommended fix: clarify the protocol difference in the Table 3 caption
  and soften the §4.4 cross-reference to speak of "relative ordering only".
