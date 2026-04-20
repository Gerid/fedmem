# FedProTrack Paper — Tables/Figures Data Provenance

**Last updated:** 2026-04-18
**Paper commit at creation:** `55f7900`

This file maps every experimental table/figure in `main.tex` to the raw result files, git commits/tags, and configs that produced its numbers. Per `feedback_paper_version_provenance.md`: every future commit that changes paper numbers MUST update this file atomically with the edit.

**Historical gap (accepted, do NOT reinvestigate):** Some pre-2026-04-17 intermediate result caches and run artifacts were cleared from the workspace before this file was back-filled. The paper numbers in those tables are from genuine runs (not fabricated), but their raw JSON source / commit / tag cannot be retraced at this point. Those cells are marked `UNKNOWN — historical, not retraceable` below. Future agents should NOT spend cycles re-greping for them. The maintenance protocol applies in full to all NEW runs going forward.

Scope of back-fill: `main.tex` at HEAD `55f7900`. Labels surveyed by grepping `\label{(tab|fig):` — 10 total labels (5 tables, 5 figures). Of these, 6 report experimental numbers (tab:main, tab:ablation, tab:full, tab:csweep, tab:misspec, tab:shrinkage_ablation) and 1 figure uses experimental data (fig:concept_count). The other 3 figures (fig:protocol, fig:tradeoff, fig:pipeline) are conceptual illustrations with no data cells and are listed at the end for completeness.

---

## Table 1 (`\label{tab:main}`) — Main accuracy and clustering-error results

**Location in paper:** lines 422–475.
**Cells:** 17 methods × 9 columns (CIFAR-100 × 5 configs, F-MNIST, fMoW, CIFAR-10, plus clustering error η on CIFAR-100).

### Raw data sources (per config)

| Config | Source file(s) | Commit | Tag | Run date (UTC+8) | n_seeds |
|---|---|---|---|---|---|
| CIFAR-100 K20/ρ17 | `runpod_heldout_cifar100_K20_rho17_all.json` | `94a1a2f` | (none — recommend `exp/table1-c100-k20-r17-20260417`) | 2026-04-17 18:07 | 2 (seed 43 RunPod timeout; retry file `runpod_retry_K20rho17_s43.json` returned SSL error and is empty) |
| CIFAR-100 K20/ρ25 | `runpod_flagship_all_methods.json` + `runpod_flagship_seed43_retry.json` | between `18847f5` and `73d6f78` (mtime 15:09/15:23 is before `73d6f78` commit at 15:54; effective HEAD = `18847f5`) | (none) | 2026-04-17 15:09, 15:23 | 3 |
| CIFAR-100 K20/ρ33 | `runpod_heldout_cifar100_K20_rho33_all.json` | `94a1a2f` | (none) | 2026-04-17 18:22 | 3 |
| CIFAR-100 K40/ρ25 | `runpod_heldout_cifar100_K40_rho25_all.json` | `94a1a2f` | (none) | 2026-04-17 18:41 | 3 |
| CIFAR-100 K40/ρ33 | `runpod_heldout_cifar100_K40_rho33_all.json` (seed 42 only) + `runpod_retry_K40rho33_s43s44.json` (seeds 43, 44) | `94a1a2f` (seed 42) + retry at later HEAD (UNKNOWN — flag for investigation, likely `4e9cfd6` or `0752a94`) | (none) | 2026-04-17 20:00 + retry | 3 |
| F-MNIST K20/ρ25 | `runpod_heldout_fmnist_all.json` | `94a1a2f` | (none) | 2026-04-17 20:21 | 3 |
| fMoW K20/ρ25 | See §"fMoW caveat" below | UNKNOWN — flag for investigation (pre-held-out-rewrite commit) | (none) | pre-2026-04-16 | 3 |
| CIFAR-10 K20/ρ25 | `runpod_heldout_cifar10_all.json` | `94a1a2f` | (none) | 2026-04-17 20:11 | 3 |

### Config summary (applies to all cells except fMoW)

- Entry script: `run_cifar100_neurips_benchmark.py` (dispatches to CIFAR-10 / F-MNIST / CIFAR-100 via `_DATASET_DISPATCH`).
- `fpt_mode = "ot"`
- `K = 20` or `40`, `T = 100`, `ρ` per config, `alpha = 0.75`, `delta = 0.85`.
- `fpt_lr = 0.02`, `fpt_epochs = 10`, `batch_size = 128`.
- `n_features = 128`, `n_samples = 400`, `samples_per_coarse_class = 120`.
- `feature_seed = 2718`, seeds `{42, 43, 44}`.
- `federation_every = 1`.
- **`drct_snr_gate = False`** (CLI omitted `--drct-snr-gate`; the flag is `store_true`, so CLI default = False overrides the dataclass default of `True`).
- **`drct_warmup_rounds = 0`** (CLI default).
- **`drct_sigma_ema_beta = 0.0`** (CLI default).
- **`drct_snr_threshold = 1.0`** (CLI default, inert when gate is off).
- **`eval_on_test_pool = True`** (80/20 stratified held-out split).
- Reported metric per cell: `mean_acc` — time-averaged held-out accuracy over 100 rounds.
- Clustering error η column: symmetric clustering error on CIFAR-100 (averaged across the 5 configs); source is the same set of JSONs.

### Known caveats

- **SNR gate was OFF in Table 1.** This contradicts the paper narrative in §4.4 and §5.2, which claims the gate is the default. See `snr_gate_diagnosis.md` for the three independent lines of evidence (JSON config blocks, dataclass-vs-CLI defaults at runtime commits, and submission-command audit). Narrative revision pending — either (a) soften paper text to describe the gate as an optional extension studied in `tab:shrinkage_ablation`, or (b) re-run Table 1 with `--drct-snr-gate` set explicitly.
- **K20/ρ17 has n=2.** Seed 43 failed with a RunPod timeout; the retry file `runpod_retry_K20rho17_s43.json` contains only an SSL error and no method data. Pre-flight artifacts for a further re-submission are documented in `runpod_heldout_cifar100_K20_rho17_seed43_PROVENANCE.md`. Paper footnote `^\dagger` correctly flags this.
- **fMoW is a different protocol.** Prequential-on-same-pool evaluation (not held-out 80/20). Numbers preserved from an earlier paper draft. Appendix `app:fmow_protocol` documents this at a high level; exact source file at HEAD is UNKNOWN — flag for investigation (candidates: `runpod_fmow_paper_config.json`, `runpod_fmow_paper_oldeval.json`, `runpod_fmow_paper_62class.json`, `runpod_paper_fmow_final.json`). Footnoted `^\ddagger` in the paper caption.
- **Config JSON blocks are incomplete.** The 23-field "classic" dump saved alongside every file omits all `drct_*` fields and `eval_on_test_pool`. Values above were inferred forensically per `snr_gate_diagnosis.md`. Future runs must dump a complete resolved config (reproducibility 5-point checklist).
- **Gap-closed row caveat.** The "FPT gap closed" row uses a ratio-of-means estimator. Per-seed mean-of-ratios with bootstrap CIs are given in `stats_recomputation.md` §3; three configs (K20/ρ25, K40/ρ25, CIFAR-10) have CIs that span 100% due to seed-43 denominator instability.
- **Std cells are pre-recomputation.** Several `±.00` and `±.01` cells in the paper understate the true sample std with ddof=1. Full per-cell recomputation and recommended LaTeX edits are in `stats_recomputation.md` §1 and §5.

---

## Table 2 (`\label{tab:ablation}`) — Component ablation on CIFAR-100

**Location in paper:** lines 508–524.
**Cells:** 5 variants × 2 metrics (accuracy, η) on CIFAR-100 K40/ρ33, 3 seeds.

### Raw data sources

| Variant | Source file | Commit | Tag | Run date | n_seeds |
|---|---|---|---|---|---|
| FPT (full) | UNKNOWN — flag for investigation | UNKNOWN | (none) | UNKNOWN | 3 |
| − Concept discovery (single group) | UNKNOWN — flag for investigation | UNKNOWN | (none) | UNKNOWN | 3 |
| − Adaptive shrinkage | UNKNOWN — flag for investigation | UNKNOWN | (none) | UNKNOWN | 3 |
| − Fingerprint clustering (use weights) | UNKNOWN — flag for investigation | UNKNOWN | (none) | UNKNOWN | 3 |
| − Eigengap (fixed Ĉ = C) | UNKNOWN — flag for investigation | UNKNOWN | (none) | UNKNOWN | 3 |

### Config summary (from paper caption — explicitly differs from Table 1)

- Dataset: CIFAR-100, K=40, ρ=33, 3 seeds.
- `lr = 0.01`, `n_epochs = 5` (vs Table 1's `0.02 / 10`).
- Evaluation protocol: **prequential** (vs Table 1's held-out 80/20). Caption explicitly warns absolute numbers are not comparable to Table 1.

### Known caveats

- **Source JSON files not yet identified at paper commit `55f7900`.** Table 2 pre-dates the held-out-evaluation rewrite; the corresponding runs likely live in one of the older `runpod_paper_cifar100_*.json`, `runpod_paper_cifar100_K40_rho33.json`, or ablation-specific files, but we did not confirm this during provenance back-fill. Flag for investigation: locate the five per-variant JSONs or rerun under the claimed prequential protocol and record here.
- **Absolute numbers not comparable to Table 1.** The caption already states this; cross-references in §5 rely on *relative* deltas only.

---

## Table 3 (`\label{tab:full}`) — Full 26-method held-out ranking (Appendix)

**Location in paper:** lines 787–826.
**Cells:** 26 methods × 3 dataset columns (CIFAR-100 cross-config mean, F-MNIST, CIFAR-10).

### Raw data sources

Per-row-per-column numbers are computed from the **same 7 held-out JSONs** as Table 1, extended to include the 9 methods that Table 1 relegates to this appendix (Adaptive-FedAvg, ATP, CompressedFedAvg, FedAvg-FPTTrain, FedCCFA-Impl, FedGWC, HCFL, pFedMe, TrackedSummary).

| Column | Source files |
|---|---|
| CIFAR-100 (cross-config mean of 5 configs) | `runpod_heldout_cifar100_K20_rho17_all.json`, `runpod_flagship_all_methods.json` + `runpod_flagship_seed43_retry.json`, `runpod_heldout_cifar100_K20_rho33_all.json`, `runpod_heldout_cifar100_K40_rho25_all.json`, `runpod_heldout_cifar100_K40_rho33_all.json` + `runpod_retry_K40rho33_s43s44.json` |
| F-MNIST (K=20, ρ=25) | `runpod_heldout_fmnist_all.json` |
| CIFAR-10 (K=20, ρ=25) | `runpod_heldout_cifar10_all.json` |

### Config summary

Identical to Table 1 (see above). Commit / run-date / seed table is the same — see Table 1 §"Raw data sources".

### Known caveats

- **fMoW column omitted** (caption explicitly notes: "different protocol; see Table~\ref{tab:main}").
- **Inherits all Table 1 caveats**: SNR gate off, truncated config dumps, K20/ρ17 n=2.
- **Per-method ± is not reported in this appendix table** — the ranking is by point estimate. If reviewers ask, std can be recomputed per §5.5 of `stats_recomputation.md` ("paper's full 26-method list").

---

## Table 4 (`\label{tab:csweep}`) — Concept-count sensitivity

**Location in paper:** lines 987–1008 (Appendix `app:concept_count`).
**Cells:** 8 methods × 4 concept counts (C ∈ {2, 4, 6, 8}).

### Raw data sources

| C | Source file | Commit | Tag | Run date | n_seeds |
|---|---|---|---|---|---|
| C = 8 (ρ = 3.75) | `runpod_csweep_ot_rho3.75.json` or `runpod_results_csweep_paper_rho3.75.json` — UNKNOWN which is canonical, flag for investigation | UNKNOWN | (none) | UNKNOWN | 3 |
| C = 6 (ρ = 5.0) | `runpod_csweep_ot_rho5.0.json` or `runpod_results_csweep_paper_rho5.0.json` — UNKNOWN which is canonical | UNKNOWN | (none) | UNKNOWN | 3 |
| C = 4 (ρ = 7.5) | `runpod_csweep_ot_rho7.5.json` or `runpod_results_csweep_paper_rho7.5.json` — UNKNOWN which is canonical | UNKNOWN | (none) | UNKNOWN | 3 |
| C = 2 (ρ = 15.0) | `runpod_csweep_ot_rho15.0.json` or `runpod_results_csweep_paper_rho15.0.json` — UNKNOWN which is canonical | UNKNOWN | (none) | UNKNOWN | 3 |

### Config summary (from paper text lines 533–535 and caption)

- Dataset: CIFAR-100 disjoint labels.
- `K = 20`, `T = 30` (vs Table 1's `T = 100`), 3 seeds.
- `fpt_mode = "ot"` (confirmed by `runpod_csweep_ot_*.json` filenames).
- Concept count swept by varying recurrence period ρ ∈ {3.75, 5.0, 7.5, 15.0} to induce C ∈ {8, 6, 4, 2}.
- Reported metric: **mean final accuracy** (final-round accuracy, not curve mean).

### Known caveats

- **Two candidate file families exist** (`runpod_csweep_ot_*` vs `runpod_results_csweep_paper_*`). The canonical one for the paper numbers in `tab:csweep` has not been confirmed at commit `55f7900`. Flag for investigation: open one file per ρ from each family and verify which matches the LaTeX numbers.
- **Different protocol from Table 1** — T=30 and final-round metric. Absolute numbers not comparable to Tables 1/3.

---

## Table 5 (`\label{tab:misspec}`) — Misspecified cluster count

**Location in paper:** lines 1034–1055 (Appendix `app:misspecified_c`).
**Cells:** 8 methods × 5 misspecified Ĉ values (Ĉ ∈ {2, 3, 4, 6, 8}), with true C = 4.

### Raw data sources

| Ĉ | Source file | Commit | Tag | Run date | n_seeds |
|---|---|---|---|---|---|
| Ĉ = 2 | `runpod_misspec_ot_nc2.json` and/or `runpod_results_misspec_paper_nc2.json` — UNKNOWN which is canonical | UNKNOWN | (none) | UNKNOWN | 3 |
| Ĉ = 3 | `runpod_misspec_ot_nc3.json` and/or `runpod_results_misspec_paper_nc3.json` | UNKNOWN | (none) | UNKNOWN | 3 |
| Ĉ = 4 | `runpod_misspec_ot_nc4.json` and/or `runpod_results_misspec_paper_nc4.json` | UNKNOWN | (none) | UNKNOWN | 3 |
| Ĉ = 6 | `runpod_misspec_ot_nc6.json` and/or `runpod_results_misspec_paper_nc6.json` | UNKNOWN | (none) | UNKNOWN | 3 |
| Ĉ = 8 | `runpod_misspec_ot_nc8.json` and/or `runpod_results_misspec_paper_nc8.json` | UNKNOWN | (none) | UNKNOWN | 3 |

### Config summary (from paper lines 1031–1032)

- Dataset: CIFAR-100, disjoint labels, true C=4.
- `K = 20`, `T = 30`, `d = 128`, 3 seeds.
- Misspecified Ĉ is given to the cluster-aware baselines (FLUX-prior, IFCA, FeSEM, FedRC, FedEM); FPT, Oracle, FedAvg, CFL run unchanged.

### Known caveats

- **Two candidate file families**, same ambiguity as Table 4 — flag for investigation.
- **FeSEM row equals IFCA row exactly** (`.530 / .519 / .503 / .490 / .490`). This matches the CLAUDE.md "FeSEM remains an alias of IFCA" note and is intentional, not a copy-paste error.

---

## Table 6 (`\label{tab:shrinkage_ablation}`) — Shrinkage variant ablation

**Location in paper:** lines 1070–1085 (Appendix `app:shrinkage_ablation`).
**Cells:** 4 variants × 2 datasets (CIFAR-10, CIFAR-100).

### Raw data sources (CIFAR-10 column — confirmed per `table1_vs_table3_diagnosis.md` §1)

| Variant | CIFAR-10 reported | CIFAR-10 source | Commit | Tag | Run date | n_seeds |
|---|---|---|---|---|---|---|
| Shrinkage off (λ̂=0) | `.854` | `runpod_cifar10_noshrink_multi.json` | UNKNOWN — flag for investigation (diagnostic-script family, pre-held-out commit) | (none) | UNKNOWN | 3 |
| Pure Stein | `.850` | `runpod_cifar10_shrink_fixed.json` | UNKNOWN | (none) | UNKNOWN | 3 |
| SNR-gated (default) | `.855` | `runpod_cifar10_gateonly.json` | UNKNOWN | (none) | UNKNOWN | 3 |
| SNR-gated + EMA (β=0.7) | `.859` | `runpod_cifar10_snrgate_fixed.json` | UNKNOWN | (none) | UNKNOWN | 3 |

### Raw data sources (CIFAR-100 column)

| Variant | CIFAR-100 reported | Source file | Commit | Tag | Run date | n_seeds |
|---|---|---|---|---|---|---|
| Shrinkage off (λ̂=0) | `---` (not reported) | n/a | — | — | — | — |
| Pure Stein | `.612` | UNKNOWN — flag for investigation (candidates: `runpod_cifar100_stein.json`) | UNKNOWN | (none) | UNKNOWN | 3 |
| SNR-gated (default) | `.615` | UNKNOWN — flag for investigation (candidates: `runpod_cifar100_gated.json`, `runpod_cifar100_gateonly.json`) | UNKNOWN | (none) | UNKNOWN | 3 |
| SNR-gated + EMA (β=0.7) | `.608` | UNKNOWN — flag for investigation | UNKNOWN | (none) | UNKNOWN | 3 |

### Config summary (DIFFERS from Table 1 — per `table1_vs_table3_diagnosis.md` §2)

- Entry script: `run_cifar10_concept_diagnostic.py` (not the benchmark script that Table 1 uses).
- `K = 20`, `T = 100`, `ρ = 25`, `alpha = 0.75`, `delta = 0.85` (same as Table 1).
- `samples_per_class = 500` (vs Table 1's `120`) — 4.17× more training data per client.
- `batch_size = 256` (vs Table 1's `128`).
- `fpt_lr = 0.01` (vs Table 1's `0.02`).
- `fpt_epochs = 5` (vs Table 1's `10`).
- `drct_shrinkage`, `drct_snr_gate`, `drct_sigma_ema_beta` vary per row (see table above).
- Reported metric: **final-round accuracy** (`fpt_final_acc`), not time-averaged curve mean.

### Known caveats

- **Absolute numbers NOT comparable to Table 1.** FedAvg baseline on CIFAR-10: `.757` (Table 1) vs `.835` (this ablation's FedAvg control) — a ~4.6pp baseline shift attributable solely to the protocol (samples_per_class, lr, n_epochs, final-round vs curve-mean). This ablation only supports the *relative* ordering of the four shrinkage variants. See `table1_vs_table3_diagnosis.md` §3–§5 for the decomposition and a proposed caption edit.
- **Dead-end candidate files.** `runpod_cifar10_snrgate_multi.json` has a stuck-in-warmup bug (`lambda_mean=1` every round → FPT ≡ FedAvg); it is **not** a valid source. `runpod_results_run_anisotropic_shrinkage_validation.json` and `runpod_results_run_implicit_shrinkage_validation.json` are CIFAR-100+ResNet-18 experiments, unrelated to this table.
- **Commits + run dates UNKNOWN** for all 7 source JSONs — flag for investigation.

---

## Figure (`\label{fig:concept_count}`) — Concept-count sensitivity plot

**Location in paper:** lines 1010–1016.

Plot of data from `tab:csweep` (panel a) and `tab:misspec` (panel b). Source PDF: `figures/concept_count_sensitivity.pdf`. Underlying data sources: inherit from Tables 4 and 5 above (UNKNOWN — flag for investigation).

---

## Non-data figures (listed for completeness)

| Label | Location | Type | Data? |
|---|---|---|---|
| `fig:protocol` | line 77–82 | Scheme A vs Scheme C illustration | No — conceptual diagram. PNG: `figures/paperbanana/scheme_comparison_candidate_0.png`. Only quantitative label is `η_A ≥ 0.30` / `η_C = 0.003` which is a cross-reference to `tab:main`. |
| `fig:tradeoff` | line 162–167 | Bias-variance crossover illustration | Partial — "validated on 609 synthetic configurations". Synthetic-data source / generator script UNKNOWN — flag for investigation. PNG: `figures/paperbanana/research_question_candidate_0.png`. |
| `fig:pipeline` | line 246–250 | 4-module pipeline diagram | No — conceptual. |

---

## Maintenance protocol

Going forward (per `feedback_paper_version_provenance.md`):

1. Every commit that changes experimental numbers in `main.tex` MUST update this file in the same commit.
2. Before running a new paper-critical experiment, create git tag `exp/<table>-<config>-<date>` and record it in the appropriate section's "Tag" column.
3. New tables added to paper: add a section here before committing the paper edit.
4. When a table's source file is retired (e.g., replaced by a rerun), leave the old entry with a `SUPERSEDED` note (date + replacement file) rather than deleting — `git log -p -- TABLES_PROVENANCE.md` should tell the full history.
5. When an `UNKNOWN — flag for investigation` entry is resolved, replace it with the confirmed value in the same commit that uses the resolution (do not leave orphaned UNKNOWNs once known).

## UNKNOWN entries (follow-up queue)

Concise list for quick triage — each points to the section above for context:

- Table 1 fMoW column: source JSON file at HEAD, commit, run date.
- Table 1 K40/ρ33 retry (`runpod_retry_K40rho33_s43s44.json`): commit at which retry was submitted.
- Table 2 (all 5 rows): source JSONs, commits, run dates. Paper-commit-55f7900 did not preserve this linkage.
- Table 4 (csweep) and Table 5 (misspec): canonical file family (`runpod_csweep_ot_*` vs `runpod_results_csweep_paper_*`; same for misspec).
- Table 6 CIFAR-10 column (4 files): commits and run dates.
- Table 6 CIFAR-100 column (3 cells): source JSONs, commits, run dates.
- fig:tradeoff: 609-configuration synthetic-sweep generator script and output file.
