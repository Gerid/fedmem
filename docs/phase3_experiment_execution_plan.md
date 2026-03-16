# FedProTrack Phase 3 Experiment Execution Plan

## Purpose

This document turns the Phase 3 experiment design into an execution checklist
that matches the current repository layout and scripts. The target paper claim
is not generic SOTA chasing. The experiment package must answer:

> When does memory help under federated concept drift?

The evidence chain is:

1. FedProTrack identifies latent concept identity better than baselines.
2. Better identity inference enables useful memory reuse.
3. Memory is not uniformly helpful; there is a regime boundary.
4. The two-phase communication protocol changes the budget frontier.
5. The phase diagram remains the signature result under matched-budget
   comparison.

## Current Code Anchors

- Main experiment entry: `run_phase3_experiments.py`
- Base experiment grid: `run_experiments.py`
- FedProTrack runner: `fedprotrack/posterior/fedprotrack_runner.py`
- Two-phase protocol: `fedprotrack/posterior/two_phase_protocol.py`
- Gibbs posterior: `fedprotrack/posterior/gibbs.py`
- Dynamic memory bank: `fedprotrack/posterior/memory_bank.py`
- Budget analysis: `fedprotrack/experiments/budget_analysis.py`
- Ablations: `fedprotrack/experiments/ablations.py`
- Figures: `fedprotrack/experiments/figures.py`
- Tables: `fedprotrack/experiments/tables.py`

## Global Protocol

### Fixed settings

- Python invocation: `conda run -n base python`
- Development grid: `K=5, T=8, n_samples=200`
- Main grid: `K=10, T=20, n_samples=500`
- Main seeds: `42,123,456,789,1024`
- Development seeds: `42,123,456`
- Main generator order: `sine`, then `sea`, then `circle`
- Primary baseline set:
  - `FedAvg`
  - `FedProto`
  - `TrackedSummary`
  - `Flash`
  - `FedDrift`
  - `IFCA`
  - `FedProTrack`
- Upper bound only:
  - `Oracle`
- Appendix only:
  - `LocalOnly`
  - `CompressedFedAvg`

### Core metrics

- `concept_re_id_accuracy`
- `assignment_entropy`
- `wrong_memory_reuse_rate`
- `worst_window_dip`
- `worst_window_recovery`
- `budget_normalized_score`
- `AUC(acc(t))`
- `final accuracy`
- `win rate`
- `mean rank`

### Fairness rules

- Same generator configuration for all methods
- Same seed suite
- Same local training protocol
- Same client participation protocol
- Same total-byte accounting for budget comparisons
- Same reporting windows for drift dip and recovery

## Output Layout

All Phase 3 outputs should stay under `results_phase3/`.

### Required directories

- `results_phase3/summary.json`
- `results_phase3/tables/main_table.tex`
- `results_phase3/tables/table_rho.tex`
- `results_phase3/tables/table_alpha.tex`
- `results_phase3/tables/table_delta.tex`
- `results_phase3/tables/appendix/`
- `results_phase3/figures/phase_diagrams/`
- `results_phase3/figures/memory_phase/`
- `results_phase3/figures/budget/`
- `results_phase3/figures/ablations/`
- `results_phase3/figures/theory/`
- `results_phase3/figures/scalability/`
- `results_phase3/logs/`

## Experiment Matrix

### E0. Controlled drift generator

Status:
- Implemented for `sine`, `sea`, `circle`
- Needs explicit noise-axis support and a clean `rho=inf` surrogate if used in
  the paper

Goal:
- Provide ground-truth concept identity and recurrence structure for all
  downstream experiments

Primary settings:
- `rho in {2, 5, 10}`
- `delta in {0.1, 0.3, 0.5, 0.7, 1.0}`
- `alpha in {0, 0.25, 0.5, 0.75, 1.0}`
- Optional `eta in {0.0, 0.1, 0.2}` after the main package is stable

Outputs:
- Concept matrix visualization
- Ground-truth concept matrix
- Per-setting config JSON

Implementation tasks:
- Add noise-axis support to generator config if absent
- Add a figure wrapper for representative concept matrix galleries

### E1. Identity inference quality

Scientific claim:
- FedProTrack helps because it infers concept identity more accurately, not
  because it gets lucky on end accuracy

Primary grid:
- Generator: `sine`
- Sweep 1: `delta in {0.1, 0.3, 0.5, 0.7, 1.0}` with `rho=5`, `alpha=0.5`
- Sweep 2: `rho in {2, 5, 10}` with `delta=0.5`, `alpha=0.5`
- Sweep 3: vary number of active clients per concept if generator support is
  exposed

Methods:
- `FedAvg`
- `FedProto`
- `TrackedSummary`
- `FedDrift`
- `IFCA`
- `Flash`
- `FedProTrack`

Metrics:
- `concept_re_id_accuracy`
- `assignment_entropy`
- `wrong_memory_reuse_rate`
- per-timestep `re-ID`

Primary artifacts:
- `results_phase3/figures/memory_phase/reid_vs_delta.png`
- `results_phase3/figures/memory_phase/wrong_memory_vs_delta.png`
- `results_phase3/figures/memory_phase/assignment_entropy_heatmap.png`

Current code coverage:
- Metrics already exist
- Phase-diagram figure helper already exists
- Missing a direct line-plot wrapper for `re-ID vs delta` and
  `wrong-memory vs delta`

Required code changes:
- Add a small helper in `fedprotrack/experiments/figures.py` for axis-sweep
  line plots with multiple methods
- Extend `run_phase3_experiments.py` to save dedicated E1 summary figures

Command template:

```powershell
conda run -n base python run_phase3_experiments.py `
  --results-dir results_phase3/e1_identity `
  --generators sine `
  --seeds 42,123,456,789,1024 `
  --skip-ablation `
  --skip-scalability
```

Result sentence template:
- "FedProTrack improves concept re-identification before improving downstream
  accuracy, indicating that memory reuse becomes useful only after identity is
  inferred reliably."

### E2. Drift-window adaptation speed

Scientific claim:
- The temporal prior improves recovery under recurring asynchronous drift

Primary grid:
- Generator: `sine`
- Fixed `rho=5`, `delta=0.5`
- Sweep `alpha in {0, 0.25, 0.5, 0.75, 1.0}`

Methods:
- `FedAvg`
- `Flash`
- `FedDrift`
- `IFCA`
- `FedProTrack`

Metrics:
- `worst_window_dip`
- `worst_window_recovery`
- `AUC(acc(t))`
- `final accuracy`

Primary artifacts:
- `results_phase3/figures/memory_phase/accuracy_over_time_alpha_*.png`
- `results_phase3/figures/memory_phase/dip_recovery_boxplot.png`
- `results_phase3/figures/memory_phase/recovery_vs_alpha.png`

Required code changes:
- Add recovery-vs-axis plotting helper
- Extend main table support to include `worst_window_dip` and
  `worst_window_recovery`

Command template:

```powershell
conda run -n base python run_phase3_experiments.py `
  --results-dir results_phase3/e2_recovery `
  --generators sine `
  --seeds 42,123,456,789,1024 `
  --skip-ablation `
  --skip-scalability
```

Result sentence template:
- "The transition prior lowers worst-case drift dip and shortens recovery time,
  with the largest gain in asynchronous recurring drift."

### E3. Memory benefit phase diagram

Scientific claim:
- Memory has a regime boundary; it is helpful only when recurrence and
  separability are jointly favorable

Primary grid:
- Generator: `sine`
- `rho in {2, 5, 10}`
- `delta in {0.1, 0.3, 0.5, 0.7, 1.0}`
- Slice by `alpha in {0, 0.5, 1.0}`

Comparisons:
- `FedProTrack - FedAvg`
- `FedProTrack - FedProto`
- `FedProTrack - IFCA`

Primary artifacts:
- `results_phase3/figures/memory_phase/memory_gain_vs_fedavg_alpha0.png`
- `results_phase3/figures/memory_phase/memory_gain_vs_fedavg_alpha05.png`
- `results_phase3/figures/memory_phase/memory_gain_vs_fedavg_alpha1.png`
- Same set for `FedProto` and `IFCA`

Current code coverage:
- Existing phase diagrams visualize one method and one metric
- Missing difference heatmaps between methods

Required code changes:
- Add a difference-heatmap function that accepts two method result grids
- Extend `run_phase3_experiments.py` to aggregate by fixed `alpha` slices, not
  only `alpha=0.5`

Command template:

```powershell
conda run -n base python run_phase3_experiments.py `
  --results-dir results_phase3/e3_memory_phase `
  --generators sine `
  --seeds 42,123,456,789,1024 `
  --skip-ablation `
  --skip-scalability
```

Result sentence template:
- "Memory is harmful in low-separability regions, but becomes dominant when
  recurrence is frequent and concepts are distinguishable."

### E4. Budget-regime crossover

Scientific claim:
- The two-phase protocol changes which method is best under a fixed total-byte
  budget

Primary grid:
- Generator: `sine`
- Fixed `rho=5`, `delta=0.5`
- Sweep `alpha in {0, 0.25, 0.5, 0.75, 1.0}`
- Budget control via `federation_every in {1, 2, 5, 10}` and compressed baselines

Methods:
- `FedAvg`
- `FedProto`
- `TrackedSummary`
- `FedDrift`
- `IFCA`
- `CompressedFedAvg`
- `FedProTrack`

Metrics:
- `budget_normalized_score`
- `AUC(acc(t))`
- `total_bytes`

Primary artifacts:
- `results_phase3/figures/budget/budget_frontier.png`
- `results_phase3/figures/budget/budget_alpha_best_method.png`
- `results_phase3/figures/budget/crossover_curve.png`
- `results_phase3/tables/appendix/overhead_budget_table.tex`

Current code coverage:
- One-dimensional budget frontier exists
- Missing best-method heatmap across budget and `alpha`

Required code changes:
- Add a budget x alpha heatmap builder
- Extend logging to expose `phase_a_bytes` and `phase_b_bytes` in table-ready
  form

Command template:

```powershell
conda run -n base python run_phase3_experiments.py `
  --results-dir results_phase3/e4_budget `
  --generators sine `
  --seeds 42,123,456,789,1024 `
  --skip-ablation `
  --skip-scalability
```

Result sentence template:
- "FedProTrack is not simply more expensive; it becomes the most
  budget-efficient method in the mid-budget, medium-to-high asynchrony regime."

### E5. End-to-end synthetic main table

Scientific claim:
- FedProTrack wins consistently across drift regimes, not only on a single
  benchmark point

Primary grid:
- Generators: `sine,sea,circle`
- Main `rho, alpha, delta` grid
- Main five-seed suite

Methods:
- `FedAvg`
- `FedProto`
- `TrackedSummary`
- `Flash`
- `FedDrift`
- `IFCA`
- `FedProTrack`
- `Oracle` as reference row

Metrics for the paper table:
- `final accuracy`
- `AUC(acc(t))`
- `concept_re_id_accuracy`
- `wrong_memory_reuse_rate`
- `budget_normalized_score`
- `mean rank`
- `win rate`

Primary artifacts:
- `results_phase3/tables/main_table.tex`
- `results_phase3/tables/table_rho.tex`
- `results_phase3/tables/table_alpha.tex`
- `results_phase3/tables/table_delta.tex`
- `results_phase3/tables/appendix/main_table_with_oracle.tex`

Current code coverage:
- Mean-plus-std tables exist
- Missing `mean rank`, `win rate`, and `final accuracy` columns

Required code changes:
- Extend `fedprotrack/experiments/tables.py` with summary ranking helpers
- Add a JSON/CSV summary export for table post-processing

Command template:

```powershell
conda run -n base python run_phase3_experiments.py `
  --results-dir results_phase3/e5_main `
  --generators sine,sea,circle `
  --seeds 42,123,456,789,1024
```

Result sentence template:
- "FedProTrack ranks first on average across the synthetic regime grid rather
  than depending on isolated easy settings."

### E6. Real or semi-real consistency

Scientific claim:
- The phase-boundary direction is not a synthetic artifact

Priority order:
1. `Rotating MNIST`
2. `Label-swapping MNIST or EMNIST`
3. `Rotating CIFAR`
4. `FMoW-Time` only if compute remains

Methods:
- `FedAvg`
- `FedProto`
- `FedDrift or IFCA`
- `FedProTrack`

Metrics:
- `AUC(acc(t))`
- `final accuracy`
- `budget_normalized_score`
- concept-identity proxy metrics if labels or synthetic transformation IDs make
  them measurable

Primary artifacts:
- `results_phase3/figures/real/realdata_phase_direction.png`
- `results_phase3/tables/realdata_main_table.tex`

Required code changes:
- Add real-data loaders and experiment entry points
- Reuse the same metric/reporting protocol wherever concept truth is available

Blocking note:
- This is the main missing subsystem in the current repository

Result sentence template:
- "The direction of the memory-helpful boundary remains consistent on image
  streams, supporting the claim that the phase diagram captures a general
  phenomenon."

### E7. Module ablations

Scientific claim:
- The method works because of the combination of posterior assignment, temporal
  prior, dynamic memory, and two-phase communication

Required ablations:
1. No temporal prior
2. Hard assignment instead of Gibbs posterior
3. No dynamic memory bank
4. No spawn/merge
5. Fixed `omega` versus adaptive `omega`
6. Phase A only
7. Phase B only
8. Distance-function ablation

Primary artifacts:
- `results_phase3/figures/ablations/ablation_prior.png`
- `results_phase3/figures/ablations/ablation_assignment.png`
- `results_phase3/figures/ablations/ablation_memory.png`
- `results_phase3/figures/ablations/ablation_protocol.png`

Current code coverage:
- Existing ablation file only sweeps scalar hyperparameters

Required code changes:
- Introduce boolean and enum ablation modes into `TwoPhaseConfig` or a wrapper
  experiment config
- Expand `run_ablation_study()` to support module toggles

Command template:

```powershell
conda run -n base python run_phase3_experiments.py `
  --results-dir results_phase3/e7_ablations `
  --generators sine `
  --seeds 42,123,456
```

Result sentence template:
- "Removing any one of the posterior, temporal, memory, or protocol modules
  degrades either identity inference, wrong-memory control, or budget
  efficiency."

### E8. Theory-linked validation

Scientific claim:
- The empirical trends follow the theorem-level conditions and boundaries

Sub-experiments:

#### E8.1 TT1 identifiability

Grid:
- `delta`
- active clients per concept
- recurrence strength

Artifact targets:
- `results_phase3/figures/theory/tt1_reid_vs_delta.png`
- `results_phase3/figures/theory/tt1_reid_vs_clients.png`

#### E8.2 TT2 memory contraction

Requirements:
- Need logged distance between current memory bank and ground-truth concept
  summaries

Artifact targets:
- `results_phase3/figures/theory/tt2_memory_distance_vs_round.png`
- `results_phase3/figures/theory/tt2_empirical_vs_theoretical_gamma.png`

#### E8.3 TT3 memory-help boundary

Requirements:
- Need logged decomposition terms for misassignment cost, reuse gain, and
  approximation error

Artifact targets:
- `results_phase3/figures/theory/tt3_boundary.png`
- `results_phase3/figures/theory/tt3_misassign_vs_gain.png`

Required code changes:
- Add logging hooks for memory-bank summary distance
- Add decomposition utilities for empirical misassignment cost and reuse gain

Result sentence template:
- "The empirical boundary follows the same direction as the theorem: memory is
  helpful only when reuse gain dominates misassignment-induced bias."

### E9. Overhead and reproducibility

Scientific claim:
- The comparisons are fair and reproducible

Metrics:
- `total_bytes`
- `phase_a_bytes`
- `phase_b_bytes`
- `wall_clock_time`
- `active_concepts`
- `spawned_concepts`
- `merged_concepts`
- `pruned_concepts`
- `rounds_to_target_accuracy`

Primary artifacts:
- `results_phase3/tables/appendix/overhead_table.tex`
- `results_phase3/figures/budget/system_cost_breakdown.png`

Required code changes:
- Extend `FedProTrackResult` or experiment logs to include system counters
- Add a dedicated appendix table generator

Result sentence template:
- "All methods are compared under matched budget and shared training protocol,
  with explicit byte and runtime accounting."

## Implementation Checklist

### P0: minimal publishable package

1. Extend `run_phase3_experiments.py` to save per-axis E1 figures
2. Add difference heatmaps for E3
3. Extend table generation with `final accuracy`, `mean rank`, and `win rate`
4. Add phase-A and phase-B byte reporting to budget outputs
5. Freeze one final synthetic grid and one final seed suite

### P1: strong paper package

1. Upgrade ablations from scalar sweeps to module toggles
2. Add budget x alpha best-method heatmap
3. Add overhead appendix table
4. Add theory logging hooks for TT2 and TT3

### P2: generalization package

1. Add one real or semi-real dataset runner
2. Rebuild main paper plots on image data
3. Add optional noise-axis study

## Recommended Execution Order

1. Run quick synthetic sanity:

```powershell
conda run -n base python run_phase3_experiments.py --quick --results-dir results_phase3/quick
```

2. Run final E1 plus E3 synthetic sweep on `sine`
3. Freeze FedProTrack hyperparameters
4. Run final E4 budget study
5. Run full E5 main table on `sine,sea,circle`
6. Run E7 ablations
7. Run E9 overhead export
8. Only then decide whether E6 or E8 fits the remaining time budget

## Merge Gate Before Paper Results

Do not treat the experiment package as frozen until all of the following are
true:

- Main seeds are fixed
- Main grid is fixed
- Baseline list is fixed
- Total-byte accounting is frozen
- Figure filenames are frozen
- Table column order is frozen
- All tests pass with:

```powershell
conda run -n base python -m pytest tests/ -v
```

## Immediate Next Coding Tasks

1. Add experiment-summary ranking helpers in `fedprotrack/experiments/tables.py`
2. Add method-difference heatmaps in `fedprotrack/experiments/figures.py`
3. Extend `run_phase3_experiments.py` to emit E1, E3, and E4 dedicated outputs
4. Extend `FedProTrackResult` logging for byte breakdown and concept lifecycle
5. Upgrade `fedprotrack/experiments/ablations.py` from scalar-only sweeps to
   module ablations

