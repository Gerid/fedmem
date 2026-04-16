# Autoresearch Prompt — Copy This to Start

Resume the autoresearch project. `research-state.yaml`, `findings.md`, and `research-log.md` are already initialized at the project root.

## Context

This is a **NeurIPS 2026 theory paper** about when concept-level aggregation helps in federated learning. The paper + extensions PDF are complete (see `paper/main.tex` and the extensions supplement). We need to **run 5 targeted experiments** to fill gaps identified by strict reviewer feedback. The theory and code infrastructure are ready — this is purely an experiment execution + result analysis task.

## What NOT to Do

- Do NOT run literature search or bootstrap — the paper is already written
- Do NOT use ML training skills (DeepSpeed, vLLM, Axolotl, etc.) — experiments are statistical simulations in NumPy/SciPy/PyTorch
- Do NOT modify the core FedProTrack codebase (`fedprotrack/`) — write NEW standalone experiment scripts
- Do NOT try to install new packages — everything needed is already in the conda environment
- Do NOT pursue Extension III (proportional high-dim regime) or Unified Theorem 23 — those are cut from scope

## Infrastructure

- **Local GPU**: RTX 4060 (8GB), use for debugging single-seed runs
  - Command: `E:/anaconda3/python.exe <script.py> --seeds 42`
- **Remote GPU**: RunPod, use for multi-seed final runs
  - Driven by the `run-experiment` skill (endpoint + API key stored there)
  - Handler: `runpod/handler.py`; results land as `runpod_*.json` / `runpod_log_*.txt`
  - 阿里云 ACS / kubectl / `k8s/` 脚本已废弃，不要再用
- Existing experiment scripts for reference: `run_experiments.py`, `run_cifar100_comparison.py`, `run_omega_calibration_experiment.py`

## The 5 Hypotheses (execute in this order)

### H1: Boundary-Focused Sweep (Priority 1)
**Goal**: Validate base theorem with dense SNR grid around crossover boundary.
**Method**:
- Stationary only (tau=inf), C ∈ {2,4,8}, K ∈ {10,20,40}
- Dense delta grid generating SNR ∈ [0.3*(C-1), 2.0*(C-1)] with ~20 points per (K,C) pair
- 10 seeds per config (not 3)
- d=20, n=200, sigma=1.0 (same as main paper)
**Output**:
- `results/boundary_sweep/` — per-config MSE for global vs concept-level vs shrinkage
- Figure: SNR/threshold vs excess-risk-difference, with theoretical 0-crossing marked
- Separate alignment rates for stationary vs non-stationary
**Success criterion**: Stationary alignment ≥ 95%

### H2: Noisy Label Crossover (Priority 1)
**Goal**: Validate Extension II Theorem 9 — threshold shifts to (C-1)/(1-ρ²) under label noise.
**Method**:
- Fix K=20, C=4, d=20, n=200, sigma=1.0
- Sweep η ∈ {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.74}
- For each η, sweep delta to find empirical crossover point
- 10 seeds per (η, delta) pair
- Also test noisy-label shrinkage (Prop 11) vs original shrinkage
**Output**:
- `results/noisy_label_crossover/` — measured thresholds
- Figure: η vs measured-crossover-threshold, overlaid with theory curve (C-1)/(1-ρ²)
- Table: shrinkage vs noisy-shrinkage MSE comparison
**Success criterion**: Measured thresholds within 15% of theoretical curve

### H3: Shrinkage on CIFAR-100 (Priority 1)
**Goal**: Demonstrate shrinkage works on real data, not just synthetic.
**Method**:
- Use existing CIFAR-100 recurrence pipeline (see `run_cifar100_comparison.py`)
- Implement shrinkage estimator for CIFAR-100: after concept-level models are trained, shrink toward global
- Compare: FedAvg, Oracle, Shrinkage on both disjoint and overlap label splits
- 5 seeds, matched n_epochs across all methods
**Output**:
- `results/shrinkage_cifar/` — accuracy per method per seed
- Table: method × label_split × accuracy (with std)
**Success criterion**: Shrinkage ≥ Oracle accuracy in overlapping setting, competitive in disjoint

### H4: Unbalanced Concept Proportions (Priority 2)
**Goal**: Validate Extension I — per-concept crossover threshold = (1-πj)/πj.
**Method**:
- Synthetic, stationary, C=4
- Three imbalance settings:
  - Balanced: π = [0.25, 0.25, 0.25, 0.25]
  - Mild: π = [0.5, 0.25, 0.15, 0.1]
  - Heavy: π = [0.7, 0.15, 0.1, 0.05]
- Sweep delta to find per-concept crossover points
- 10 seeds per config
**Output**:
- `results/unbalanced_crossover/` — per-concept crossover thresholds
- Figure: per-concept measured threshold vs predicted (1-πj)/πj
- Verify Corollary 6: weighted-average threshold still ≈ C-1
**Success criterion**: Per-concept thresholds within 20% of theory

### H5: r_eff Prospective Prediction (Priority 2)
**Goal**: Turn post-hoc r_eff correction into prospective prediction protocol.
**Method**:
- Estimate r_eff on held-out CIFAR-100 features (participation ratio method)
- Use r_eff to predict: does Oracle or FedAvg win for each (label_split, seed)?
- Compare predictions vs actual outcomes
- If possible, test with a second backbone (MobileNetV2 or similar)
**Output**:
- `results/reff_prospective/` — predictions and actuals
- Table: backbone × label_split × predicted_winner × actual_winner
**Success criterion**: ≥ 5/6 correct directional predictions

## Proxy Metric

Track `theory_experiment_alignment_rate` — fraction of configs where theoretical prediction matches empirical outcome. Current baseline: 91.7% (108 configs, mixed). Target: ≥ 95% on stationary configs.

## Output Requirements

- Every experiment script goes in project root as `run_<name>.py`
- Every script must accept `--seeds`, `--results-dir`, `--data-root`, `--feature-cache-dir`, `--n-workers` CLI args
- Git commit before and after each experiment
- Update findings.md after each hypothesis completes
- Generate a progress presentation in `to_human/` after H1-H3 complete
