from __future__ import annotations
# NeurIPS Experiment Setup Gap Analysis (2026-04-07)

**Purpose**: Align FedProTrack experiments with top-conference baselines (NeurIPS/ICML/ICLR 2024-2025).
**Status**: Post first-round alignment (commit c35f04e). This document tracks remaining gaps.

---

## 1. Competitor Experiment Setups (Reference)

| Paper | Venue | K (clients) | T (rounds) | E (local epochs) | Model | Datasets | Drift | Participation |
|-------|-------|-------------|------------|-------------------|-------|----------|-------|---------------|
| FedDrift | AISTATS'23 | 10 | 200 | 5 | FNN/CNN | SINE, SEA, MNIST, FMoW | synthetic changepoints | 100% |
| Flash | ICML'23 | ~10-50 | 200+ | adaptive (early-stop) | CNN | CIFAR-10, MNIST | gradient-based detect | partial |
| FedRC | ICML'24 | 100 | 200 | -- | CNN/MobileNetV2/ResNet18 | FMNIST, CIFAR-10/100, TinyImageNet | LDA alpha=1.0 + feature shift (CIFAR-C) + concept shift | partial |
| FedCCFA | NeurIPS'24 | 20 (full) / 100 (20%) | 200 | 5 | CNN | FMNIST, CIFAR-10, CINIC-10 | label-swap (sudden/incremental/recurrent) | 100% / 20% |
| HCFL | ICLR'25 | 100 | 200 | -- | MobileNetV2/ResNet18 | CIFAR-10/100 (CIFAR-C) | feature + label shift | -- |
| FedGWC | ICML'25 | varied | -- | -- | -- | CIFAR-100, iNaturalist, Landmarks | class heterogeneity | -- |
| FedDAA | arXiv'25 | -- | multi-timestep | -- | MobileNet (C100)/ResNet18 (C10) | FMNIST, CIFAR-10/100 | real+virtual+label drift, 6 timesteps | 50% |
| FedStein | NeurIPS'24 WS | 20-100 | 50 | 1 | CNN/AlexNet/ResNet18 | Digits-Five, DomainNet, Office | cross-domain shift | -- |

---

## 2. Gaps Closed (commit c35f04e, 2026-04-06)

| # | Gap | Before | After |
|---|-----|--------|-------|
| 1 | K too small | 4-6 | 20 (default) / 100 (--preset large) |
| 2 | T too small | 6-12 | 200 (default) / 100 (--preset small) |
| 3 | No partial participation | none | --participation 0.2 |
| 4 | No E2E model | linear only | TorchSmallCNN (545K) + TorchMobileNetV2 |
| 5 | No FedCCFA baseline | missing | fedccfa_impl.py (DBSCAN clustering + feature alignment, 49 tests) |
| 6 | No FedProx baseline | missing | fedprox.py (proximal term, 21 tests) |
| 7 | Only CIFAR-100 | 1 vision dataset | +CIFAR-10 recurrence + Fashion-MNIST recurrence |
| 8 | Only recurrent drift | 1 drift type | sudden + incremental + recurrent |
| 9 | No NeurIPS-scale script | smoke-level only | run_cifar100_neurips_benchmark.py (presets: small/medium/large/full) |

Test count: 381 -> 740 collected, 726 pass, 0 regressions.

---

## 3. Remaining Gaps

### 3.1 Code-Level Gaps (need implementation)

**G1 [P0, 10 min] -- Benchmark script missing FedProx + FedCCFA-Impl wiring**
- `run_cifar100_neurips_benchmark.py` uses old `run_fedccfa_full()` not the new `run_fedccfa_impl_full()`.
- `FedProx` not in `_build_methods()` or `METHOD_GROUPS`.
- Fix: add two entries in `_build_methods()`, add "FedProx" to appropriate METHOD_GROUP.

**G2 [P1, 1h] -- Benchmark script has no multi-dataset support**
- Script is hardcoded to CIFAR-100. CIFAR-10 and FMNIST dataset code exists but is not wired in.
- Fix: add `--dataset {cifar100,cifar10,fmnist}` argument, dispatch to corresponding recurrence module.

**G3 [P1, 2h] -- E2E models not wired into baseline runners**
- All `run_xxx_full()` in `baselines/runners.py` hardcode `TorchLinearClassifier`.
- SmallCNN and MobileNetV2 exist in `models/cnn.py` but baselines cannot use them.
- Fix: add `model_type` parameter to each runner, factory-dispatch to correct model class.

**G4 [P2, 3h] -- SCAFFOLD baseline missing**
- Used by FedCCFA as one of its 14 baselines.
- Requires variance reduction via control variates (server + client correction terms).
- Reference: Karimireddy et al., ICML 2020.

**G5 [P2, 2h] -- Ditto baseline missing**
- Used by FedCCFA as PFL baseline.
- FedAvg for global model + per-client fine-tuning with proximal regularization.
- Reference: Li et al., ICML 2021.

**G6 [P2, 4h] -- HCFL / HCFL+ baseline missing**
- ICLR'25, latest clustered FL. Four-tier framework with integration of strategies.
- Need to read full paper algorithm section for implementation.

**G7 [P2, 4h] -- FedGWC baseline missing**
- ICML'25, Gaussian weighting + Wasserstein Adjusted Score for clustering.
- Need full paper for implementation details.

**G8 [P2, 2h] -- Adaptive-FedAvg baseline missing**
- Used by FedDrift and FedDAA as baseline.
- FedAvg with drift-adaptive learning rate adjustment.

### 3.2 Experiment Design Gaps (need decisions + runs)

**E1 [P1] -- Non-IID severity ablation**
- Competitors: FedCCFA tests Dir(0.01/0.1/0.5) to vary heterogeneity.
- We only have coarse-label-based splits.
- Fix: add Dirichlet-controllable heterogeneity to recurrence datasets.

**E2 [P1] -- Multi-architecture results**
- Competitors: HCFL reports MobileNetV2 + ResNet18 on same benchmarks.
- We must run at least linear + SmallCNN (two architectures) for the paper.

**E3 [P2] -- Feature shift (CIFAR-C style corruption)**
- FedRC and HCFL use CIFAR-10-C / CIFAR-100-C with image corruptions.
- Our drift is label-based only.
- Could leave for appendix or add as robustness check.

**E4 [P1] -- Convergence curves**
- All competitors show accuracy-vs-round plots.
- Benchmark script outputs per-round CSV data; need plotting code.

**E5 [P2] -- Ablation at NeurIPS scale**
- Phase 4 ablations were done at K=4-6, T=12.
- Need to re-run ablations at K=20, T=200 for the paper.

**E6 [P0] -- End-to-end validation of benchmark script**
- The script has never completed a full run.
- Background smoke test failed due to CIFAR-100 cache path issue (not code bug).
- Must run `--quick --data-root E:\fedprotrack\.cifar100_cache` locally.

### 3.3 Paper Narrative Gaps

**N1** -- Related work must discuss FedCCFA (NeurIPS'24), HCFL (ICLR'25), FedGWC (ICML'25).
**N2** -- Main results table needs method x dataset format with bold-best, matching competitor style.
**N3** -- If DRCT is added: 4-arm shrinkage ablation table + FedStein comparison.
**N4** -- Communication-accuracy tradeoff figure (our unique advantage) must be re-drawn at new scale.

---

## 4. Priority Roadmap

```
Immediate (< 30 min):
  G1  -- Wire FedProx + FedCCFA-Impl into benchmark script
  E6  -- Local --quick smoke run to validate end-to-end

This week (P1):
  G2  -- Multi-dataset support in benchmark script
  G3  -- E2E model support in baseline runners
  E2  -- Run linear + SmallCNN dual-architecture experiments
  E4  -- Convergence curve plotting

Pre-submission (P2):
  G4-G8  -- Additional baselines (driven by reviewer expectations)
  E1     -- Dirichlet heterogeneity ablation
  E5     -- Ablations at NeurIPS scale

Optional / reviewer-driven (P3):
  E3     -- CIFAR-C feature shift
  G6-G7  -- HCFL / FedGWC baselines
```

---

## 5. Baseline Coverage Matrix

| Method | Implemented? | In benchmark script? | In method_registry? | Competitor usage |
|--------|-------------|---------------------|--------------------|--------------------|
| FedAvg | yes | yes | yes | universal |
| FedProx | **yes (new)** | **no** | yes | FedCCFA, universal |
| SCAFFOLD | no | no | no | FedCCFA |
| IFCA | yes | yes | yes | FedRC, FedCCFA, FedDrift |
| FeSEM | yes | yes | yes | FedRC |
| CFL | yes | yes | yes | multiple |
| FedRC | yes | yes | yes | HCFL, FedDAA |
| FedEM | yes | yes | yes | FedRC |
| FedCCFA (old) | yes | yes (uses old impl) | yes | -- |
| FedCCFA-Impl (new) | **yes (new)** | **no** | yes | direct competitor |
| FedDrift | yes | yes | yes | FedCCFA, FedDAA |
| Flash | yes | yes | yes | FedCCFA, FedDAA |
| pFedMe | yes | yes | yes | multiple |
| APFL | yes | yes | yes | multiple |
| ATP | yes | yes | yes | -- |
| Ditto | no | no | no | FedCCFA |
| FedRep/FedBABU | no | no | no | FedCCFA |
| Adaptive-FedAvg | no | no | no | FedDrift, FedDAA |
| HCFL/HCFL+ | no | no | no | ICLR'25 latest |
| FedGWC | no | no | no | ICML'25 latest |
| FedStein | no (bib only) | no | no | shrinkage competitor |
| FLUX / FLUX-prior | yes | yes | yes | -- |
| CompressedFedAvg | yes | yes | yes | -- |
| TrackedSummary | yes | yes | yes | -- |
| FedProto | yes | yes | yes | -- |
| Oracle | yes | yes | yes | universal |
| LocalOnly | yes | yes | yes | universal |

**Current count**: 19 implemented / 7 missing.
**Minimum for submission**: implement G1 (wire existing) + G4 (SCAFFOLD) + G5 (Ditto) = 21 methods.
**Ideal**: all 26 = covers every baseline used by any direct competitor.

---

## 6. Key Competitor Papers (for references.bib)

Already in bib:
- FedDrift (AISTATS'23), Flash (ICML'23), FedCCFA (NeurIPS'24), FedRC (ICML'24)
- FedGWC (ICML'25), FedStein (arXiv'24), SCAFFOLD (ICML'20), Ditto (ICML'21)

Need to add:
- HCFL (ICLR'25): arxiv 2310.05397
- FedDAA (arXiv'25): arxiv 2506.21054
- Hot-Pluggable FL (ICLR'25): bridging GFL and PFL via dynamic selection
