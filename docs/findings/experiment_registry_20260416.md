# Experiment Registry — 2026-04-16

Records which code version, script, and config produced each set of results.

---

## 1. Held-out Evaluation Flagship (CIFAR-100 disjoint)

**Commit**: `3deb650` (Add held-out test pool for prequential evaluation)  
**Script**: inline Python (local), not a standalone script  
**Config**:
```python
CIFAR100RecurrenceConfig(
    K=20, T=100, n_samples=200, rho=25.0, n_features=128,
    samples_per_coarse_class=500, label_split="disjoint",
    seed=42, n_workers=0,
    eval_on_test_pool=True, test_split_ratio=0.2,
)
# FPT: concept_discovery="ot", lr=0.01, n_epochs=5, federation_every=2
# Baselines: lr=0.01, n_epochs=5, federation_every=2
```
**Results** (single seed=42):

| Method | final_acc |
|--------|-----------|
| FPT-OT | 0.7362 |
| Oracle | 0.7322 |
| FedAvg | 0.6865 |
| FedCCFA | 0.6710 |
| FedEM(4) | 0.6562 |
| FedDrift | 0.6183 |
| FedRC(4) | 0.5845 |

---

## 2. Pool-resampling Evaluation (OLD, pre-held-out fix)

**Commit**: `3754aa9` (FedDrift spawn threshold fix)  
**Script**: inline Python (local)  
**Config**: Same as above BUT `eval_on_test_pool=False`, `samples_per_coarse_class=120`  
**Results** (single seed=42):

| Method | final_acc | Note |
|--------|-----------|------|
| Oracle | 0.8160 | Inflated ~8pp |
| FPT-OT | 0.8055 | Inflated ~7pp |
| FedAvg | 0.7550 | Inflated ~7pp |
| FedDrift | 0.7115 | Inflated ~9pp |
| FedRC(4) | 0.6968 | Inflated ~11pp |
| FedEM | (not run this commit) | |
| FedCCFA | (not run this commit) | |

---

## 3. RunPod Baseline Patch (5 CIFAR-100 configs × 3 seeds)

**Commit**: `ad951d0` (Update paper with corrected baseline numbers)  
**Note**: RunPod used OLD code — FedRC did NOT have hard-assignment fix  
**Script**: `run_baseline_patch.py`  
**RunPod result files**: `runpod_patch4_seed{42,43,44}.json`  
**Config**:
```python
# 5 configs: K20/rho17, K20/rho25, K20/rho33, K40/rho25, K40/rho33
# T=100, n_samples=200, n_features=128, spc=120, lr=0.01, n_epochs=5
# label_split="disjoint", eval_on_test_pool=False (old pool-resampling)
```
**Cross-config mean results**:

| Method | Old paper | RunPod new |
|--------|-----------|------------|
| FedDrift | 0.211 | 0.310 |
| FedEM | 0.523 | 0.710 |
| FedRC | 0.291 | 0.386 (old code) |
| FedCCFA | 0.503 | 0.696 |

---

## 4. Concept-Count Sensitivity (C-sweep, OT path)

**Commit**: `b72da14` (Concept-count sensitivity experiments)  
**Script**: `run_concept_count_sweep.py`  
**RunPod result files**: `runpod_csweep_ot_rho{15.0,7.5,5.0,3.75}.json`  
**Config**:
```python
CIFAR100RecurrenceConfig(
    K=20, T=30, n_samples=200, rho={15.0,7.5,5.0,3.75},
    n_features=128, samples_per_coarse_class=120,
    label_split="disjoint", seed={42,43,44},
)
# FPT: concept_discovery="ot"
# eval_on_test_pool=False (old protocol)
```
**Results** (3-seed means):

| Method | C=2 | C=4 | C=6 | C=8 |
|--------|-----|-----|-----|-----|
| FPT-OT | 0.656 | 0.699 | 0.800 | 0.888 |
| Oracle | 0.653 | 0.702 | 0.806 | 0.897 |
| FedAvg | 0.646 | 0.655 | 0.676 | 0.774 |

Eigengap: C=2→2, C=4→4, C=6→6.7, C=8→9.0

---

## 5. Misspecified-C (true C=4, OT path)

**Commit**: `b72da14`  
**Script**: `run_misspecified_c.py`  
**RunPod result files**: `runpod_misspec_ot_nc{2,3,4,6,8}.json`  
**Config**: Same as C-sweep but fixed rho=7.5 (C=4), varying n_clusters  
**Results** (3-seed means):

| Method | nc=2 | nc=3 | nc=4 | nc=6 | nc=8 |
|--------|------|------|------|------|------|
| FPT-OT | 0.699 | 0.699 | 0.699 | 0.699 | 0.699 |
| IFCA | 0.530 | 0.519 | 0.503 | 0.490 | 0.490 |

---

## 6. Fair Label-Permutation Comparison (pairwise swap)

**Commit**: `3deb650` (held-out pool commit, also has label_permutation)  
**Script**: `run_cifar100_neurips_benchmark.py` with `--label-split none --label-permutation`  
**RunPod result file**: `runpod_fair_labelperm_seed42.json`  
**Config**:
```bash
python run_cifar100_neurips_benchmark.py \
  --fpt-mode ot --K 20 --T 100 --rho 25 \
  --label-split none --label-permutation --label-permutation-type pairwise_swap \
  --methods core,cluster,drift --seeds 42 --n-seeds 1
```
**Results** (single seed=42, T=100, pairwise swap):

| Method | Acc |
|--------|-----|
| Oracle | 0.661 |
| FedRC | 0.609 |
| FedEM | 0.597 |
| FedDrift | 0.597 |
| Flash | 0.596 |
| FedAvg | 0.588 |
| FPT-OT | 0.588 |
| CFL | 0.586 |
| IFCA | 0.583 |
| FedCCFA | 0.474 |

**Note**: Pairwise swap is too subtle (2/20 labels differ per concept). FPT = FedAvg.

---

## Key Commits Reference

| Commit | Description |
|--------|-------------|
| `3deb650` | Held-out test pool across all datasets + label_permutation |
| `9c55ed0` | SNR-gated DRCT shrinkage |
| `3754aa9` | FedDrift log(n_classes) spawn threshold |
| `aac7b0c` | FedRC hard cluster assignment |
| `ddff64a` | FedRC batch_size weight + FedCCFA memory normalization |
| `78b38fd` | FedDrift loss-based pool eval + FedEM EMA/blending fix |
| `b72da14` | Concept-count sensitivity + misspecified-C scripts |
| `ad951d0` | Paper numbers update |
| `c3af051` | Paper: pipeline v2, extensions, concept-count appendices |

---

## Next Steps

- [ ] Re-run Table 1 旗舰 with held-out eval (5 configs × 3 seeds × 4 datasets)
- [ ] Update paper Table 1 with held-out numbers
- [ ] Run label-permutation with full_permutation (stronger drift signal)
- [ ] Consider adding samples_per_coarse_class CLI arg to neurips benchmark script
