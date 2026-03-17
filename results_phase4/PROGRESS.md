# Phase 4 Experiment Progress

**Last updated**: 2026-03-17
**Status**: Stage 1 (E5 grid) in progress, ~580/1125 settings complete

## Pipeline Overview

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | E5 Synthetic Grid (1125 settings × 10 methods) | **In progress** (SINE ✅, SEA 200/375, CIRCLE 0/375) |
| 2 | E6 Rotating MNIST Grid (375 settings × 4 methods) | Pending |
| 3 | E5 Conditional Analysis (from CSV) | Pending |
| 4 | E6 Stability Analysis (from CSV) | Pending |
| 5 | Case Studies (6 settings × FPT+IFCA) | Pending |
| 6 | Component Ablation (18 anchors × 7 variants × 3 seeds) | Pending |
| 7 | Hyperparameter Robustness (5 params × ~7 values × 3 settings × 3 seeds) | Pending |
| 8 | Statistical Significance (5 settings × 20 seeds × 4 methods) | Pending |
| 9 | E4 Byte Breakdown + Event-Triggered | Pending |

## Current Data: raw_e5.csv

- **Total rows**: 6000 (600 unique settings × 10 methods)
- **SINE**: 375/375 settings ✅
- **SEA**: 200/375 settings (rho=2.0 all done, rho=5.0 partially done through alpha=0.75)
- **CIRCLE**: 0/375 settings

## Key Results So Far

### FedProTrack vs IFCA (concept re-ID accuracy)

| Generator | FPT Mean | IFCA Mean | Δ | n |
|-----------|----------|-----------|---|---|
| SINE | 0.797±0.182 | 0.665±0.236 | **+0.133** | 375 |
| SEA | 0.561±0.179 | 0.370±0.212 | **+0.191** | 200 |

### Per-rho Breakdown

| Setting | FPT | IFCA | Δ |
|---------|-----|------|---|
| SINE rho=2.0 | 0.561 | 0.382 | +0.179 |
| SINE rho=5.0 | 0.909 | 0.707 | +0.202 |
| SINE rho=10.0 | 0.922 | 0.905 | +0.017 |
| SEA rho=2.0 | 0.435 | 0.214 | +0.221 |
| SEA rho=5.0 | 0.717 | 0.565 | +0.153 |

### Key Observations
1. **FedProTrack consistently beats IFCA** across all completed settings
2. **Advantage is largest at low rho** (hard concept separation): +0.18 to +0.22
3. **SEA shows even stronger advantage** than SINE (+0.19 vs +0.13 overall)
4. At high rho (easy separation), gap shrinks to ~0.02 (both methods near ceiling)
5. **E1 gate PASS confirmed**: SINE re-ID 0.797 > IFCA 0.665 (+0.133)

## How to Resume

```bash
# Resume from where we left off (appends to existing CSV)
PYTHONUNBUFFERED=1 E:/anaconda3/python.exe resume_phase4.py

# Or run specific stages only (after Stage 1 is complete)
PYTHONUNBUFFERED=1 E:/anaconda3/python.exe run_phase4_analysis.py \
    --results-dir results_phase4 --skip-stage1 --skip-stage2

# Or run analysis-only stages on existing data
PYTHONUNBUFFERED=1 E:/anaconda3/python.exe run_phase4_analysis.py \
    --results-dir results_phase4 --only 3,4,5
```

## File Structure

```
results_phase4/
├── raw_e5.csv          # 6000 rows (580 settings × 10 methods, partial)
├── PROGRESS.md         # This file
run_phase4_analysis.py  # Full 9-stage pipeline
resume_phase4.py        # Resume script with incremental append
fedprotrack/experiments/phase4_analysis.py  # Analysis functions
```
