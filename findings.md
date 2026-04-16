# Research Findings — Crossover Theory Experiment Supplement

## Status: Starting experiment phase

## Context
The main paper proves SNR_concept > C-1 as a sharp crossover threshold for concept-level vs global aggregation in linear-Gaussian FL. The extensions document relaxes 4 assumptions (unbalanced, noisy labels, high-dim, anisotropic). Current experiments only validate the base theorem on 108 synthetic configs + 6-seed CIFAR-100 bridge.

## Key Gaps to Fill
1. **Boundary calibration**: 9/108 mismatches all near SNR ≈ C-1; need denser sweep
2. **Noisy labels**: Extension II predicts threshold inflation by 1/(1-ρ²) — zero experiments
3. **Shrinkage on real data**: 72% win rate on synthetic only; never tested on CIFAR-100
4. **Unbalanced**: Extension I predicts per-concept threshold (1-πj)/πj — zero experiments
5. **r_eff prospective**: CIFAR-100 correction is post-hoc; needs prospective protocol

## Findings
(To be populated as experiments complete)
