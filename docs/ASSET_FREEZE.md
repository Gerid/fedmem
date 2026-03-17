# Paper Asset Freeze

## Main Text Assets (6 items)

| ID | Asset | Question Answered | Experiment | Status |
|----|-------|-------------------|------------|--------|
| Table 1 | Main synthetic results table | FedProTrack competitive overall? | E5 | DONE (1125 settings) |
| Fig 1 | Identity/reuse panel (re-ID vs delta, wrong-mem vs delta) | Why does memory help? | E1 | DONE, E1 gate PASS |
| Fig 2 | Memory-help phase diagram (diff heatmaps) | When is memory helpful/harmful? | E3 | DONE (signature result) |
| Fig 3 | Budget frontier + budget/alpha heatmap | Two-phase protocol worth it? | E4 | DONE, E4 gate FAIL (expected) |
| Fig 4 | Recovery/adaptation curves | Temporal prior improves recovery? | E2 | DONE, can demote to appendix |
| Table 2 | Real/semi-real results | Not a synthetic artifact? | E6 | DONE (375 settings) |

## If page-constrained: drop Fig 4 to appendix, keep Table 1 + Fig 1-3 + Table 2

## Appendix Assets

| Asset | Experiment | Status |
|-------|------------|--------|
| Axis tables (rho/alpha/delta) | E5 | Code ready |
| Module ablation bar charts | E7 | Code ready, needs frozen-config rerun |
| Overhead table | E9 | Code ready, needs lifecycle counters |
| Full alpha-slice phase diagrams | E3 | Code ready |
| Scalability plots | appendix | Code ready |
| Theory-linked plots | E8 | Missing |

## Gates (SINE-only final, 375 settings, 5 seeds)

- **E1 gate**: PASS — FedProTrack re-ID 0.797 > IFCA 0.665 (+0.133)
  - Key fix: `loss_novelty_threshold=0.02` (was 0.05), `sticky_dampening=1.0` (was 1.5)
- **E4 gate**: FAIL — FedProTrack dominated at all budget levels by simpler methods
  - Structural: Phase A overhead adds bytes without proportional accuracy AUC on simple SINE
  - Paper narrative: FedProTrack trades communication for concept identity (re-ID), not raw accuracy
  - Best budget-normalized: CompressedFedAvg (0.049 AUC/byte)

## Final Results Summary

### E5 Synthetic (1125 settings, 3 generators, 5 seeds)
| Method | Re-ID | Final Acc | Mean Rank | Win Rate |
|--------|-------|-----------|-----------|----------|
| **FedProTrack** | **0.639** | **0.678** | **2.92** | 0.23 |
| IFCA | 0.537 | 0.672 | 3.52 | 0.28 |
| FedProto | -- | 0.657 | 3.77 | **0.47** |
| TrackedSummary | 0.473 | 0.665 | 3.63 | 0.05 |
| FedDrift | 0.460 | 0.565 | 6.08 | 0.00 |

### E6 Rotating MNIST (375 settings, 5 seeds)
| Method | Re-ID | Final Acc | Mean Rank |
|--------|-------|-----------|-----------|
| IFCA | **0.544** | **0.755** | **1.78** |
| FedProTrack | 0.505 | 0.702 | 3.53 |
| FedProto | -- | 0.753 | 1.89 |

## Frozen Settings

- Main grid: K=10, T=20, n_samples=500
- Seeds: 42, 123, 456, 789, 1024
- Generators: sine, sea, circle
- Primary methods: FedAvg, FedProto, TrackedSummary, Flash, FedDrift, IFCA, FedProTrack
- Reference: Oracle (appendix only)
- E6: K=5, T=10, n_samples=200, n_features=20, Rotating MNIST
