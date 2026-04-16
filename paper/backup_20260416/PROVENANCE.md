# Paper backup — 2026-04-16

Before rewriting `main.tex` for held-out + SNR-gated shrinkage narrative.

## Preserved artifacts
- `main.tex` — copy of `paper/main.tex` as of this commit
- `main.pdf` — last compiled PDF

## Code version at backup time
- HEAD: `285f722` (Add experiment registry: code versions, scripts, configs, and results)
- Preceded by `9c55ed0` (Add SNR-gated DRCT shrinkage with EMA variance smoothing)
- Preceded by `3deb650` (Add held-out test pool for prequential evaluation)

## Key runtime features active
- `drct_snr_gate: bool = True` (default-on, added in 9c55ed0)
- `eval_on_test_pool: bool = True` (default-on, added in 3deb650)
- OT path now increments `protocol._round` (fix in 9c55ed0)

## Final held-out numbers (3 seeds: 42/43/44)

### CIFAR-10 paper config
- Script: `run_cifar100_neurips_benchmark.py`
- Flags: `--K 20 --T 100 --rho 25 --alpha 0.75 --delta 0.85 --dataset cifar10 --methods core --n-seeds 1 --feature-seed 2718 --lr 0.02 --n-epochs 10`
- FedAvg mean = 0.7572
- FedProTrack mean = 0.7942 (+3.70pp vs FedAvg, −1.20pp vs Oracle)
- Oracle mean = 0.8063

### CIFAR-100 disjoint paper config
- Same script + flags, `--dataset cifar100 --label-split disjoint`
- FedAvg mean = 0.7278
- FedProTrack mean = 0.7503 (+2.25pp vs FedAvg, −0.95pp vs Oracle)
- Oracle mean = 0.7598

## Raw JSON outputs preserved
- `runpod_paper_cifar10_v2.json`
- `runpod_paper_cifar100_disjoint.json`
- `runpod_cifar10_snrgate_fixed.json` (SNR-gate ablation, CIFAR-10 diagnostic)
- `runpod_cifar100_gateonly.json` (gate-only β=0 ablation)
- `runpod_cifar100_stein.json` (pure Stein baseline)

## What the old paper claimed (to be revised)
- CIFAR-10 as negative control: FPT=0.907 vs FedAvg=0.917 (−1.0pp)
- CIFAR-100: FPT=0.806 under old pool-reuse protocol
Both numbers were on the train-pool-reuse evaluation that is now known
to inflate absolute accuracy by ~7pp due to sample revisits.
