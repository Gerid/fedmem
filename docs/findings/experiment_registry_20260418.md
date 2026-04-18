# Experiment Registry — 2026-04-18

Records which code version, script, and config produced each set of results.
This registry updates 20260416 entries with correct protocol tagging following the
`label_split` protocol-mismatch diagnosis of 2026-04-18.

---

## 0. Protocol mismatch bug (RESOLVED)

**Bug**: `run_cifar100_neurips_benchmark.py:1442` `config_dump` did NOT persist
`label_split`, `label_permutation`, `eval_on_test_pool`, `dirichlet_alpha`,
`fmow_n_classes`, or several other protocol-defining fields. As a result,
CIFAR-100 retry runs that copied from the original `config.json` silently
switched from `label_split="disjoint"` to `label_split="none"` (the module-level
default), producing ~8pp uniformly lower accuracy for ALL 17 methods and
flipping Oracle < FedAvg.

**Resolution**: `config_dump` in `run_cifar100_neurips_benchmark.py:1467-1484`
now persists these fields. Diagnosis by Codex `gpt-5.4` xhigh; fix applied in
the same session.

**Invalidated results** (wrong protocol `label_split=none`, quarantined):
- `runpod_heldout_cifar100_K20_rho17_seed43_WRONG_PROTOCOL_lsnone.json`
- `runpod_supp_cifar100_K20_rho25_seeds454647_WRONG_PROTOCOL_lsnone.json`

---

## 1. Main Table 1 CIFAR-100 runs (disjoint, held-out)

All runs use `label_split="disjoint"`, `eval_on_test_pool=true`.

**Commit at submission time**: `55f7900` (Fix stale appendix data).

**Shared CLI flags**:
```
--dataset cifar100 --K 20|40 --T 100 --rho 17|25|33 --alpha 0.75 --delta 0.85
--drift-type recurrent --n-features 128 --n-samples 400 --samples-per-coarse-class 120
--batch-size 128 --participation 1.0 --federation-every 1 --feature-seed 2718
--lr 0.02 --n-epochs 10 --fpt-lr 0.02 --fpt-epochs 10 --methods all
--model-type linear --eval-on-test-pool true --label-split disjoint
--drct-snr-threshold 1.0 --drct-sigma-ema-beta 0.0 --drct-warmup-rounds 0
```

| Config | Seeds | RunPod JSON | Git tag |
|--------|-------|-------------|---------|
| K20/ρ17 | 42, 44 | `runpod_heldout_cifar100_K20_rho17_all.json` | `exp/cifar100_k20_rho17_heldout-20260417` |
| K20/ρ17 | 43 retry | `runpod_heldout_cifar100_K20_rho17_seed43_disjoint.json` | `exp/table1-k20-rho17-seed43-20260418` (pending regen with correct protocol) |
| K20/ρ25 | 42, 43, 44 | `runpod_flagship_all_methods.json` | `exp/flagship_cifar100-20260417` |
| K20/ρ25 | 45, 46, 47 | `runpod_supp_cifar100_K20_rho25_seeds454647_disjoint.json` | `exp/supp-cifar100-seeds454647-20260418` (pending regen) |
| K20/ρ33 | 42, 43, 44 | `runpod_heldout_cifar100_K20_rho33_all.json` | — |
| K40/ρ25 | 42, 43, 44 | `runpod_heldout_cifar100_K40_rho25_all.json` | — |
| K40/ρ33 | 42, 43, 44 | `runpod_heldout_cifar100_K40_rho33_all.json` | — |

---

## 2. Supplementary seeds (held-out, Main Table 1 columns)

| Dataset | Seeds | RunPod JSON | Protocol |
|---------|-------|-------------|----------|
| F-MNIST | 45, 46, 47 | `runpod_supp_fmnist_K20_rho25_seeds454647.json` | held-out (no label_split param for F-MNIST) |
| CIFAR-10 | 45, 46, 47 | `runpod_supp_cifar10_K20_rho25_seeds454647.json` | held-out (no label_split param for CIFAR-10) |

**These two are VALID** — F-MNIST and CIFAR-10 recurrence generators don't use
`label_split`, so the protocol-mismatch bug doesn't apply. Can be pooled with
seeds 42/43/44 for n=6 analysis.

---

## 3. Independent verification (Codex gpt-5.4 reproduction, 2026-04-18)

Codex ran `run_cifar100_neurips_benchmark.py` at commit `55f7900` four times with
K=20, ρ=25, lr=0.02, e=10, held-out protocol:

| Seed | label_split | Oracle | FedAvg |
|------|-------------|--------|--------|
| 42 | disjoint | 0.7804 | 0.7428 |
| 45 | disjoint | 0.7730 | 0.7380 |
| 45 | none | 0.5556 | 0.5806 |
| 43 on ρ=17 | none | 0.5651 | 0.5809 |
| 43 on ρ=17 | disjoint | 0.8471 | 0.7964 |

Conclusion: seeds 45/46 are not intrinsically hard; the 8pp drop was entirely
`label_split="none"` vs `label_split="disjoint"`.
