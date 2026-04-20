# SNR-Gate / Held-Out Eval Runtime Configuration Diagnosis — Table 1 (seeds 42/43/44)

**Question:** What were the actual runtime values of `drct_snr_gate`,
`drct_warmup_rounds`, `drct_sigma_ema_beta`, and `eval_on_test_pool` when
seeds 42/43/44 were executed for the NeurIPS 2026 paper Table 1? These
fields are ABSENT from the saved `config.json` blocks.

**Bottom line:**

| Field | Inferred runtime value | Confidence |
|---|---|---|
| `drct_snr_gate` | **`False`** (gate OFF) | HIGH |
| `drct_warmup_rounds` | `0` | HIGH |
| `drct_sigma_ema_beta` | `0.0` | HIGH |
| `eval_on_test_pool` | `True` (held-out eval ON) | HIGH |

This means Table 1 numbers come from pure Stein shrinkage **without** the
SNR gate, with held-out test-pool evaluation. The paper narrative in
§4.4 and §5.2 (which claims the SNR gate is ON by default and fires on
1–3 early rounds) does not match the Table 1 data; the SNR-gate
narrative comes from a separate ablation (Table 3, Appendix
`app:shrinkage_ablation`) that was run with the gate explicitly enabled.

---

## 1. Runtime signals per file

All 8 files saved the same truncated `config.json` block containing only
23 "classic" fields (K, T, alpha, batch_size, delta, drift_type,
feature_seed, federation_every, fpt_epochs, fpt_lr, fpt_mode, lr,
methods, n_epochs, n_features, n_samples, n_seeds, participation,
preset, quick, rho, samples_per_coarse_class, seeds). None of them
carry DRCT or eval-protocol signals.

| File | mtime (UTC+8) | has_drct_lambda_log | has_sigma_logs | has_warmup_events | has_eval_on_test_pool | has_drct_snr_* | Inferred gate at run time |
|---|---|---|---|---|---|---|---|
| `runpod_flagship_all_methods.json` | 2026-04-17 15:09 | no | no | no | no | no | **OFF** |
| `runpod_flagship_seed43_retry.json` | 2026-04-17 15:23 | no | no | no | no | no | **OFF** |
| `runpod_heldout_cifar100_K20_rho17_all.json` | 2026-04-17 18:07 | no | no | no | no | no | **OFF** |
| `runpod_heldout_cifar100_K20_rho33_all.json` | 2026-04-17 18:22 | no | no | no | no | no | **OFF** |
| `runpod_heldout_cifar100_K40_rho25_all.json` | 2026-04-17 18:41 | no | no | no | no | no | **OFF** |
| `runpod_heldout_cifar100_K40_rho33_all.json` | 2026-04-17 20:00 | no | no | no | no | no | **OFF** |
| `runpod_heldout_cifar10_all.json` | 2026-04-17 20:11 | no | no | no | no | no | **OFF** |
| `runpod_heldout_fmnist_all.json` | 2026-04-17 20:21 | no | no | no | no | no | **OFF** |

The "OFF" inference is based on the CLI-default + submission-command
evidence below, not on the absence of logs (DRCT code does not
conditionally create lambda_log keys based on the gate flag; absence of
any diagnostic log means the FedProTrack runner doesn't write these at
all in this entrypoint).

---

## 2. Git commit timeline and dataclass defaults

### Relevant commits

| Commit | Timestamp | Title | What it changed |
|---|---|---|---|
| `9c55ed0` | 2026-04-16 00:57 | Add SNR-gated DRCT shrinkage | Introduces SNR-gate feature (default in dataclass: **False**) + `--drct-snr-gate` CLI (store_true, CLI default False) |
| `3deb650` | 2026-04-16 02:59 | Add held-out test pool | Introduces `eval_on_test_pool` field in dataset configs (dataclass default **True**). No CLI flag yet. |
| `285f722` | 2026-04-16 09:34 | Add experiment registry | metadata only |
| `18847f5` | 2026-04-16 15:18 | Enable SNR-gated shrinkage by default | Flips dataclass default `drct_snr_gate: False → True`. CLI flag remains `store_true` (CLI default still False). |
| `73d6f78` | 2026-04-16 15:54 | Sync long-pending WIP | — |
| `94a1a2f` | 2026-04-17 16:28 | Rewrite paper with held-out + SNR-gated narrative | Adds `--eval-on-test-pool` CLI flag (default="true"); adds `--fmow-n-classes` CLI flag. |
| `4e9cfd6` | 2026-04-17 22:05 | Replace Table 1 with full 26-method numbers | — |
| `0752a94` | 2026-04-17 22:54 | Address Round 1-2 reviewer concerns | — |
| `55f7900` | 2026-04-17 23:27 | Fix stale appendix data (HEAD) | — |

### Defaults at run-time commits

For each raw JSON file, the commit that was HEAD when it was produced:

| File | mtime window | Commit at run time | `TwoPhaseConfig.drct_snr_gate` dataclass default | `TwoPhaseConfig.drct_warmup_rounds` | `TwoPhaseConfig.drct_sigma_ema_beta` | `cifar{10,100,fmnist}Config.eval_on_test_pool` |
|---|---|---|---|---|---|---|
| `runpod_flagship_all_methods.json` | 15:09 | `73d6f78` (15:54 not yet committed at 15:09 — actually between 18847f5 15:18 and 73d6f78 15:54). Closest committed HEAD is `18847f5`. | `True` | `0` | `0.0` | `True` |
| `runpod_flagship_seed43_retry.json` | 15:23 | same as above | `True` | `0` | `0.0` | `True` |
| `runpod_heldout_cifar100_K20_rho17_all.json` | 18:07 | `94a1a2f` | `True` | `0` | `0.0` | `True` |
| `runpod_heldout_cifar100_K20_rho33_all.json` | 18:22 | `94a1a2f` | `True` | `0` | `0.0` | `True` |
| `runpod_heldout_cifar100_K40_rho25_all.json` | 18:41 | `94a1a2f` | `True` | `0` | `0.0` | `True` |
| `runpod_heldout_cifar100_K40_rho33_all.json` | 20:00 | `94a1a2f` | `True` | `0` | `0.0` | `True` |
| `runpod_heldout_cifar10_all.json` | 20:11 | `94a1a2f` | `True` | `0` | `0.0` | `True` |
| `runpod_heldout_fmnist_all.json` | 20:21 | `94a1a2f` | `True` | `0` | `0.0` | `True` |

At every candidate commit (`18847f5`, `73d6f78`, `94a1a2f`), the
`TwoPhaseConfig` dataclass default for `drct_snr_gate` is **True**.

---

## 3. CLI override check (this is the decisive evidence)

The benchmark script `run_cifar100_neurips_benchmark.py` has argparse
definitions (identical at all three candidate commits `18847f5`,
`73d6f78`, `94a1a2f`, `55f7900`):

```python
# line ~1048
parser.add_argument("--drct-warmup-rounds", type=int, default=0, ...)
# line ~1055
parser.add_argument("--drct-snr-gate", action="store_true",
                    help="Enable SNR-gated shrinkage ...")
# line ~1057
parser.add_argument("--drct-snr-threshold", type=float, default=1.0)
# line ~1058
parser.add_argument("--drct-sigma-ema-beta", type=float, default=0.0, ...)
# line ~1080 (present at 94a1a2f+, absent at 73d6f78/18847f5)
parser.add_argument("--eval-on-test-pool", type=str, default="true", ...)
```

The CLI value flows into `TwoPhaseConfig(...)` as an explicit kwarg
(line ~405 and ~409 of the benchmark script), so the CLI value
**overrides** the dataclass default unconditionally.

- `--drct-snr-gate` uses `action="store_true"` → if the flag is not on
  the command line, `args.drct_snr_gate == False` regardless of the
  dataclass default.
- `--drct-warmup-rounds` CLI default = `0`.
- `--drct-sigma-ema-beta` CLI default = `0.0`.
- `--eval-on-test-pool` CLI default = `"true"` (and before `94a1a2f`,
  the CLI flag didn't exist, so the dataclass default `True` applied —
  same effect).

### Submission-time corroboration

`runpod_heldout_cifar100_K20_rho17_seed43_PROVENANCE.md` (a retry of
seed 43 for the K20/rho17 column), written 2026-04-18 as a faithful
copy of the original command for seeds 42/44, **does not pass
`--drct-snr-gate`, `--drct-snr-threshold`, or `--drct-sigma-ema-beta`
on the CLI** and explicitly documents:

> The 42/44 runs did not pass `--drct-snr-gate`,
> `--drct-snr-threshold`, or `--drct-sigma-ema-beta` on the CLI, so we
> do not pass them here either. Defaults at this commit are
> `drct_snr_gate=False` (store_true, off), `drct_snr_threshold=1.0`,
> `drct_sigma_ema_beta=0.0`.

The three supp-seed provenance files
(`runpod_supp_{cifar100,fmnist,cifar10}_K20_rho25_seeds454647_PROVENANCE.md`)
repeat the same inference. `RUNPOD_SUPP_SEEDS_TO_RUN.md` under "Known
caveats" explicitly notes that the supp commands assume the 42/43/44
runs used "gate off, threshold 1.0, ema_beta 0.0, warmup 0, held-out
test pool true".

The RunPod handler (`runpod/handler.py`) does not inject any DRCT
flags; it only auto-adds `--data-root`, `--feature-cache-dir`, and
`--n-workers` before appending user-supplied `extra_args`. So the
submission-command CLI surface is the authoritative record of what
each run received.

---

## 4. Conclusion

**CONFIRMED GATE OFF across all 8 files.**

All three independent lines of evidence agree:

1. **Raw JSON config blocks:** None of the 8 files contain any
   `drct_*` keys or SNR/warmup diagnostic logs — `config.json` is
   truncated to the pre-9c55ed0 field set and the FedProTrack
   `summary.json` per-method block contains only accuracy / AUC /
   bytes / clustering metrics with no runtime echo.
2. **Dataclass defaults at run-time commits:** `drct_snr_gate` is True
   in `TwoPhaseConfig`, but this default is **overridden by the CLI
   pathway**. The dataclass default is moot because the benchmark
   script always passes the CLI value as an explicit kwarg.
3. **CLI defaults + submission commands:** `--drct-snr-gate` is a
   `store_true` flag with CLI default False; the preserved retry
   provenance for seed 43 (K20/rho17) explicitly confirms that the
   original 42/44 submission commands did not include the flag, so the
   effective runtime value was False.

For `eval_on_test_pool` the story is simpler: both paths (CLI
default="true" at `94a1a2f`+ and dataclass default `True` at `73d6f78`)
land on held-out evaluation. **Held-out was ON.**

### Paper narrative mismatch (flag for revision)

The paper claims in §4.4, §5.2 and Appendix
`app:shrinkage_ablation` that the SNR gate is the default and "fires
on 1–3 early rounds". The main Table 1 numbers were in fact produced
with `drct_snr_gate=False` (pure Stein shrinkage, no gate). The
shrinkage-variant ablation table (Table 3) reports different CIFAR-10
FPT numbers from the main Table 1 (0.855 vs 0.794), consistent with
those two tables being different runs: the ablation run explicitly
enabled the gate, while the Table 1 flagship sweep did not.

Any reviewer cross-checking Table 1 and the "SNR gate on by default"
claim will notice this discrepancy. Either:
(a) the paper text should be softened to say the gate is an *optional*
extension that the ablation studies, not the default used in Table 1;
or
(b) Table 1 should be re-run with `--drct-snr-gate` explicitly enabled
to match the claim.

---

## 5. Implications for supplementary seeds 45/46/47

To pool seeds 42–47 as a valid n=6 paired-seed table, seeds 45/46/47
must reproduce the same runtime configuration as 42/43/44. That means:

- `--drct-snr-gate` → **OMIT** (store_true flag; omission = False,
  matching 42/43/44).
- `--drct-warmup-rounds 0` → optional (matches both CLI default and
  original); including it is defensive per the 5-point protocol.
- `--drct-sigma-ema-beta 0.0` → optional, same story.
- `--drct-snr-threshold 1.0` → optional, same story (and inert when
  gate is off).
- `--eval-on-test-pool true` → **INCLUDE** explicitly (matches
  original behaviour; the original runs implicitly hit this value via
  the dataclass default at `73d6f78` and via the CLI default at
  `94a1a2f`, so explicitly passing `true` is a no-op but defensive).

### Audit of `RUNPOD_SUPP_SEEDS_TO_RUN.md` submission commands

The current submission commands in `RUNPOD_SUPP_SEEDS_TO_RUN.md` are:

```
--eval-on-test-pool true
--drct-snr-threshold 1.0
--drct-sigma-ema-beta 0.0
--drct-warmup-rounds 0
```

and they **omit** `--drct-snr-gate`. This is exactly the
reproducibility-consistent configuration. **No flag changes needed**
— the submission commands as written are valid for n=6 pooling with
seeds 42/43/44.

### One residual caution

Seeds 42 and 44 for the flagship K20/ρ25 CIFAR-100 run (the
`runpod_flagship_all_methods.json` file) were executed when HEAD was
between `18847f5` (Apr 16 15:18) and `73d6f78` (Apr 16 15:54).
Supplementary seeds 45/46/47 will be executed at HEAD = `55f7900`.
Between those commits the only changes touching any paper-critical
code path were:
- `73d6f78` — WIP sync, did not modify SNR / eval / DRCT code.
- `3754aa9` — FedDrift spawn threshold (FedDrift-only).
- `94a1a2f` — paper rewrite + added `--eval-on-test-pool` and
  `--fmow-n-classes` CLI flags. The functional behaviour is unchanged
  for CIFAR-10/CIFAR-100/F-MNIST when those flags are passed
  explicitly as the submission command does.
- `4e9cfd6` / `0752a94` / `55f7900` — paper-only and appendix-only
  edits.

Therefore seeds 42/44 (at `73d6f78`-ish) and seeds 45/46/47 (at
`55f7900`) should produce bit-comparable results under the submission
commands already in `RUNPOD_SUPP_SEEDS_TO_RUN.md`.
