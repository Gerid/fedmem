# PROVENANCE — CIFAR-100 K=20 rho=17 seed 43 retry

**Purpose**: Retry seed 43 for the paper Table 1 `K=20 / rho=17` column. A
prior RunPod API call on 2026-04-17 timed out (`Job
5aaf987f-2db0-41f0-a8a3-19142a95bd90-u1 timed out after 1800s`), leaving the
column with only 2 seeds (42, 44). Filling seed 43 lets the paper drop the
`^dagger` footnote "`K20/rho17`: 2 seeds (1 RunPod timeout)".

This is a **retry of the same logical experiment**, not a new experiment —
the config is copied verbatim from the seeds that succeeded.

## Reproducibility anchors (5-point protocol)

1. **Git tag**: `exp/table1-k20-rho17-seed43-20260418`
2. **Git SHA**: `55f79006b16395b1e1edef3fdec1c8611b3cb9fc`
3. **HEAD commit title**: "Fix stale appendix data: update 26-method table, add protocol notes"
4. **Frozen config**: [`runpod_heldout_cifar100_K20_rho17_seed43_config.json`](./runpod_heldout_cifar100_K20_rho17_seed43_config.json)
5. **Source of truth for config values**: [`runpod_heldout_cifar100_K20_rho17_all.json`](./runpod_heldout_cifar100_K20_rho17_all.json) → `["42"].results["config.json"]` (identical structure in `["44"]`).

## Environment

- Submitter host: WSL2 on Windows, `/mnt/e/fedprotrack`
- RunPod Serverless endpoint: per `/run-experiment` skill config (API key via `RUNPOD_API_KEY`, endpoint via `RUNPOD_ENDPOINT_ID`)
- RunPod image: per `runpod/Dockerfile` on the endpoint
- Remote feature cache: `/runpod-volume/cache/.feature_cache/` (keyed by `cifar100_recurrence_c0_delta85_spc120_nf128_fseed2718_lsnone_nc17.npz` for this config; handler injects `--feature-cache-dir` and `--data-root` automatically)
- Local companion cache (for reference only, not used on RunPod): `/mnt/e/fedprotrack/.feature_cache/cifar100_recurrence_c0_delta85_spc120_nf128_fseed2718_lsnone_nc17.npz` mtime 2026-04-09 17:18

## Submission command (canonical — to be run from /mnt/e/fedprotrack)

```bash
python runpod/submit_experiment.py \
  --script run_cifar100_neurips_benchmark.py \
  --seeds 43 \
  --fpt-mode ot \
  --timeout 2700 \
  --out-file runpod_heldout_cifar100_K20_rho17_seed43.json \
  --extra-args \
    --dataset cifar100 \
    --K 20 \
    --T 100 \
    --rho 17 \
    --alpha 0.75 \
    --delta 0.85 \
    --n-samples 400 \
    --n-features 128 \
    --samples-per-coarse-class 120 \
    --batch-size 128 \
    --drift-type recurrent \
    --federation-every 1 \
    --participation 1 \
    --lr 0.02 \
    --fpt-lr 0.02 \
    --n-epochs 10 \
    --fpt-epochs 10 \
    --feature-seed 2718 \
    --methods all \
    --n-seeds 1 \
    --eval-on-test-pool true
```

Notes on the CLI surface:

- `--seed 43` is injected by the RunPod handler (it passes a single-seed
  override). `--seeds 43` above is read by `submit_experiment.py` to pick the
  single worker; `--n-seeds 1` keeps the script's seed loop at 1.
- `--fpt-mode ot` reproduces the exact `fpt_mode` used by the 42/44 runs
  (confirmed via the seed 42/44 `results.config.json.fpt_mode`).
- The 42/44 runs did **not** pass `--drct-snr-gate`, `--drct-snr-threshold`,
  or `--drct-sigma-ema-beta` on the CLI, so we do not pass them here either.
  Defaults at this commit are `drct_snr_gate=False` (store_true, off),
  `drct_snr_threshold=1.0`, `drct_sigma_ema_beta=0.0`. These defaults have
  not moved in the commits underlying the 42/44 runs.
- `--eval-on-test-pool true` is passed explicitly (defensive — also the
  default) to obey the "paper-critical params explicit" rule.
- Timeout bumped from 1800s → 2700s to give headroom over the prior failure
  (typical run took ~505s on seed 42, ~450s on seed 44).

## Timestamp

Retry launched 2026-04-18, by an automated session. Previous timed-out
attempt: 2026-04-17 (job id `5aaf987f-2db0-41f0-a8a3-19142a95bd90-u1`).

## Link to prior results (for cross-checking)

- `runpod_heldout_cifar100_K20_rho17_all.json` — seeds 42 & 44 COMPLETED with
  FedProTrack acc 0.8305 (seed 42) and FedAvg acc 0.7498 (seed 42). Seed 44
  similar. Seed 43 should land in a comparable regime.

## Post-run checklist

After this job completes:

1. Save returned JSON as `runpod_heldout_cifar100_K20_rho17_seed43.json`.
2. Merge seed 43 into the aggregate file if downstream analysis reads a
   single merged JSON.
3. Update paper `paper/main.tex`: remove the `^dagger` footnote in Table 1
   for `K=20 / rho=17`.
4. Re-run the mean/std CI computation over seeds {42, 43, 44} for that
   column.
