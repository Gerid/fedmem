# PROVENANCE: Supplementary seeds 45/46/47 - CIFAR-100 K=20 rho=25

## Purpose
Raise Table 1 statistical power from n=3 (seeds 42/43/44) to n=6 (+ seeds
45/46/47) for the CIFAR-100 flagship weak config. Current paired t-test
p-values on weak configs are 0.07 to 0.42; supplementary seeds are needed
to meet NeurIPS 2026 statistical-reporting standards.

## Reference
- Config source: `/mnt/e/fedprotrack/runpod_flagship_all_methods.json`
  (seed 42 complete) and `/mnt/e/fedprotrack/runpod_flagship_seed43_retry.json`
  (seed 43 retry). Seed 44 was part of the original flagship batch.
- Frozen config: `/mnt/e/fedprotrack/runpod_supp_cifar100_K20_rho25_seeds454647_config.json`

## Git state
- Git SHA at preparation: `55f7900` (verify with `git rev-parse HEAD` before submit;
  update this file if HEAD differs).
- Proposed tag: `exp/supp-cifar100-seeds454647-20260418`
- Create tag before submission:
  `cd /mnt/e/fedprotrack && git tag exp/supp-cifar100-seeds454647-20260418`

## Submission command (copy-paste-ready)

```bash
cd /mnt/e/fedprotrack && python runpod/submit_experiment.py \
  --script run_cifar100_neurips_benchmark.py \
  --fpt-mode ot \
  --seeds 45 46 47 \
  --timeout 2700 \
  --out-file runpod_supp_cifar100_K20_rho25_seeds454647.json \
  --extra-args \
    --dataset cifar100 \
    --K 20 --T 100 \
    --rho 25 --alpha 0.75 --delta 0.85 \
    --drift-type recurrent \
    --n-features 128 --n-samples 400 \
    --samples-per-coarse-class 120 \
    --batch-size 128 --participation 1.0 \
    --federation-every 1 \
    --feature-seed 2718 \
    --lr 0.02 --n-epochs 10 \
    --fpt-lr 0.02 --fpt-epochs 10 \
    --methods all \
    --model-type linear \
    --eval-on-test-pool true \
    --drct-snr-threshold 1.0 \
    --drct-sigma-ema-beta 0.0 \
    --drct-warmup-rounds 0 \
    --n-seeds 1
```

Notes:
- `submit_experiment.py` submits one parallel RunPod job per seed, each
  forwards the `--extra-args` tail (plus `--seed <N>`) to the remote
  `run_cifar100_neurips_benchmark.py` entrypoint.
- `--drct-snr-gate` is deliberately OMITTED (flag is `store_true`;
  omission = gate OFF) to match seeds 42/43/44 behaviour inferred from the
  reference config.json, which did not record any drct_* fields. IF the
  original runs had the gate ON, the n=3 and n=6 results are NOT directly
  comparable for a paired test — VERIFY before submitting.
- `--n-classes` is NOT a CLI flag for cifar100; class count is fixed by
  the dataset generator. The user-prompt template included it, but the
  underlying benchmark script does not accept it.

## Expected output
- File: `/mnt/e/fedprotrack/runpod_supp_cifar100_K20_rho25_seeds454647.json`
- Timestamp: <FILL_AFTER_COMPLETION>
- Structure: top-level keys `"45"`, `"46"`, `"47"`; each with
  `results.summary.json` containing 26 method entries.
- Timeout: 2700 s per seed; 3 seeds run in parallel on RunPod Serverless.

## Parameters that could NOT be determined from reference configs
These fields are NOT present in the reference `config.json` blocks; values
below are taken from script defaults (per `run_cifar100_neurips_benchmark.py`
argparse at lines ~1055-1080). **Confirm with the original runs' full
command log / PROVENANCE if one exists before trusting these:**
- `drct_snr_gate` -> False (script default)
- `drct_snr_threshold` -> 1.0 (script default)
- `drct_sigma_ema_beta` -> 0.0 (script default)
- `drct_warmup_rounds` -> 0 (script default)
- `eval_on_test_pool` -> "true" (script default)
- `model_type` -> "linear" (script default)

All other paper-critical knobs (K, T, rho, alpha, delta, n_features,
n_samples, samples_per_coarse_class, feature_seed, fpt_mode, fpt_lr,
fpt_epochs, lr, n_epochs, methods) were recorded in the reference
config.json and are passed explicitly above.

## Post-run verification
1. `jq 'keys' runpod_supp_cifar100_K20_rho25_seeds454647.json` -> ["45","46","47"]
2. For each seed: `jq '.["45"].results["summary.json"] | keys | length'`
   should return 26.
3. Paired-t combined across seeds 42-47 to update Table 1 statistical
   power column.
