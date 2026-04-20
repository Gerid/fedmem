# PROVENANCE: Supplementary seeds 45/46/47 - F-MNIST K=20 rho=25

## Purpose
Raise Table 1 statistical power from n=3 (seeds 42/43/44) to n=6 (+ seeds
45/46/47) for the F-MNIST held-out weak config. Current paired t-test
p-values on weak configs are 0.07 to 0.42; supplementary seeds are needed
to meet NeurIPS 2026 statistical-reporting standards.

## Reference
- Config source: `/mnt/e/fedprotrack/runpod_heldout_fmnist_all.json`
  (seeds 42, 43, 44 complete; 26 methods x 3 seeds in summary).
- Frozen config: `/mnt/e/fedprotrack/runpod_supp_fmnist_K20_rho25_seeds454647_config.json`

## Git state
- Git SHA at preparation: `55f7900` (verify with `git rev-parse HEAD` before submit;
  update this file if HEAD differs).
- Proposed tag: `exp/supp-fmnist-seeds454647-20260418`
- Create tag before submission:
  `cd /mnt/e/fedprotrack && git tag exp/supp-fmnist-seeds454647-20260418`

## Submission command (copy-paste-ready)

```bash
cd /mnt/e/fedprotrack && python runpod/submit_experiment.py \
  --script run_cifar100_neurips_benchmark.py \
  --fpt-mode ot \
  --seeds 45 46 47 \
  --timeout 2700 \
  --out-file runpod_supp_fmnist_K20_rho25_seeds454647.json \
  --extra-args \
    --dataset fmnist \
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
- `--dataset fmnist` MUST be passed explicitly; the script's `--dataset`
  default is `cifar100`.
- F-MNIST internally uses `samples_per_class` (10 classes native),
  populated from the CLI's `--samples-per-coarse-class 120` value.
- `--drct-snr-gate` is deliberately OMITTED (flag is `store_true`;
  omission = gate OFF) to match seeds 42/43/44 behaviour inferred from the
  reference config.json, which did not record any drct_* fields. IF the
  original runs had the gate ON, the n=3 and n=6 results are NOT directly
  comparable for a paired test — VERIFY before submitting.
- `--n-classes` is NOT a CLI flag for fmnist; F-MNIST is 10-class and
  fixed in the dataset generator.

## Expected output
- File: `/mnt/e/fedprotrack/runpod_supp_fmnist_K20_rho25_seeds454647.json`
- Timestamp: <FILL_AFTER_COMPLETION>
- Structure: top-level keys `"45"`, `"46"`, `"47"`; each with
  `results.summary.json` containing 26 method entries.
- Timeout: 2700 s per seed; 3 seeds run in parallel on RunPod Serverless.

## Parameters that could NOT be determined from reference configs
These fields are NOT present in the reference `config.json` block; values
below are taken from script defaults. **Confirm with the original runs'
full command log / PROVENANCE if one exists:**
- `drct_snr_gate` -> False (script default)
- `drct_snr_threshold` -> 1.0 (script default)
- `drct_sigma_ema_beta` -> 0.0 (script default)
- `drct_warmup_rounds` -> 0 (script default)
- `eval_on_test_pool` -> "true" (script default)
- `model_type` -> "linear" (script default)
- `dataset` -> "fmnist" (inferred from filename `runpod_heldout_fmnist_all.json`;
  the reference config.json does NOT echo the dataset value, so this is a
  filename-based inference. Confirm if uncertain.)

All other paper-critical knobs were recorded in the reference config.json
and are passed explicitly above.

## Post-run verification
1. `jq 'keys' runpod_supp_fmnist_K20_rho25_seeds454647.json` -> ["45","46","47"]
2. For each seed: `jq '.["45"].results["summary.json"] | keys | length'`
   should return 26.
3. Paired-t combined across seeds 42-47 to update Table 1 statistical
   power column.
