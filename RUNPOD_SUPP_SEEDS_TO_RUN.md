# Supplementary seeds 45/46/47 for NeurIPS Table 1 statistical power

The three weak configs currently have n=3 seeds with paired t-test
p-values in the 0.07-0.42 range. Adding seeds 45/46/47 (n=6) brings the
paired comparison up to NeurIPS 2026 statistical-reporting standards
without re-running any of the already-complete 42/43/44 rows.

## Run these git tags BEFORE submitting (5-point reproducibility checklist)

```bash
cd /mnt/e/fedprotrack
git tag exp/supp-cifar100-seeds454647-20260418
git tag exp/supp-fmnist-seeds454647-20260418
git tag exp/supp-cifar10-seeds454647-20260418
```

Verify `git rev-parse HEAD` returns `55f7900` (the assumed commit at prep
time). If HEAD has moved, decide whether the supplementary seeds should
match the original commit (check out `55f7900` first, then tag + submit)
or the current tip (update the tag target and the three PROVENANCE files
accordingly).

## Cost and wall-time estimates
- ~$25 total across all 3 jobs (RunPod A4000 rate; 26 methods x 3 seeds
  x ~30 min).
- Wall time: 3 submissions run in parallel (each submission internally
  forks one RunPod worker per seed, 3 workers per submission). Expect
  ~30 min per submission if all 9 workers are cold-started
  simultaneously. Submit the three below as separate background shells.

## 1. CIFAR-100 K=20 rho=25 (supplementary)
Raises the flagship CIFAR-100 weak row from n=3 to n=6.

- Config JSON: `/mnt/e/fedprotrack/runpod_supp_cifar100_K20_rho25_seeds454647_config.json`
- PROVENANCE: `/mnt/e/fedprotrack/runpod_supp_cifar100_K20_rho25_seeds454647_PROVENANCE.md`

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

## 2. F-MNIST K=20 rho=25 (supplementary)
Raises the F-MNIST held-out weak row from n=3 to n=6.

- Config JSON: `/mnt/e/fedprotrack/runpod_supp_fmnist_K20_rho25_seeds454647_config.json`
- PROVENANCE: `/mnt/e/fedprotrack/runpod_supp_fmnist_K20_rho25_seeds454647_PROVENANCE.md`

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

## 3. CIFAR-10 K=20 rho=25 (supplementary)
Raises the CIFAR-10 held-out weak row from n=3 to n=6.

- Config JSON: `/mnt/e/fedprotrack/runpod_supp_cifar10_K20_rho25_seeds454647_config.json`
- PROVENANCE: `/mnt/e/fedprotrack/runpod_supp_cifar10_K20_rho25_seeds454647_PROVENANCE.md`

```bash
cd /mnt/e/fedprotrack && python runpod/submit_experiment.py \
  --script run_cifar100_neurips_benchmark.py \
  --fpt-mode ot \
  --seeds 45 46 47 \
  --timeout 2700 \
  --out-file runpod_supp_cifar10_K20_rho25_seeds454647.json \
  --extra-args \
    --dataset cifar10 \
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

## Success verification (after each output file lands)

For each of the three output JSONs, run:

```bash
jq 'keys' runpod_supp_<dataset>_K20_rho25_seeds454647.json
# -> ["45","46","47"]

for s in 45 46 47; do
  echo -n "seed $s methods: "
  jq ".[\"$s\"].results[\"summary.json\"] | keys | length" \
    runpod_supp_<dataset>_K20_rho25_seeds454647.json
done
# -> 26, 26, 26 (confirms all 26 dedupe'd methods populated)
```

If any seed reports fewer than 26 methods, inspect its
`results.stderr.txt` entry in the JSON and re-submit that single seed.

## Known caveats flagged in per-run PROVENANCE
- `drct_*` and `eval_on_test_pool` were NOT recorded in the reference
  `config.json` blocks for seeds 42/43/44; the supplementary commands
  above assume they used script defaults (gate off, threshold 1.0,
  ema_beta 0.0, warmup 0, held-out test pool true). Confirm with any
  remaining log of the original submissions before trusting a paired
  t-test on n=6.
- `--n-classes` mentioned in the user-prompt template is NOT a valid CLI
  flag on `run_cifar100_neurips_benchmark.py` for these three datasets —
  class counts are fixed by the dataset generators.
- The reference file for CIFAR-100 covers seeds 42 (flagship) and 43
  (retry). Seed 44's original result file was not explicitly identified
  but its config is expected to match by construction.
