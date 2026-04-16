# RunPod Serverless Optimization — Findings & Changes

Date: 2026-04-12
Status: Phase 1 deployed, Phase 2 pending

## Context

FedProTrack experiments run on RunPod Serverless (endpoint `ewt09p168i0ll2`).
Workload: linear classifiers on frozen ResNet features (~1290 params), 26
federated methods × 3 datasets × 3 seeds per sweep.  GPU utilisation <0.1% —
the bottleneck is Python interpreter overhead, not FLOPS.

## Diagnoses

### 1. GPU tier mismatch (fixed)

RTX 4090 ($0.34/hr pod) was used for a workload that needs <1% of its compute.
A4000 ($0.17/hr) runs identical wall time for our linear-on-features pipeline.

**Fix**: Added `NVIDIA RTX A4000` and `NVIDIA RTX 4000 Ada Generation` as
preferred GPU types via REST PATCH.  4090 remains as fallback for CNN workloads.

### 2. `refresh_worker=True` killed warm-worker benefits (fixed)

Old handler.py line 147: `refresh_worker=True`.  This restarts the worker
process after every request, destroying:
- Module-level Python imports (~3-5s per `import torch`)
- Linux page cache for .npz feature files (~5-10s per cache miss on NAS)
- `--help` probe results (2-3s per script per request)

Per RunPod docs: "load models at module level, not inside handler()"
and "module-level initialization runs once per container with
`MODEL = load_model()` being cached across warm starts."

Reference: runpod-workers/worker-vllm#111 confirms flashboot alone doesn't
help if handler architecture re-initializes on every request.

**Fix**: `refresh_worker=False`.  Code extraction now uses MD5 hash comparison
(idempotent, no lock needed).  `--help` probe results cached in module dict.

### 3. Per-request subprocess overhead (mitigated, not eliminated)

Each request forks a new Python process (`subprocess.run([python, script.py])`)
which re-imports torch + fedprotrack + reads feature cache.

Full fix (direct function import) requires refactoring experiment scripts to
expose callable entry points — deferred to Phase 2.

**Mitigation**: `_prewarm_page_cache()` at module level reads headers of all
.npz files, populating Linux page cache.  Subprocess reads then hit RAM instead
of NAS.  FlashBoot captures this state in the snapshot.

## Endpoint Configuration (current)

```
Endpoint ID:     ewt09p168i0ll2
workersMax:      10  (was 5)
workersStandby:  5
idleTimeout:     30  (was 5)
flashboot:       true
gpuTypeIds:      [NVIDIA RTX A4000, NVIDIA RTX 4000 Ada, NVIDIA GeForce RTX 4090]
executionTimeout: 600s
scalerType:      QUEUE_DELAY (scalerValue=4)
```

## Network Volume (7kga5cg276, 100 GB, $7/mo)

Measured 2026-04-11 via du probe job:

| Directory | Size | Notes |
|-----------|------|-------|
| pip_packages/ | 3.4 GB | Runtime deps, stable |
| cache/.cifar100_cache/ | 771 MB | CIFAR-100 raw images |
| cache/.feature_cache/ | 203 MB | Frozen ResNet features (.npz) |
| code/ | 222 MB | Project source |
| results/ | 46 MB | 17 experiment output dirs |
| **Total** | **~4.65 GB** | **4.7% of 100 GB** |

95 GB free — sufficient for adding Tiny-ImageNet, EMNIST, QMNIST.
Full ImageNet-1k (160 GB raw) would require volume expansion.

## Cost Observability (new)

`runpod/submit_experiment.py` now captures RunPod's `delayTime` and
`executionTime` from each job response and appends to
`tmp/runpod_cost_ledger.jsonl`.  Includes per-GPU-tier cost estimate.

## Estimated Impact

| Metric | Before | Phase 1 | Phase 2 (future) |
|--------|--------|---------|-------------------|
| GPU cost/hr | $0.34 (4090) | $0.17 (A4000) | same |
| Cold start | ~9s | ~3-5s (flashboot snapshot) | <1s (module import) |
| Per-request overhead | ~15-20s | ~5-8s (page cache) | ~0s (direct call) |
| Sweep wall time (9 jobs) | ~6 min | ~4 min | ~2 min |
| Sweep cost | ~$1.50 | ~$0.60 | ~$0.30 |

## Phase 2 Roadmap (not started)

1. **Direct function import** in handler.py — expose `main(config_dict)` in
   experiment scripts, call directly instead of subprocess.
2. **ProcessPoolExecutor** for intra-job method parallelism (6-way, 26→5 rounds).
3. **Multi-endpoint routing** — separate A4000 endpoint for linear workloads,
   4090 endpoint for CNN experiments (EMNIST end-to-end).

## Files Changed

- `runpod/handler.py` — full rewrite (refresh_worker=False, hash-based extract,
  help-cache, page-cache prewarm, symlink torch hub, timing in output)
- `runpod/submit_experiment.py` — `_runpod_meta` timing capture, cost ledger,
  GPU price table
- Endpoint config — 3× REST PATCH (workersMax, idleTimeout, gpuTypeIds)

## GitHub References

No existing open-source project does workload-aware GPU selection for RunPod
serverless.  All 36 repos under `runpod-serverless` GitHub topic are single-model
LLM/diffusion deployments.  Our handler pattern is novel for scientific-computing
batch workloads.  Key references used:

- https://github.com/justinwlin/Runpod-OpenLLM-Pod-and-Serverless (module-level load)
- https://github.com/runpod-workers/model-store-cache-example (volume cache)
- https://github.com/sruckh/parakeet-runpod (network volume warm start)
- https://github.com/runpod/runpod-python/blob/main/docs/serverless/worker.md
