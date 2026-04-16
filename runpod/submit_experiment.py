"""Submit FedProTrack experiments to RunPod Serverless.

Usage:
    python runpod/submit_experiment.py --script run_cifar100_all_baselines_smoke.py --seeds 42 43 44 45 46
    python runpod/submit_experiment.py --script run_cifar100_all_baselines_smoke.py --seeds 42 --fpt-mode calibrated

Environment variables:
    RUNPOD_API_KEY      — from RunPod Settings > API Keys
    RUNPOD_ENDPOINT_ID  — from Serverless > your endpoint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests")
    sys.exit(1)


def get_config():
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "")
    if not api_key:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        print("  Get it from: RunPod Console > Settings > API Keys")
        sys.exit(1)
    if not endpoint_id:
        print("ERROR: Set RUNPOD_ENDPOINT_ID environment variable")
        print("  Get it from: RunPod Console > Serverless > your endpoint")
        sys.exit(1)
    return api_key, endpoint_id


def submit_seed(api_key: str, endpoint_id: str, script: str, seed: int,
                fpt_mode: str = "base", extra_args: list[str] | None = None) -> str:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "input": {
            "script": script,
            "seed": seed,
            "fpt_mode": fpt_mode,
            "extra_args": extra_args or [],
        }
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    return job_id


def poll_job(api_key: str, endpoint_id: str, job_id: str,
             interval: int = 10, timeout: int = 600) -> dict:
    """Poll a RunPod job to completion.

    Returns the `output` dict from RunPod, augmented with timing metadata
    under the `_runpod_meta` key so callers can inspect cold-start overhead
    and billed seconds without an extra round-trip.
    """
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        resp = requests.get(url, headers=headers, timeout=30)
        data = resp.json()
        status = data.get("status", "UNKNOWN")
        if status == "COMPLETED":
            output = data.get("output", {}) or {}
            # Attach timing metadata from RunPod's response. Fields:
            #   delayTime     - ms queued before a worker picked up the job
            #   executionTime - ms of billed worker execution (cold start + work)
            #   workerId      - which worker ran the job (useful to see reuse)
            if isinstance(output, dict):
                output["_runpod_meta"] = {
                    "job_id": job_id,
                    "delay_ms": data.get("delayTime"),
                    "execution_ms": data.get("executionTime"),
                    "worker_id": data.get("workerId"),
                    "polled_elapsed_s": round(time.time() - t0, 1),
                }
            return output
        elif status == "FAILED":
            raise RuntimeError(f"Job {job_id} FAILED: {json.dumps(data, indent=2)}")
        elapsed = int(time.time() - t0)
        print(f"    [{elapsed}s] {job_id}: {status}")
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


# RunPod Serverless pricing per GPU tier, $/second (flex active).
# Update when RunPod changes their published rates.
_PRICE_PER_SECOND = {
    "NVIDIA GeForce RTX 4090": 0.00031,
    "NVIDIA RTX A4000": 0.00016,
    "NVIDIA RTX A5000": 0.00019,
    "NVIDIA L4": 0.00019,
    "NVIDIA L40": 0.00053,
    "NVIDIA L40S": 0.00053,
    "NVIDIA A100 80GB PCIe": 0.00116,
    "NVIDIA H100 80GB HBM3": 0.00246,
}


def append_cost_ledger(ledger_path: str, script: str, job_id: str,
                       meta: dict, returncode: int,
                       gpu_type_id: str | None = None) -> None:
    """Append one job's timing + estimated cost to a local JSONL ledger.

    The ledger is a simple append-only file so multiple submissions can write
    concurrently without locking. Use it to diagnose cold-start overhead,
    per-worker reuse, and running spend without querying RunPod again.
    """
    if not meta:
        return
    exec_ms = meta.get("execution_ms") or 0
    price_per_s = _PRICE_PER_SECOND.get(gpu_type_id or "", 0.00031)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": script,
        "job_id": job_id,
        "worker_id": meta.get("worker_id"),
        "delay_ms": meta.get("delay_ms"),
        "execution_ms": exec_ms,
        "execution_s": round(exec_ms / 1000, 2) if exec_ms else 0,
        "gpu_type": gpu_type_id or "unknown",
        "cost_usd": round((exec_ms / 1000) * price_per_s, 6) if exec_ms else 0,
        "returncode": returncode,
    }
    try:
        os.makedirs(os.path.dirname(ledger_path) or ".", exist_ok=True)
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        print(f"  [ledger] failed to append {job_id}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Submit FedProTrack experiments to RunPod")
    parser.add_argument("--script", required=True, help="Experiment script name")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds (each gets a parallel worker)")
    parser.add_argument("--fpt-mode", default="base",
                        help="FPT mode (base/calibrated/hybrid/...)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-job timeout in seconds")
    parser.add_argument("--extra-args", nargs="*", default=[],
                        help="Extra args passed to the experiment script")
    parser.add_argument("--out-file", default=None,
                        help="Output JSON path (defaults to runpod_results_<script>.json). "
                             "Use a unique name when running multiple submissions in parallel "
                             "to avoid overwriting results.")
    # Use parse_known_args so unknown flags (e.g. --K, --rho) are also
    # forwarded to the remote script, not rejected locally.
    args, passthrough = parser.parse_known_args()
    args.extra_args = list(args.extra_args) + list(passthrough)

    api_key, endpoint_id = get_config()

    print(f"=== RunPod Experiment Submission ===")
    print(f"  Script:   {args.script}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  FPT mode: {args.fpt_mode}")
    print()

    # Submit all seeds in parallel
    jobs = {}
    for seed in args.seeds:
        job_id = submit_seed(api_key, endpoint_id, args.script, seed,
                             args.fpt_mode, args.extra_args)
        jobs[seed] = job_id
        print(f"  Submitted seed={seed} -> {job_id}")

    print(f"\n=== Waiting for {len(jobs)} jobs ===")

    # Cost ledger lives under tmp/ so it's git-ignored and survives across runs.
    ledger_path = os.path.join("tmp", "runpod_cost_ledger.jsonl")

    # Poll all jobs
    results = {}
    total_cost = 0.0
    for seed, job_id in jobs.items():
        try:
            output = poll_job(api_key, endpoint_id, job_id, timeout=args.timeout)
            rc = output.get("returncode", -1)
            status_icon = "OK" if rc == 0 else "FAIL"
            meta = output.get("_runpod_meta", {})
            exec_s = (meta.get("execution_ms") or 0) / 1000
            delay_s = (meta.get("delay_ms") or 0) / 1000
            print(f"  seed={seed}: {status_icon} (rc={rc})  "
                  f"delay={delay_s:.1f}s  exec={exec_s:.1f}s  worker={meta.get('worker_id', '?')}")
            append_cost_ledger(ledger_path, args.script, job_id, meta, rc)
            # Sum cost using the cheapest plausible tier (A4000 preferred by endpoint).
            total_cost += exec_s * _PRICE_PER_SECOND.get("NVIDIA RTX A4000", 0.00016)
            results[seed] = output
        except Exception as e:
            print(f"  seed={seed}: ERROR — {e}")
            results[seed] = {"error": str(e)}
    print(f"\n  [ledger] appended {len(results)} rows to {ledger_path}")
    print(f"  [cost]   est. total for this batch: ${total_cost:.4f} (A4000 rate)")

    # Save aggregated results locally
    if args.out_file:
        out_file = args.out_file
    else:
        out_file = f"runpod_results_{args.script.replace('.py', '')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Summary
    ok = sum(1 for r in results.values() if r.get("returncode") == 0)
    print(f"\n=== Done: {ok}/{len(results)} seeds succeeded ===")


if __name__ == "__main__":
    main()
