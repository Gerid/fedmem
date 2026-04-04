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
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.time()
    while time.time() - t0 < timeout:
        resp = requests.get(url, headers=headers, timeout=30)
        data = resp.json()
        status = data.get("status", "UNKNOWN")
        if status == "COMPLETED":
            return data.get("output", {})
        elif status == "FAILED":
            raise RuntimeError(f"Job {job_id} FAILED: {json.dumps(data, indent=2)}")
        elapsed = int(time.time() - t0)
        print(f"    [{elapsed}s] {job_id}: {status}")
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} timed out after {timeout}s")


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
    args = parser.parse_args()

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

    # Poll all jobs
    results = {}
    for seed, job_id in jobs.items():
        try:
            output = poll_job(api_key, endpoint_id, job_id, timeout=args.timeout)
            rc = output.get("returncode", -1)
            status_icon = "OK" if rc == 0 else "FAIL"
            print(f"  seed={seed}: {status_icon} (rc={rc})")
            results[seed] = output
        except Exception as e:
            print(f"  seed={seed}: ERROR — {e}")
            results[seed] = {"error": str(e)}

    # Save aggregated results locally
    out_file = f"runpod_results_{args.script.replace('.py', '')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Summary
    ok = sum(1 for r in results.values() if r.get("returncode") == 0)
    print(f"\n=== Done: {ok}/{len(results)} seeds succeeded ===")


if __name__ == "__main__":
    main()
