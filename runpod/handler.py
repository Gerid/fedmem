"""RunPod Serverless handler — reads code & data from Network Volume."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import runpod

# Network Volume mount point (RunPod convention)
VOL = "/runpod-volume"
CODE_DIR = f"{VOL}/code"
CACHE_DIR = f"{VOL}/cache"
RESULTS_DIR = f"{VOL}/results"


def _auto_extract_code():
    """Extract code.tar.gz on volume if sync_code.py uploaded a new one.

    Uses a lock file to prevent race conditions when multiple workers
    start simultaneously.
    """
    tar_path = f"{VOL}/code.tar.gz"
    marker = f"{VOL}/code_needs_extract"
    lock_path = f"{VOL}/code_extract.lock"

    if not os.path.isfile(tar_path):
        return

    # Try to atomically claim the extract job via lock file.
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        # Another worker is extracting. Wait for it.
        import time
        for _ in range(30):
            if not os.path.isfile(lock_path):
                break
            time.sleep(1)
        return

    try:
        if os.path.isfile(marker):
            print("Auto-extracting code.tar.gz ...")
            os.makedirs(CODE_DIR, exist_ok=True)
            subprocess.run(
                ["tar", "xzf", tar_path, "-C", CODE_DIR],
                check=True,
            )
            try:
                os.remove(marker)
            except FileNotFoundError:
                pass  # Another worker already removed it
            print(f"Code extracted to {CODE_DIR}")
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def handler(job: dict) -> dict:
    """Run a FedProTrack experiment script on GPU."""
    job_input = job["input"]
    script = job_input.get("script", "run_cifar100_all_baselines_smoke.py")
    seed = job_input.get("seed", 42)
    extra_args = job_input.get("extra_args", [])
    fpt_mode = job_input.get("fpt_mode", "base")

    # Auto-extract code if a new tarball was uploaded
    _auto_extract_code()

    experiment_name = script.replace(".py", "")
    out_dir = f"{RESULTS_DIR}/{experiment_name}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Build command — mirrors k8s/submit-parallel-seeds.sh
    # Only add --data-root/--feature-cache-dir/--n-workers for scripts that
    # use real data (detect by checking if script has these args).
    cmd = [
        sys.executable, script,
        "--seed", str(seed),
        "--results-dir", out_dir,
    ]
    # Probe which optional args the script accepts.
    probe = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=CODE_DIR, capture_output=True, text=True, env=env,
    )
    help_text = probe.stdout + probe.stderr
    if "--data-root" in help_text:
        cmd.extend(["--data-root", f"{CACHE_DIR}/.cifar100_cache"])
    if "--feature-cache-dir" in help_text:
        cmd.extend(["--feature-cache-dir", f"{CACHE_DIR}/.feature_cache"])
    if "--n-workers" in help_text:
        cmd.extend(["--n-workers", "0"])
    if fpt_mode != "base":
        cmd.extend(["--fpt-mode", fpt_mode])
    cmd.extend(extra_args)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"

    # Pre-cache torch hub weights from volume (if available)
    hub_src = f"{CACHE_DIR}/.torch_hub/checkpoints"
    hub_dst = "/root/.cache/torch/hub/checkpoints"
    if os.path.isdir(hub_src):
        os.makedirs(hub_dst, exist_ok=True)
        subprocess.run(["cp", "-n"] + [
            os.path.join(hub_src, f) for f in os.listdir(hub_src)
        ] + [hub_dst], check=False)

    print(f"=== Seed {seed} | Script {script} | Mode {fpt_mode} ===")
    result = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True, env=env)
    print(result.stdout[-3000:] if result.stdout else "(no stdout)")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])

    # Collect JSON results
    output_data = {}
    if os.path.isdir(out_dir):
        for fname in os.listdir(out_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(out_dir, fname)
                try:
                    with open(fpath) as f:
                        output_data[fname] = json.load(f)
                except Exception as e:
                    output_data[fname] = {"error": str(e)}

    return {
        "seed": seed,
        "script": script,
        "fpt_mode": fpt_mode,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-5000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        "results": output_data,
    }


# refresh_worker=True: Worker 每次任务结束后自动重启，
# 下次任务自动加载最新的代码和 handler，无需手动删 Worker
runpod.serverless.start({"handler": handler, "refresh_worker": True})
