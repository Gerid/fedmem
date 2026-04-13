"""RunPod Serverless handler — reads code & data from Network Volume.

Optimised for warm-worker reuse (refresh_worker=False):
  - Worker process stays alive across requests → module-level imports
    and Linux page-cache for .npz feature files persist.
  - Code extraction uses a content hash so a new tarball is unpacked
    exactly once; subsequent requests on the same worker skip it.
  - The --help probe that detects which CLI flags a script supports is
    cached per script name so the subprocess is only spawned once per
    worker lifetime.

Reference patterns:
  - justinwlin/Runpod-OpenLLM-Pod-and-Serverless (module-level model load)
  - runpod-workers/model-store-cache-example (HF_HUB_OFFLINE cached model)
  - RunPod docs: "load models at module level, not inside handler()"
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

import runpod

# ── Paths on the Network Volume ──────────────────────────────────
VOL = "/runpod-volume"
CODE_DIR = f"{VOL}/code"
CACHE_DIR = f"{VOL}/cache"
RESULTS_DIR = f"{VOL}/results"

# ── Module-level caches (persist across requests when refresh=False) ─
# Maps script name -> set of supported CLI flags (e.g. {"--data-root", ...})
_HELP_CACHE: dict[str, set[str]] = {}

# Content hash of the last extracted code tarball so we don't re-extract.
_LAST_CODE_HASH: str | None = None


# ── Warm-start: pre-populate Linux page cache for feature files ──
def _prewarm_page_cache() -> None:
    """Read a few KB from each .npz feature file so the OS caches the
    filesystem pages.  Subsequent reads from experiment subprocesses
    hit RAM instead of NAS.  Takes <200 ms for the current cache size.
    """
    feature_dir = os.path.join(CACHE_DIR, ".feature_cache")
    if not os.path.isdir(feature_dir):
        return
    for fname in os.listdir(feature_dir):
        fpath = os.path.join(feature_dir, fname)
        try:
            with open(fpath, "rb") as f:
                f.read(8192)  # read header → kernel caches the inode + first pages
        except OSError:
            pass


# Run once at import time (worker startup).  FlashBoot snapshots capture
# this state, so subsequent cold starts skip the I/O entirely.
_prewarm_page_cache()


# ── Torch hub weights ────────────────────────────────────────────
def _symlink_torch_hub() -> None:
    """Make cached torch hub weights available at the default path."""
    hub_src = os.path.join(CACHE_DIR, ".torch_hub", "checkpoints")
    hub_dst = "/root/.cache/torch/hub/checkpoints"
    if not os.path.isdir(hub_src):
        return
    os.makedirs(hub_dst, exist_ok=True)
    for fname in os.listdir(hub_src):
        src = os.path.join(hub_src, fname)
        dst = os.path.join(hub_dst, fname)
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)  # symlink is instant, cp is slow
            except OSError:
                subprocess.run(["cp", "-n", src, dst], check=False)


_symlink_torch_hub()


# ── Code extraction ──────────────────────────────────────────────
def _md5_file(path: str) -> str:
    """Return hex MD5 of a file (fast for ~5 MB tarballs)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_extract_code() -> None:
    """Extract code.tar.gz only when its content actually changed.

    Uses an MD5 content hash instead of a marker file so:
      - Multiple workers can safely check in parallel (no lock needed
        because extract is idempotent on the same tarball).
      - Re-synced code (same name, new content) is always picked up.
      - Workers that already extracted the current version skip it.
    """
    global _LAST_CODE_HASH
    tar_path = f"{VOL}/code.tar.gz"
    if not os.path.isfile(tar_path):
        return

    current_hash = _md5_file(tar_path)
    if current_hash == _LAST_CODE_HASH:
        return  # same tarball, already extracted by this worker

    print(f"Extracting code.tar.gz ({current_hash[:12]}...) ...")
    os.makedirs(CODE_DIR, exist_ok=True)
    subprocess.run(
        ["tar", "xzf", tar_path, "-C", CODE_DIR],
        check=True,
    )
    _LAST_CODE_HASH = current_hash
    print(f"Code extracted to {CODE_DIR}")


# Extract once at startup (FlashBoot captures the result).
_maybe_extract_code()


# ── Script flag probing (cached) ─────────────────────────────────
def _get_script_flags(script: str, env: dict[str, str]) -> set[str]:
    """Return the set of CLI flags a script supports.

    Results are cached per script name so the subprocess only runs once
    per worker lifetime (~2-3 s saved on every subsequent request).
    """
    if script in _HELP_CACHE:
        return _HELP_CACHE[script]

    probe = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=CODE_DIR, capture_output=True, text=True, env=env,
        timeout=120,
    )
    text = probe.stdout + probe.stderr
    flags = set()
    for flag in ("--data-root", "--feature-cache-dir", "--n-workers"):
        if flag in text:
            flags.add(flag)
    _HELP_CACHE[script] = flags
    return flags


# ── Handler ──────────────────────────────────────────────────────
def handler(job: dict) -> dict:
    """Run a FedProTrack experiment script on GPU.

    This function is called once per request.  Heavy setup (code
    extraction, page-cache warming, hub weights) is done at module
    level so it happens once per worker, not once per request.
    """
    t0 = time.time()
    job_input = job["input"]
    script = job_input.get("script", "run_cifar100_all_baselines_smoke.py")
    seed = job_input.get("seed", 42)
    extra_args = job_input.get("extra_args", [])
    fpt_mode = job_input.get("fpt_mode", "base")

    # Check if code was re-synced since last request on this worker.
    _maybe_extract_code()

    experiment_name = script.replace(".py", "")
    out_dir = f"{RESULTS_DIR}/{experiment_name}/seed_{seed}"
    os.makedirs(out_dir, exist_ok=True)

    # Build command
    cmd = [
        sys.executable, script,
        "--seed", str(seed),
        "--results-dir", out_dir,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"

    # Use cached flag probe (only spawns subprocess on first call).
    flags = _get_script_flags(script, env)
    if "--data-root" in flags:
        cmd.extend(["--data-root", f"{CACHE_DIR}/.cifar100_cache"])
    if "--feature-cache-dir" in flags:
        cmd.extend(["--feature-cache-dir", f"{CACHE_DIR}/.feature_cache"])
    if "--n-workers" in flags:
        cmd.extend(["--n-workers", "0"])
    if fpt_mode != "base":
        cmd.extend(["--fpt-mode", fpt_mode])
    cmd.extend(extra_args)

    setup_ms = int((time.time() - t0) * 1000)
    print(f"=== Seed {seed} | Script {script} | Mode {fpt_mode} | setup {setup_ms}ms ===")

    t1 = time.time()
    result = subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True, env=env)
    work_ms = int((time.time() - t1) * 1000)

    print(result.stdout[-3000:] if result.stdout else "(no stdout)")
    if result.stderr:
        print("STDERR:", result.stderr[-2000:])
    print(f"=== Done | rc={result.returncode} | work {work_ms}ms | total {setup_ms + work_ms}ms ===")

    # Collect JSON results
    output_data: dict[str, object] = {}
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
        "setup_ms": setup_ms,
        "work_ms": work_ms,
        "stdout_tail": result.stdout[-5000:] if result.stdout else "",
        "stderr_tail": result.stderr[-2000:] if result.stderr else "",
        "results": output_data,
    }


# refresh_worker=False: keep this process alive across requests.
# Module-level state (_HELP_CACHE, _LAST_CODE_HASH, Linux page cache)
# persists and subsequent requests skip repeated setup work.
#
# If you need to force a restart after deploying a new handler.py,
# use the RunPod console to restart the endpoint (or set workersMin=0
# briefly to drain all workers).
runpod.serverless.start({"handler": handler, "refresh_worker": False})
