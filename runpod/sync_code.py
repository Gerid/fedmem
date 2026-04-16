"""Sync local code to RunPod Network Volume via S3 API.

Usage:
    python runpod/sync_code.py

Environment variables (or set in .env):
    RUNPOD_S3_ACCESS_KEY  — from RunPod Settings > S3 API Keys
    RUNPOD_S3_SECRET_KEY  — from RunPod Settings > S3 API Keys
"""
from __future__ import annotations

import io
import os
import sys
import tarfile
import time
from pathlib import Path

try:
    import boto3
except ImportError:
    print("boto3 not installed. Run: pip install boto3")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────
VOLUME_ID = "7kga5cg276"
REGION = "us-il-1"
ENDPOINT_URL = f"https://s3api-{REGION}.runpod.io"
BUCKET = VOLUME_ID

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".cifar100_cache", ".feature_cache",
    ".mnist_cache", ".cifar10_cache", ".fmow_cache", ".cache",
    ".tmp_pytest", ".pytest_cache", ".claude",
    "tmp", ".tmp", "node_modules", "runpod_results",
    "paper", "outputs", "cowork", "refine-logs",
    "results", "results_neurips_benchmark",
    "results_phase3_v2_smoke", "results_phase3_v2_targeted_sine",
    "results_phase3_v3_sine_final",
}
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".egg-info", ".csv", ".png", ".pdf", ".jpg"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def get_s3_client():
    access_key = os.environ.get("RUNPOD_S3_ACCESS_KEY", "")
    secret_key = os.environ.get("RUNPOD_S3_SECRET_KEY", "")
    if not access_key or not secret_key:
        print("ERROR: Set RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY")
        print("  Get them from: RunPod Console > Settings > S3 API Keys")
        sys.exit(1)
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=REGION,
        endpoint_url=ENDPOINT_URL,
    )


def should_include(path: Path) -> bool:
    parts = path.relative_to(PROJECT_ROOT).parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
        # Exclude results_* dirs
        if part.startswith("results_"):
            return False
    if path.suffix in EXCLUDE_EXTENSIONS:
        return False
    # Skip large files
    try:
        if path.stat().st_size > MAX_FILE_SIZE:
            return False
    except OSError:
        return False
    return True


def create_tar_buffer() -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        count = 0
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if not path.is_file():
                continue
            if not should_include(path):
                continue
            arcname = str(path.relative_to(PROJECT_ROOT))
            tar.add(str(path), arcname=arcname)
            count += 1
    buf.seek(0)
    print(f"  Packed {count} files ({buf.getbuffer().nbytes / 1024 / 1024:.1f} MB)")
    return buf


def sync():
    print(f"=== Syncing code to RunPod Volume ===")
    print(f"  Volume: {VOLUME_ID} ({REGION})")
    print(f"  Source: {PROJECT_ROOT}")

    s3 = get_s3_client()

    # 1. Pack code
    print("Packing code...")
    tar_buf = create_tar_buffer()

    # 2. Upload tar
    remote_key = "code.tar.gz"
    print(f"Uploading to s3://{BUCKET}/{remote_key} ...")
    t0 = time.time()
    s3.upload_fileobj(tar_buf, BUCKET, remote_key)
    dt = time.time() - t0
    print(f"  Uploaded in {dt:.1f}s")

    # 3. Upload marker so handler knows to extract
    s3.put_object(Bucket=BUCKET, Key="code_needs_extract", Body=b"1")

    # 4. Also upload handler.py directly (so it's always up-to-date)
    handler_path = PROJECT_ROOT / "runpod" / "handler.py"
    if handler_path.is_file():
        s3.upload_file(str(handler_path), BUCKET, "handler.py")
        print("  handler.py updated on volume")

    print()
    print("=== Sync complete ===")
    print("Worker 设置了 refresh_worker=True，每次任务结束后自动重启。")
    print("下次提交的任务会自动加载最新代码，无需手动操作。")


if __name__ == "__main__":
    sync()
