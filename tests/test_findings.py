from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fedprotrack.findings import (
    GitContext,
    create_finding,
    get_finding,
    list_findings,
    load_finding_body,
    promote_finding,
)


def _context(tmp_path: Path) -> GitContext:
    repo_root = tmp_path / "repo"
    git_common_dir = tmp_path / "common-git"
    repo_root.mkdir()
    git_common_dir.mkdir()
    return GitContext(
        repo_root=repo_root,
        git_common_dir=git_common_dir,
        worktree_root=repo_root,
        branch="codex/test-findings",
        commit="abc123",
    )


def test_create_finding_writes_markdown_and_index(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ledger_root = tmp_path / "ledger"

    record = create_finding(
        context=ctx,
        title="CIFAR-100 fingerprint collapse",
        body="Root cause summary.",
        kind="root-cause",
        status="validated",
        dataset="cifar100",
        tags=["cifar100", "fingerprint"],
        artifacts=["results/run1/summary.json"],
        commands=["python run_cifar100_comparison.py"],
        ledger_root=ledger_root,
        created_at=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc),
    )

    assert Path(record.ledger_path).exists()
    assert (ledger_root / "index.jsonl").exists()
    assert record.finding_id.startswith("20260318-120000-")

    loaded = get_finding(ledger_root, record.finding_id)
    assert loaded.title == "CIFAR-100 fingerprint collapse"
    assert loaded.dataset == "cifar100"
    assert load_finding_body(loaded) == "Root cause summary.\n"


def test_list_findings_filters_by_dataset_and_tag(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ledger_root = tmp_path / "ledger"

    create_finding(
        context=ctx,
        title="CIFAR summary",
        body="A",
        dataset="cifar100",
        tags=["cifar100"],
        ledger_root=ledger_root,
        created_at=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc),
    )
    create_finding(
        context=ctx,
        title="MNIST summary",
        body="B",
        dataset="rotating-mnist",
        tags=["mnist"],
        ledger_root=ledger_root,
        created_at=datetime(2026, 3, 18, 12, 1, tzinfo=timezone.utc),
    )

    cifar_rows = list_findings(ledger_root, dataset="cifar100")
    assert len(cifar_rows) == 1
    assert cifar_rows[0].dataset == "cifar100"

    mnist_rows = list_findings(ledger_root, tag="mnist")
    assert len(mnist_rows) == 1
    assert mnist_rows[0].title == "MNIST summary"


def test_promote_finding_copies_into_docs(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    ledger_root = tmp_path / "ledger"

    record = create_finding(
        context=ctx,
        title="Promote me",
        body="Stable conclusion.",
        status="final",
        ledger_root=ledger_root,
        created_at=datetime(2026, 3, 18, 12, 0, tzinfo=timezone.utc),
    )

    target = promote_finding(
        ledger_root,
        record.finding_id,
        docs_root=ctx.repo_root / "docs",
    )
    assert target.exists()
    assert "Stable conclusion." in target.read_text(encoding="utf-8")

    updated = get_finding(ledger_root, record.finding_id)
    assert updated.promoted_doc == str(target)
