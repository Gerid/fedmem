from __future__ import annotations

import json
from pathlib import Path

from fedprotrack.findings import GitContext, create_finding
from fedprotrack.research_review import review_workspace, should_fail_review


def _context(tmp_path: Path) -> GitContext:
    repo_root = tmp_path / "repo"
    git_common_dir = tmp_path / "common-git"
    repo_root.mkdir()
    git_common_dir.mkdir()
    return GitContext(
        repo_root=repo_root,
        git_common_dir=git_common_dir,
        worktree_root=repo_root,
        branch="codex/test-research-review",
        commit="abc123",
    )


def _write_minimal_workspace(repo_root: Path) -> None:
    (repo_root / "research-state.yaml").write_text(
        "\n".join(
            [
                "project:",
                '  question: "Does the audit stay reproducible?"',
                "evaluation:",
                '  proxy_metric: "alignment_rate"',
                "hypotheses:",
                "  - id: H1",
                '    statement: "Keep a command trail."',
                '    status: "READY"',
                '    prediction: "Audit stays green."',
                '    experiment_template: "python run_example.py"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo_root / "research-log.md").write_text("# Log\n", encoding="utf-8")
    (repo_root / "findings.md").write_text("# Findings\n", encoding="utf-8")
    docs_dir = repo_root / "docs"
    docs_dir.mkdir()
    (docs_dir / "hypotheses.jsonl").write_text(
        json.dumps(
            {
                "id": "H1",
                "claim": "Keep reviewer-facing evidence wired.",
                "experiment": "python run_example.py",
                "pass_criterion": "Audit exits zero.",
                "status": "open",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_review_workspace_passes_with_complete_evidence(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _write_minimal_workspace(ctx.repo_root)
    ledger_root = tmp_path / "ledger"
    artifact = ctx.repo_root / "results" / "run1" / "summary.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    create_finding(
        context=ctx,
        title="Stable CIFAR conclusion",
        body="Final numbers are frozen.",
        status="validated",
        artifacts=[str(artifact.relative_to(ctx.repo_root))],
        commands=["python run_example.py --seed 42"],
        ledger_root=ledger_root,
    )

    issues = review_workspace(ctx.repo_root, ledger_root=ledger_root)
    assert issues == []
    assert should_fail_review(issues, strict=True) is False


def test_review_workspace_flags_validated_hypothesis_without_evidence(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _write_minimal_workspace(ctx.repo_root)
    hypothesis_path = ctx.repo_root / "docs" / "hypotheses.jsonl"
    hypothesis_path.write_text(
        json.dumps(
            {
                "id": "H7",
                "claim": "Validated claims need evidence.",
                "experiment": "python run_missing.py",
                "pass_criterion": "Have a result_summary.",
                "status": "validated",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    issues = review_workspace(ctx.repo_root)
    assert any(issue.code == "hypotheses.missing_evidence" for issue in issues)
    assert should_fail_review(issues, strict=False) is True


def test_review_workspace_flags_missing_artifact_for_validated_finding(tmp_path: Path) -> None:
    ctx = _context(tmp_path)
    _write_minimal_workspace(ctx.repo_root)
    ledger_root = tmp_path / "ledger"

    create_finding(
        context=ctx,
        title="Broken artifact trail",
        body="Numbers exist somewhere else.",
        status="validated",
        artifacts=["results/missing/summary.json"],
        commands=["python run_missing.py --seed 42"],
        ledger_root=ledger_root,
    )

    issues = review_workspace(ctx.repo_root, ledger_root=ledger_root)
    assert any(issue.code == "finding.missing_artifact" for issue in issues)
