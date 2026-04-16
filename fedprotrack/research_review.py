from __future__ import annotations

"""Reviewer-facing audit for the local research pipeline.

The goal is not to score the paper. The goal is to catch credibility breaks
before a hostile reviewer does: missing evidence, stale artifact paths, and
validated claims without a reproducible command trail.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any

import yaml

from fedprotrack.findings import (
    default_ledger_root,
    list_findings,
    load_finding_body,
    resolve_git_context,
)


_TERMINAL_HYPOTHESIS_STATUSES = {
    "accepted",
    "completed",
    "done",
    "final",
    "partially_validated",
    "refuted",
    "validated",
}
_VALIDATED_FINDING_STATUSES = {"validated", "final"}


@dataclass(frozen=True)
class ReviewIssue:
    """One concrete reviewer-facing problem in the research workspace."""

    severity: str
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert the issue to a JSON-serializable dictionary."""
        return asdict(self)


def normalize_status(value: object) -> str:
    """Normalize free-form status strings for gate checks."""
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def hypothesis_has_evidence(payload: dict[str, Any]) -> bool:
    """Return whether a hypothesis includes concrete supporting evidence."""
    for key in ("result_summary", "finding_ids", "artifacts", "outputs", "evidence"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
        if isinstance(value, dict) and value:
            return True
    return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from *path*."""
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(payload)
    return rows


def resolve_artifact_path(base_dir: Path, artifact: str) -> Path:
    """Resolve a relative artifact path against the recorded worktree root."""
    candidate = Path(artifact)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def review_workspace(
    repo_root: Path,
    *,
    docs_root: Path | None = None,
    ledger_root: Path | None = None,
) -> list[ReviewIssue]:
    """Audit the local research workspace for reproducibility gaps."""
    repo_root = repo_root.resolve()
    docs_root = (docs_root or (repo_root / "docs")).resolve()
    issues: list[ReviewIssue] = []

    required_files = {
        "research-state.yaml": repo_root / "research-state.yaml",
        "research-log.md": repo_root / "research-log.md",
        "findings.md": repo_root / "findings.md",
    }
    for label, path in required_files.items():
        if not path.exists():
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="workspace.missing_file",
                    message=f"Required research workspace file is missing: {label}",
                    path=str(path),
                )
            )

    research_state_path = required_files["research-state.yaml"]
    if research_state_path.exists():
        issues.extend(review_research_state(research_state_path))

    hypothesis_index = docs_root / "hypotheses.jsonl"
    if hypothesis_index.exists():
        issues.extend(review_hypothesis_index(hypothesis_index))
    else:
        issues.append(
            ReviewIssue(
                severity="warning",
                code="hypotheses.missing_index",
                message="docs/hypotheses.jsonl is missing; claim-level evidence coverage cannot be checked.",
                path=str(hypothesis_index),
            )
        )

    effective_ledger_root = ledger_root
    if effective_ledger_root is None:
        try:
            effective_ledger_root = default_ledger_root(resolve_git_context(repo_root))
        except Exception:
            effective_ledger_root = None
            issues.append(
                ReviewIssue(
                    severity="warning",
                    code="findings.ledger_unresolved",
                    message="Shared findings ledger could not be resolved from git metadata.",
                    path=str(repo_root),
                )
            )

    if effective_ledger_root is not None:
        issues.extend(review_findings_ledger(Path(effective_ledger_root).resolve(), repo_root))

    return issues


def review_research_state(path: Path) -> list[ReviewIssue]:
    """Check the autoresearch state file for completeness."""
    issues: list[ReviewIssue] = []
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [
            ReviewIssue(
                severity="error",
                code="workspace.invalid_yaml",
                message=f"research-state.yaml failed to parse: {exc}",
                path=str(path),
            )
        ]

    if not isinstance(payload, dict):
        return [
            ReviewIssue(
                severity="error",
                code="workspace.invalid_yaml",
                message="research-state.yaml must contain a top-level mapping.",
                path=str(path),
            )
        ]

    project = payload.get("project")
    if not isinstance(project, dict) or not str(project.get("question", "")).strip():
        issues.append(
            ReviewIssue(
                severity="error",
                code="workspace.missing_project_question",
                message="research-state.yaml must record the active research question.",
                path=str(path),
            )
        )

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict) or not str(evaluation.get("proxy_metric", "")).strip():
        issues.append(
            ReviewIssue(
                severity="error",
                code="workspace.missing_proxy_metric",
                message="research-state.yaml must define a proxy metric for the current loop.",
                path=str(path),
            )
        )

    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        issues.append(
            ReviewIssue(
                severity="error",
                code="workspace.missing_hypotheses",
                message="research-state.yaml must contain a non-empty hypotheses list.",
                path=str(path),
            )
        )
        return issues

    for idx, hypothesis in enumerate(hypotheses, start=1):
        if not isinstance(hypothesis, dict):
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="workspace.invalid_hypothesis",
                    message=f"Hypothesis #{idx} is not a mapping.",
                    path=str(path),
                )
            )
            continue
        ref = hypothesis.get("id") or f"hypothesis[{idx}]"
        if not str(hypothesis.get("statement", "")).strip():
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="workspace.hypothesis_missing_statement",
                    message=f"{ref} is missing a statement.",
                    path=str(path),
                )
            )
        if not str(hypothesis.get("experiment_template", "")).strip():
            issues.append(
                ReviewIssue(
                    severity="warning",
                    code="workspace.hypothesis_missing_template",
                    message=f"{ref} has no experiment_template; reviewers cannot trace how to rerun it.",
                    path=str(path),
                )
            )
        status = normalize_status(hypothesis.get("status"))
        if status in _TERMINAL_HYPOTHESIS_STATUSES and not hypothesis_has_evidence(hypothesis):
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="workspace.hypothesis_missing_evidence",
                    message=f"{ref} is marked {status} but has no result_summary / finding_ids / artifacts.",
                    path=str(path),
                )
            )
    return issues


def review_hypothesis_index(path: Path) -> list[ReviewIssue]:
    """Check the long-term hypothesis register for evidence completeness."""
    issues: list[ReviewIssue] = []
    try:
        rows = load_jsonl(path)
    except ValueError as exc:
        return [
            ReviewIssue(
                severity="error",
                code="hypotheses.invalid_jsonl",
                message=str(exc),
                path=str(path),
            )
        ]

    seen_ids: set[str] = set()
    for row in rows:
        hypothesis_id = str(row.get("id", "")).strip()
        if not hypothesis_id:
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="hypotheses.missing_id",
                    message="A docs/hypotheses.jsonl record is missing its id.",
                    path=str(path),
                )
            )
            continue
        if hypothesis_id in seen_ids:
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="hypotheses.duplicate_id",
                    message=f"Duplicate hypothesis id in docs/hypotheses.jsonl: {hypothesis_id}",
                    path=str(path),
                )
            )
        seen_ids.add(hypothesis_id)

        if not str(row.get("claim", "")).strip():
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="hypotheses.missing_claim",
                    message=f"{hypothesis_id} is missing its claim text.",
                    path=str(path),
                )
            )
        if not str(row.get("experiment", "")).strip():
            issues.append(
                ReviewIssue(
                    severity="warning",
                    code="hypotheses.missing_experiment",
                    message=f"{hypothesis_id} has no experiment field.",
                    path=str(path),
                )
            )
        if not str(row.get("pass_criterion", "")).strip():
            issues.append(
                ReviewIssue(
                    severity="warning",
                    code="hypotheses.missing_pass_criterion",
                    message=f"{hypothesis_id} has no pass_criterion.",
                    path=str(path),
                )
            )

        status = normalize_status(row.get("status"))
        if status in _TERMINAL_HYPOTHESIS_STATUSES and not hypothesis_has_evidence(row):
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="hypotheses.missing_evidence",
                    message=f"{hypothesis_id} is marked {status} but has no result_summary / finding_ids / artifacts.",
                    path=str(path),
                )
            )
    return issues


def review_findings_ledger(ledger_root: Path, repo_root: Path) -> list[ReviewIssue]:
    """Audit saved findings for reviewer-facing reproducibility gaps."""
    issues: list[ReviewIssue] = []
    if not ledger_root.exists():
        issues.append(
            ReviewIssue(
                severity="warning",
                code="findings.missing_ledger",
                message="Shared findings ledger is missing; stable conclusions are not centrally indexed.",
                path=str(ledger_root),
            )
        )
        return issues

    records = list_findings(ledger_root)
    known_ids = {record.finding_id for record in records}
    for record in records:
        entry_path = Path(record.ledger_path)
        if not entry_path.exists():
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="finding.missing_entry",
                    message=f"Ledger index points to a missing finding entry: {record.finding_id}",
                    path=str(entry_path),
                )
            )
            continue

        body = load_finding_body(record).strip()
        if not body:
            issues.append(
                ReviewIssue(
                    severity="error",
                    code="finding.empty_body",
                    message=f"{record.finding_id} has an empty markdown body.",
                    path=str(entry_path),
                )
            )

        if record.status in _VALIDATED_FINDING_STATUSES:
            if not record.commands:
                issues.append(
                    ReviewIssue(
                        severity="error",
                        code="finding.missing_command",
                        message=f"{record.finding_id} is {record.status} but records no rerun command.",
                        path=str(entry_path),
                    )
                )
            if not record.artifacts:
                issues.append(
                    ReviewIssue(
                        severity="error",
                        code="finding.missing_artifact",
                        message=f"{record.finding_id} is {record.status} but records no supporting artifact.",
                        path=str(entry_path),
                    )
                )

        worktree_root = Path(record.worktree_root or repo_root)
        for artifact in record.artifacts:
            artifact_path = resolve_artifact_path(worktree_root, artifact)
            if not artifact_path.exists():
                issues.append(
                    ReviewIssue(
                        severity="error",
                        code="finding.missing_artifact",
                        message=f"{record.finding_id} references a missing artifact: {artifact}",
                        path=str(artifact_path),
                    )
                )

        for related_id in record.related_ids:
            if related_id not in known_ids:
                issues.append(
                    ReviewIssue(
                        severity="error",
                        code="finding.unknown_related_id",
                        message=f"{record.finding_id} references unknown related_id {related_id}.",
                        path=str(entry_path),
                    )
                )

        if record.promoted_doc:
            promoted_path = Path(record.promoted_doc)
            if not promoted_path.exists():
                issues.append(
                    ReviewIssue(
                        severity="error",
                        code="finding.missing_promoted_doc",
                        message=f"{record.finding_id} points to a missing promoted doc.",
                        path=str(promoted_path),
                    )
                )
    return issues


def should_fail_review(issues: list[ReviewIssue], strict: bool) -> bool:
    """Return whether the audit should exit non-zero."""
    if any(issue.severity == "error" for issue in issues):
        return True
    if strict and any(issue.severity == "warning" for issue in issues):
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Audit the local research pipeline for adversarial-review gaps.",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--docs-root", default=None)
    parser.add_argument("--ledger-root", default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failing gates.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for the research review gate."""
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    docs_root = Path(args.docs_root) if args.docs_root else None
    ledger_root = Path(args.ledger_root) if args.ledger_root else None
    issues = review_workspace(
        repo_root,
        docs_root=docs_root,
        ledger_root=ledger_root,
    )

    if args.json:
        payload = {
            "repo_root": str(repo_root.resolve()),
            "strict": bool(args.strict),
            "issue_count": len(issues),
            "errors": sum(issue.severity == "error" for issue in issues),
            "warnings": sum(issue.severity == "warning" for issue in issues),
            "issues": [issue.to_dict() for issue in issues],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("Research Review")
        print(f"repo_root: {repo_root.resolve()}")
        if not issues:
            print("status: PASS")
        else:
            for issue in issues:
                location = f" [{issue.path}]" if issue.path else ""
                print(f"{issue.severity.upper():7s} {issue.code}: {issue.message}{location}")
            print(
                f"summary: {sum(issue.severity == 'error' for issue in issues)} error(s), "
                f"{sum(issue.severity == 'warning' for issue in issues)} warning(s)"
            )

    raise SystemExit(1 if should_fail_review(issues, strict=bool(args.strict)) else 0)


if __name__ == "__main__":
    main()
