from __future__ import annotations

"""Shared findings ledger for cross-worktree experiment summaries.

The ledger lives under the repository's git common directory so every
worktree can read and append the same summary records. Each finding is
stored twice:

1. as a markdown entry with YAML front matter for human reading, and
2. as a JSONL metadata row for machine filtering and indexing.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml


_STATUS_VALUES = {"draft", "validated", "final"}


@dataclass(frozen=True)
class GitContext:
    """Repository metadata needed for traceable finding storage."""

    repo_root: Path
    git_common_dir: Path
    worktree_root: Path
    branch: str
    commit: str


@dataclass
class FindingRecord:
    """Metadata for one summary finding."""

    finding_id: str
    title: str
    slug: str
    created_at: str
    kind: str
    status: str
    repo_root: str
    git_common_dir: str
    worktree_root: str
    branch: str
    commit: str
    ledger_path: str
    dataset: str | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    promoted_doc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the record into a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FindingRecord:
        """Construct a record from persisted metadata."""
        return cls(**payload)


def resolve_git_context(cwd: Path | str | None = None) -> GitContext:
    """Resolve the current repository and shared git metadata paths."""
    base = Path(cwd or ".").resolve()
    repo_root = Path(_git(["rev-parse", "--show-toplevel"], cwd=base))
    git_common_dir = Path(_git(["rev-parse", "--git-common-dir"], cwd=base))
    branch = _git(["branch", "--show-current"], cwd=base)
    if not branch:
        branch = "DETACHED"
    commit = _git(["rev-parse", "HEAD"], cwd=base)
    return GitContext(
        repo_root=repo_root,
        git_common_dir=git_common_dir,
        worktree_root=repo_root,
        branch=branch,
        commit=commit,
    )


def default_ledger_root(context: GitContext) -> Path:
    """Return the shared ledger root used by all worktrees."""
    return context.git_common_dir / "fedprotrack" / "findings"


def slugify(value: str) -> str:
    """Convert a title into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "finding"


def create_finding(
    *,
    context: GitContext,
    title: str,
    body: str,
    kind: str = "finding",
    status: str = "draft",
    dataset: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    artifacts: list[str] | None = None,
    commands: list[str] | None = None,
    related_ids: list[str] | None = None,
    ledger_root: Path | None = None,
    created_at: datetime | None = None,
) -> FindingRecord:
    """Persist a human-readable finding plus its ledger metadata."""
    if status not in _STATUS_VALUES:
        raise ValueError(
            f"status must be one of {sorted(_STATUS_VALUES)}, got {status!r}"
        )

    when = created_at or datetime.now().astimezone()
    stamp = when.strftime("%Y%m%d-%H%M%S")
    slug = slugify(title)
    finding_id = f"{stamp}-{slug}"

    root = Path(ledger_root) if ledger_root is not None else default_ledger_root(context)
    entries_dir = root / "entries"
    entries_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = entries_dir / f"{finding_id}.md"

    record = FindingRecord(
        finding_id=finding_id,
        title=title.strip(),
        slug=slug,
        created_at=when.isoformat(timespec="seconds"),
        kind=kind.strip(),
        status=status,
        repo_root=str(context.repo_root),
        git_common_dir=str(context.git_common_dir),
        worktree_root=str(context.worktree_root),
        branch=context.branch,
        commit=context.commit,
        ledger_path=str(ledger_path),
        dataset=dataset.strip() if dataset else None,
        source=source.strip() if source else None,
        tags=[t.strip() for t in tags or [] if t.strip()],
        artifacts=[a.strip() for a in artifacts or [] if a.strip()],
        commands=[c.strip() for c in commands or [] if c.strip()],
        related_ids=[r.strip() for r in related_ids or [] if r.strip()],
    )

    _write_markdown_entry(ledger_path, record, body)
    _append_index_row(root / "index.jsonl", record)
    return record


def list_findings(
    ledger_root: Path,
    *,
    kind: str | None = None,
    status: str | None = None,
    dataset: str | None = None,
    tag: str | None = None,
    text: str | None = None,
) -> list[FindingRecord]:
    """Load and filter finding records from the shared ledger."""
    index_path = Path(ledger_root) / "index.jsonl"
    if not index_path.exists():
        return []

    records: list[FindingRecord] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = FindingRecord.from_dict(json.loads(line))
        if kind is not None and record.kind != kind:
            continue
        if status is not None and record.status != status:
            continue
        if dataset is not None and record.dataset != dataset:
            continue
        if tag is not None and tag not in record.tags:
            continue
        if text is not None:
            haystack = " ".join(
                [
                    record.finding_id,
                    record.title,
                    record.dataset or "",
                    " ".join(record.tags),
                ]
            ).lower()
            if text.lower() not in haystack:
                continue
        records.append(record)

    records.sort(key=lambda item: item.created_at, reverse=True)
    return records


def get_finding(ledger_root: Path, finding_id: str) -> FindingRecord:
    """Return one finding record by id."""
    matches = [item for item in list_findings(ledger_root) if item.finding_id == finding_id]
    if not matches:
        raise FileNotFoundError(f"Unknown finding id: {finding_id}")
    return matches[0]


def promote_finding(
    ledger_root: Path,
    finding_id: str,
    *,
    docs_root: Path,
    overwrite: bool = False,
) -> Path:
    """Copy a shared finding into versioned project docs."""
    record = get_finding(ledger_root, finding_id)
    source = Path(record.ledger_path)
    if not source.exists():
        raise FileNotFoundError(f"Finding file missing: {source}")

    target_dir = Path(docs_root) / "findings"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists() and not overwrite:
        raise FileExistsError(f"Promoted finding already exists: {target}")

    shutil.copy2(source, target)
    _rewrite_promoted_doc_path(Path(ledger_root) / "index.jsonl", finding_id, target)
    return target


def load_finding_body(record: FindingRecord) -> str:
    """Load the markdown body of a stored finding."""
    text = Path(record.ledger_path).read_text(encoding="utf-8")
    parts = text.split("---\n", 2)
    if len(parts) < 3:
        return text
    return parts[2].lstrip("\n")


def _write_markdown_entry(path: Path, record: FindingRecord, body: str) -> None:
    metadata = record.to_dict()
    front_matter = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    payload = f"---\n{front_matter}\n---\n\n{body.rstrip()}\n"
    path.write_text(payload, encoding="utf-8")


def _append_index_row(index_path: Path, record: FindingRecord) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False))
        fh.write("\n")


def _rewrite_promoted_doc_path(index_path: Path, finding_id: str, target: Path) -> None:
    rows = list_findings(index_path.parent)
    updated = False
    for row in rows:
        if row.finding_id == finding_id:
            row.promoted_doc = str(target)
            updated = True
            break

    if not updated:
        raise FileNotFoundError(f"Unknown finding id: {finding_id}")

    with index_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False))
            fh.write("\n")


def _git(args: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
