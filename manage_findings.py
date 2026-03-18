from __future__ import annotations

"""CLI for saving and managing cross-worktree research findings."""

import argparse
from pathlib import Path
import sys

from fedprotrack.findings import (
    create_finding,
    default_ledger_root,
    get_finding,
    list_findings,
    load_finding_body,
    promote_finding,
    resolve_git_context,
)


def _read_body(args: argparse.Namespace) -> str:
    if args.body is not None:
        return args.body
    if args.body_file is not None:
        return Path(args.body_file).read_text(encoding="utf-8")
    raise ValueError("one of --body or --body-file is required")


def cmd_save(args: argparse.Namespace) -> None:
    context = resolve_git_context()
    body = _read_body(args)
    record = create_finding(
        context=context,
        title=args.title,
        body=body,
        kind=args.kind,
        status=args.status,
        dataset=args.dataset,
        source=args.source,
        tags=args.tag,
        artifacts=args.artifact,
        commands=args.command,
        related_ids=args.related,
    )
    print(f"saved: {record.finding_id}")
    print(f"ledger: {record.ledger_path}")


def cmd_list(args: argparse.Namespace) -> None:
    context = resolve_git_context()
    root = default_ledger_root(context)
    rows = list_findings(
        root,
        kind=args.kind,
        status=args.status,
        dataset=args.dataset,
        tag=args.tag,
        text=args.text,
    )
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        print("no findings")
        return

    for row in rows:
        dataset = row.dataset or "-"
        promoted = " promoted" if row.promoted_doc else ""
        print(
            f"{row.finding_id} | {row.status} | {row.kind} | "
            f"{dataset} | {row.title}{promoted}"
        )


def cmd_show(args: argparse.Namespace) -> None:
    context = resolve_git_context()
    root = default_ledger_root(context)
    record = get_finding(root, args.finding_id)
    print(f"id: {record.finding_id}")
    print(f"title: {record.title}")
    print(f"status: {record.status}")
    print(f"kind: {record.kind}")
    print(f"dataset: {record.dataset or '-'}")
    print(f"branch: {record.branch}")
    print(f"commit: {record.commit}")
    print(f"ledger: {record.ledger_path}")
    if record.promoted_doc:
        print(f"promoted_doc: {record.promoted_doc}")
    print()
    print(load_finding_body(record))


def cmd_promote(args: argparse.Namespace) -> None:
    context = resolve_git_context()
    root = default_ledger_root(context)
    target = promote_finding(
        root,
        args.finding_id,
        docs_root=context.repo_root / "docs",
        overwrite=args.overwrite,
    )
    print(f"promoted: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage_findings",
        description="Save and retrieve cross-worktree experiment findings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    save = sub.add_parser("save", help="Save a new finding into the shared ledger")
    save.add_argument("--title", required=True)
    save.add_argument("--body", default=None)
    save.add_argument("--body-file", default=None)
    save.add_argument("--kind", default="finding")
    save.add_argument("--status", default="draft", choices=["draft", "validated", "final"])
    save.add_argument("--dataset", default=None)
    save.add_argument("--source", default=None)
    save.add_argument("--tag", action="append", default=[])
    save.add_argument("--artifact", action="append", default=[])
    save.add_argument("--command", action="append", default=[])
    save.add_argument("--related", action="append", default=[])
    save.set_defaults(func=cmd_save)

    ls = sub.add_parser("list", help="List saved findings")
    ls.add_argument("--kind", default=None)
    ls.add_argument("--status", default=None)
    ls.add_argument("--dataset", default=None)
    ls.add_argument("--tag", default=None)
    ls.add_argument("--text", default=None)
    ls.add_argument("--limit", type=int, default=None)
    ls.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="Show one finding")
    show.add_argument("finding_id")
    show.set_defaults(func=cmd_show)

    promote = sub.add_parser("promote", help="Copy a finding into docs/findings")
    promote.add_argument("finding_id")
    promote.add_argument("--overwrite", action="store_true")
    promote.set_defaults(func=cmd_promote)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover - CLI safety net
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
