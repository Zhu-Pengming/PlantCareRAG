#!/usr/bin/env python3
"""Interactively review v1.2 evidence without weakening human verification."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PENDING = "agent_source_checked_pending_human_review"
VERIFIED = "human_verified"
REJECTED = "rejected"
UNREVIEWED = "unreviewed"
ALLOWED_STATUSES = {PENDING, VERIFIED, REJECTED, UNREVIEWED}
RISKY_SCOPES = {"genus", "common_name_mapping"}

REPO_DEFAULT = Path(__file__).resolve().parents[2]
DEFAULT_OVERLAY = "evaluation_v1_2/data/evidence_overlay.json"
DEFAULT_CHECKLIST = "evaluation_v1_2/REVIEW_CHECKLIST.md"
DEFAULT_LOG = "evaluation_v1_2/data/review_audit.jsonl"


def load_overlay(path: Path) -> tuple[Any, list[dict]]:
    """Return the root document and its claims list."""
    root = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(root, list):
        return root, root
    if isinstance(root, dict):
        for key in ("claims", "evidence", "records", "items"):
            if isinstance(root.get(key), list):
                return root, root[key]
    raise ValueError(
        f"Could not locate claims in {path}; expected a list or a known claims key."
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Write in the destination directory, then atomically replace the file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_overlay(path: Path, root: Any) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    atomic_write_text(
        path,
        json.dumps(root, ensure_ascii=False, indent=2) + "\n",
    )


def append_audit(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def update_checklist(path: Path, claims: list[dict]) -> int:
    """Tick a source row only when every claim for that URL is settled."""
    if not path.exists():
        return 0
    settled_by_url: dict[str, bool] = {}
    for claim in claims:
        url = (claim.get("locator") or {}).get("url", "")
        if url:
            settled = claim.get("review_status") in {VERIFIED, REJECTED}
            settled_by_url[url] = settled_by_url.get(url, True) and settled

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = 0
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*-\s*\[)([ xX])(\].*)$", line.rstrip("\n"))
        if not match:
            continue
        urls = [url for url in settled_by_url if url in line]
        if not urls:
            continue
        wanted = "x" if all(settled_by_url[url] for url in urls) else " "
        if match.group(2).lower() != wanted:
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = match.group(1) + wanted + match.group(3) + newline
            changed += 1
    if changed:
        atomic_write_text(path, "".join(lines))
    return changed


def show_claim(claim: dict, index: int, total: int) -> None:
    locator = claim.get("locator") or {}
    scope = claim.get("entity_scope", "?")
    warning = "  ⚠ SCOPE-SENSITIVE" if scope in RISKY_SCOPES else ""
    print("\n" + "=" * 92)
    print(f"[{index}/{total}]  {claim.get('claim_id', '?')}")
    print("-" * 92)
    print(f"  plant        : {claim.get('plant_name')} / {claim.get('scientific_name')}")
    print(f"  entity_scope : {scope}{warning}")
    print(f"  dimension    : {claim.get('dimension')}")
    qualifiers = claim.get("qualifiers") or {}
    if qualifiers:
        print(f"  qualifiers   : {json.dumps(qualifiers, ensure_ascii=False)}")
    else:
        print("  qualifiers   : (none) ← 检查原文是否有 indoor/season 限定")
    print("\n  claim_text:")
    print(textwrap.fill(str(claim.get("claim_text", "")), width=88, initial_indent="    ", subsequent_indent="    "))
    print(f"\n  source : {claim.get('source_title')} [{claim.get('source_type')}]")
    print(f"  url    : {locator.get('url')}")
    print(
        f"  section: {locator.get('section') or '(none)'}; "
        f"paragraph: {locator.get('paragraph') or '(none)'}"
    )
    if claim.get("review_notes"):
        print(f"\n  review_notes: {claim['review_notes']}")
    print("=" * 92)


def prompt(message: str) -> str:
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def status_counts(claims: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(claim.get("review_status", "<missing>") for claim in claims).items()))


def print_summary(claims: list[dict]) -> None:
    print("\nreview_status tally:")
    for status, count in status_counts(claims).items():
        print(f"  {status}: {count}")


def check(claims: list[dict]) -> list[str]:
    """Validate mixed review states without treating pending as verified."""
    errors: list[str] = []
    for claim in claims:
        claim_id = claim.get("claim_id", "<missing-id>")
        status = claim.get("review_status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{claim_id}: unknown review_status {status!r}")
        elif status == UNREVIEWED:
            errors.append(f"{claim_id}: unreviewed evidence cannot enter the overlay")
        elif status == REJECTED and not str(claim.get("review_notes") or "").strip():
            errors.append(f"{claim_id}: rejected evidence requires review_notes")
        elif status == VERIFIED:
            locator = claim.get("locator") or {}
            if not str(locator.get("url") or "").startswith("https://"):
                errors.append(f"{claim_id}: human_verified evidence requires an HTTPS locator")
    return errors


def record_decision(
    claim: dict,
    decision: str,
    note: str | None,
    reviewer: str,
    root: Any,
    overlay_path: Path,
    audit_path: Path,
) -> None:
    claim["review_status"] = decision
    if decision == REJECTED:
        claim["review_notes"] = note
    save_overlay(overlay_path, root)
    append_audit(
        audit_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reviewer": reviewer,
            "claim_id": claim.get("claim_id"),
            "source_id": claim.get("source_id"),
            "entity_scope": claim.get("entity_scope"),
            "decision": decision,
            "note": note,
        },
    )


def review(
    claims: list[dict],
    root: Any,
    overlay_path: Path,
    checklist_path: Path,
    audit_path: Path,
    reviewer: str,
    source_filter: str | None,
) -> None:
    queue = [
        claim
        for claim in claims
        if claim.get("review_status") in {PENDING, UNREVIEWED}
    ]
    if source_filter:
        folded_filter = source_filter.casefold()
        queue = [
            claim
            for claim in queue
            if folded_filter
            in (
                str(claim.get("source_id", ""))
                + " "
                + str((claim.get("locator") or {}).get("url", ""))
            ).casefold()
        ]
    queue.sort(
        key=lambda claim: (
            claim.get("source_id", ""),
            claim.get("dimension", ""),
            claim.get("claim_id", ""),
        )
    )
    if not queue:
        print("没有匹配的待复核 claim。")
        print_summary(claims)
        return

    counts = {"approved": 0, "rejected": 0, "skipped": 0}
    checklist_updates = 0
    print(f"\n待复核 {len(queue)} 条；同一 source 的 claims 连续显示。")
    print("按键: [a]pprove [r]eject [s]kip [n]ote [u]rl [q]uit")
    print("核对 scientific name、claim 范围、限定条件和 locator。")
    for index, claim in enumerate(queue, 1):
        show_claim(claim, index, len(queue))
        while True:
            choice = prompt("  decision [a/r/s/n/u/q] > ").casefold()
            if choice in {"u", "url"}:
                print(f"  {(claim.get('locator') or {}).get('url')}")
                continue
            if choice in {"n", "note"}:
                note = prompt("  note > ")
                if note:
                    claim["review_notes"] = note
                    save_overlay(overlay_path, root)
                    print("  note saved")
                continue
            if choice in {"s", "skip", ""}:
                counts["skipped"] += 1
                break
            if choice in {"q", "quit"}:
                checklist_updates += update_checklist(checklist_path, claims)
                print_summary(claims)
                return
            if choice in {"a", "approve"}:
                scope = claim.get("entity_scope")
                if scope in RISKY_SCOPES:
                    print(f"  ⚠ entity_scope={scope}")
                    if scope == "genus":
                        print("  属级 claim 只能表达一般规律，不能保证所有物种/品种适用。")
                    else:
                        print("  请确认 dataset 俗名能够安全映射到当前 scientific_name。")
                    if prompt("  确认该范围也成立？输入 yes > ").casefold() != "yes":
                        print("  未批准。")
                        continue
                record_decision(
                    claim, VERIFIED, claim.get("review_notes"), reviewer,
                    root, overlay_path, audit_path,
                )
                checklist_updates += update_checklist(checklist_path, claims)
                counts["approved"] += 1
                break
            if choice in {"r", "reject"}:
                note = prompt("  拒绝原因（必填）> ")
                if not note:
                    print("  rejected 必须填写原因。")
                    continue
                record_decision(
                    claim, REJECTED, note, reviewer,
                    root, overlay_path, audit_path,
                )
                checklist_updates += update_checklist(checklist_path, claims)
                counts["rejected"] += 1
                break
            print("  未知输入。")

    checklist_updates += update_checklist(checklist_path, claims)
    print(
        f"\napproved={counts['approved']} rejected={counts['rejected']} "
        f"skipped={counts['skipped']} checklist_updated={checklist_updates}"
    )
    print_summary(claims)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--checklist", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--source", help="Match a source_id or URL substring")
    parser.add_argument("--reviewer")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    overlay_path = args.overlay or args.repo / DEFAULT_OVERLAY
    checklist_path = args.checklist or args.repo / DEFAULT_CHECKLIST
    audit_path = args.log or args.repo / DEFAULT_LOG
    if not overlay_path.exists():
        print(f"找不到 overlay: {overlay_path}", file=sys.stderr)
        return 2
    try:
        root, claims = load_overlay(overlay_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.check:
        errors = check(claims)
        print_summary(claims)
        if errors:
            print("\nFAILED:")
            for error in errors:
                print(f"  - {error}")
            return 1
        print("\nOK — mixed review states are valid.")
        return 0

    reviewer = args.reviewer or os.environ.get("REVIEWER") or getpass.getuser()
    review(
        claims, root, overlay_path, checklist_path, audit_path,
        reviewer, args.source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
