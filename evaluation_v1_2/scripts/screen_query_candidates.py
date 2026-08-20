#!/usr/bin/env python3
"""Interactively screen observed queries before manual benchmark annotation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "query_candidates.json"
TRIAGE = ROOT / "data" / "query_auto_triage.json"
SHORTLIST = ROOT / "data" / "query_shortlist.json"
SCREENING = ROOT / "data" / "query_screening.json"
AUDIT = ROOT / "data" / "query_screening_audit.jsonl"
SELECTED = "selected_for_annotation"
REJECTED = "rejected_irrelevant"
ALLOWED = {SELECTED, REJECTED}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_audit(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prompt(message: str) -> str:
    try:
        return input(message).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def plant_ids(candidate: dict) -> set[str]:
    return {match["plant_id"] for match in candidate["alias_matches"]}


def has_title_match(candidate: dict) -> bool:
    return any("title" in match["locations"] for match in candidate["alias_matches"])


def show(candidate: dict, auto: dict | None, index: int, total: int) -> None:
    metrics = candidate["source_metrics"]
    print("\n" + "=" * 96)
    print(f"[{index}/{total}] {candidate['candidate_id']}")
    print("-" * 96)
    print("  plants  : " + ", ".join(sorted(plant_ids(candidate))))
    print(
        "  aliases : "
        + "; ".join(
            f"{match['alias']}@{'+'.join(match['locations'])}"
            for match in candidate["alias_matches"]
        )
    )
    print(
        f"  source  : score={metrics['score']} views={metrics['view_count']} "
        f"answers={metrics['answer_count']} accepted={bool(metrics['accepted_answer_id'])}"
    )
    print(f"  license : {candidate['content_license']}")
    print(f"  author  : {candidate['author']['display_name'] or '(deleted user)'}")
    print(f"  url     : {candidate['source_url']}")
    if auto:
        print(f"  auto    : {auto['predicted_answerability']} score={auto['priority_score']}")
        print(f"  dims    : {', '.join(auto['predicted_dimensions']) or '(none)'}")
        print(f"  risks   : {', '.join(auto['mapping_risks'] + auto['context_flags']) or '(none)'}")
        print(f"  evidence: {', '.join(auto['candidate_evidence_ids']) or '(none; not gold)'}")
    print("\n  title:")
    print(textwrap.fill(candidate["raw_title"], width=92, initial_indent="    ", subsequent_indent="    "))
    print("\n  body:")
    body = candidate["raw_body"] or "(empty; inspect title/page)"
    if len(body) > 1800:
        body = body[:1800].rstrip() + " … [truncated; use u for URL]"
    print(textwrap.fill(body, width=92, initial_indent="    ", subsequent_indent="    "))
    print("=" * 96)


def validate(candidates: list[dict], decisions: list[dict]) -> list[str]:
    errors: list[str] = []
    candidate_ids = {row["candidate_id"] for row in candidates}
    seen: set[str] = set()
    for row in decisions:
        candidate_id = row.get("candidate_id")
        if candidate_id in seen:
            errors.append(f"duplicate decision: {candidate_id}")
        seen.add(candidate_id)
        if candidate_id not in candidate_ids:
            errors.append(f"unknown candidate: {candidate_id}")
        if row.get("screen_status") not in ALLOWED:
            errors.append(f"invalid screen_status: {candidate_id}")
        if row.get("screen_status") == REJECTED and not str(row.get("review_notes", "")).strip():
            errors.append(f"rejection lacks notes: {candidate_id}")
    return errors


def summary(candidates: list[dict], decisions: list[dict], pool_ids: set[str] | None = None) -> None:
    pool_ids = pool_ids or {row["candidate_id"] for row in candidates}
    decisions = [row for row in decisions if row["candidate_id"] in pool_ids]
    counts = Counter(row["screen_status"] for row in decisions)
    selected_ids = {row["candidate_id"] for row in decisions if row["screen_status"] == SELECTED}
    by_plant = Counter()
    for candidate in candidates:
        if candidate["candidate_id"] in selected_ids:
            by_plant.update(plant_ids(candidate))
    print("\nscreening tally:")
    print(f"  pool: {len(pool_ids)}")
    print(f"  pending: {len(pool_ids) - len(decisions)}")
    for status in sorted(ALLOWED):
        print(f"  {status}: {counts[status]}")
    if by_plant:
        print("  selected per plant:")
        for plant_id, count in sorted(by_plant.items()):
            print(f"    {plant_id}: {count}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", help="plant_id or short suffix, e.g. zz_plant")
    parser.add_argument("--reviewer")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--all", action="store_true", help="Review all 399 candidates instead of the automated shortlist")
    args = parser.parse_args()
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    auto_rows = json.loads(TRIAGE.read_text(encoding="utf-8")) if TRIAGE.exists() else []
    auto_by_id = {row["candidate_id"]: row for row in auto_rows}
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8")) if SHORTLIST.exists() else []
    shortlist_ids = {row["candidate_id"] for row in shortlist}
    pool_ids = {row["candidate_id"] for row in candidates} if args.all else shortlist_ids
    if not pool_ids:
        print("找不到自动 shortlist；先运行 auto_triage_queries.py。", file=sys.stderr)
        return 2
    decisions = json.loads(SCREENING.read_text(encoding="utf-8"))
    errors = validate(candidates, decisions)
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if args.check or args.summary:
        summary(candidates, decisions, pool_ids)
        return 0

    decided = {row["candidate_id"] for row in decisions}
    queue = [
        row for row in candidates
        if row["candidate_id"] in pool_ids and row["candidate_id"] not in decided
    ]
    if args.plant:
        wanted = args.plant.removeprefix("plant:")
        queue = [
            row for row in queue
            if any(plant.removeprefix("plant:") == wanted for plant in plant_ids(row))
        ]
    shortlist_order = {
        row["candidate_id"]: (row["quota_plant_id"], row["quota_rank"])
        for row in shortlist
    }
    queue.sort(
        key=lambda row: shortlist_order.get(
            row["candidate_id"],
            ("~", not has_title_match(row), -row["source_metrics"]["score"]),
        )
    )
    if not queue:
        print("没有匹配的待筛选 query。")
        summary(candidates, decisions, pool_ids)
        return 0
    reviewer = args.reviewer or getpass.getuser()
    scope = "全部候选" if args.all else "自动 shortlist"
    print(f"待筛选 {len(queue)} 条（{scope}）。这里只判断是否进入人工 gold 标注池。")
    print("按键: [a]ccept [r]eject [s]kip [u]rl [q]uit")
    for index, candidate in enumerate(queue, 1):
        show(candidate, auto_by_id.get(candidate["candidate_id"]), index, len(queue))
        while True:
            choice = prompt("  decision [a/r/s/u/q] > ").casefold()
            if choice in {"u", "url"}:
                print(f"  {candidate['source_url']}")
                continue
            if choice in {"s", "skip", ""}:
                break
            if choice in {"q", "quit"}:
                summary(candidates, decisions, pool_ids)
                return 0
            if choice in {"a", "accept"}:
                status, notes = SELECTED, prompt("  note（可空）> ")
            elif choice in {"r", "reject"}:
                notes = prompt("  不相关原因（必填）> ")
                if not notes:
                    print("  reject 必须填写原因。")
                    continue
                status = REJECTED
            else:
                print("  未知输入。")
                continue
            record = {
                "candidate_id": candidate["candidate_id"],
                "screen_status": status,
                "reviewer": reviewer,
                "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "review_notes": notes,
            }
            decisions.append(record)
            atomic_json(SCREENING, decisions)
            append_audit(AUDIT, record)
            break
    summary(candidates, decisions, pool_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
