#!/usr/bin/env python3
"""Validate provenance and invariants of unlabeled real-query candidates."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "query_candidates.json"
PLANTS = ROOT / "data" / "overlay_plants.json"
ALLOWED_LICENSES = {"CC BY-SA 2.5", "CC BY-SA 3.0", "CC BY-SA 4.0"}


def validate(path: Path = CANDIDATES) -> list[str]:
    errors: list[str] = []
    rows = json.loads(path.read_text(encoding="utf-8"))
    plants = {row["plant_id"] for row in json.loads(PLANTS.read_text(encoding="utf-8"))}
    seen_ids: set[str] = set()
    seen_records: set[str] = set()
    for row in rows:
        candidate_id = row.get("candidate_id", "")
        record_id = row.get("source_record_id", "")
        if candidate_id in seen_ids:
            errors.append(f"duplicate candidate_id: {candidate_id}")
        if record_id in seen_records:
            errors.append(f"duplicate source_record_id: {record_id}")
        seen_ids.add(candidate_id)
        seen_records.add(record_id)
        if candidate_id != f"qv12:stackexchange_gardening:{record_id}":
            errors.append(f"candidate/source ID mismatch: {candidate_id}")
        if row.get("source_url") != f"https://gardening.stackexchange.com/questions/{record_id}":
            errors.append(f"source URL mismatch: {candidate_id}")
        expected_hash = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
        if row.get("split_hash") != expected_hash:
            errors.append(f"split hash mismatch: {candidate_id}")
        if row.get("annotation_status") != "pending_human_annotation":
            errors.append(f"candidate is not pending: {candidate_id}")
        if row.get("benchmark_eligible") is not False:
            errors.append(f"unreviewed candidate marked eligible: {candidate_id}")
        if row.get("content_license") not in ALLOWED_LICENSES:
            errors.append(f"unsupported/missing license: {candidate_id}")
        if not row.get("alias_matches"):
            errors.append(f"candidate lacks selection evidence: {candidate_id}")
        for match in row.get("alias_matches", []):
            if match.get("plant_id") not in plants:
                errors.append(f"unknown matched plant: {candidate_id}/{match.get('plant_id')}")
        author = row.get("author", {})
        if author.get("owner_user_id") and not author.get("display_name"):
            errors.append(f"known owner lacks attribution name: {candidate_id}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"query candidates invalid ({len(errors)} errors):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    count = len(json.loads(CANDIDATES.read_text(encoding="utf-8")))
    print(f"query candidates valid ({count} pending; 0 benchmark eligible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
