#!/usr/bin/env python3
"""Validate evidence-conditioned synthetic benchmark invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "synthetic_benchmark.json"
EVIDENCE = ROOT / "data" / "evidence_overlay.json"


def validate() -> list[str]:
    errors: list[str] = []
    rows = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    verified = {
        row["claim_id"]: row
        for row in json.loads(EVIDENCE.read_text(encoding="utf-8"))
        if row["review_status"] == "human_verified"
    }
    seen: set[str] = set()
    if len(rows) != 80:
        errors.append(f"expected 80 rows, found {len(rows)}")
    for row in rows:
        query_id = row.get("query_id", "<missing>")
        if query_id in seen:
            errors.append(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        if row.get("provenance") != "synthetic_generated_case":
            errors.append(f"wrong provenance: {query_id}")
        if row.get("source_url") is not None:
            errors.append(f"synthetic query has source URL: {query_id}")
        if row.get("benchmark_eligible") is not True:
            errors.append(f"synthetic contract case is disabled: {query_id}")
        if row.get("split_hash") != hashlib.sha256(query_id.encode("utf-8")).hexdigest():
            errors.append(f"split hash mismatch: {query_id}")
        gold = row.get("gold_evidence_ids", [])
        if row.get("answerability") == "answerable" and not gold:
            errors.append(f"answerable query lacks gold: {query_id}")
        if row.get("answerability") != "answerable" and gold:
            errors.append(f"non-answerable query has gold: {query_id}")
        for claim_id in gold:
            claim = verified.get(claim_id)
            if claim is None:
                errors.append(f"gold is not verified: {query_id}/{claim_id}")
                continue
            if claim["plant_id"] != row["plant_id"]:
                errors.append(f"gold plant mismatch: {query_id}/{claim_id}")
            if claim["dimension"] not in row["expected_dimensions"]:
                errors.append(f"gold dimension mismatch: {query_id}/{claim_id}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"synthetic benchmark invalid ({len(errors)} errors):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("synthetic benchmark valid (80 deterministic cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
