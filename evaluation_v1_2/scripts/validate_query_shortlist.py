#!/usr/bin/env python3
"""Validate that automated query outputs cannot masquerade as benchmark gold."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "query_candidates.json"
TRIAGE = ROOT / "data" / "query_auto_triage.json"
SHORTLIST = ROOT / "data" / "query_shortlist.json"


def validate() -> list[str]:
    errors: list[str] = []
    candidates = {row["candidate_id"] for row in json.loads(CANDIDATES.read_text(encoding="utf-8"))}
    triage = json.loads(TRIAGE.read_text(encoding="utf-8"))
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))
    triage_ids = [row["candidate_id"] for row in triage]
    shortlist_ids = [row["candidate_id"] for row in shortlist]
    if len(triage_ids) != len(set(triage_ids)):
        errors.append("auto-triage contains duplicate candidate IDs")
    if len(shortlist_ids) != len(set(shortlist_ids)):
        errors.append("shortlist contains duplicate candidate IDs")
    if set(triage_ids) != candidates:
        errors.append("auto-triage does not cover the frozen candidate set exactly")
    if not set(shortlist_ids) <= candidates:
        errors.append("shortlist contains unknown candidates")
    quotas = Counter(row.get("quota_plant_id") for row in shortlist)
    if len(set(quotas.values())) > 1:
        errors.append(f"shortlist plant quotas are unbalanced: {dict(quotas)}")
    for row in triage + shortlist:
        candidate_id = row.get("candidate_id", "<missing>")
        if row.get("automation_only") is not True:
            errors.append(f"automation_only is not true: {candidate_id}")
        if row.get("benchmark_eligible") is not False:
            errors.append(f"automated row marked benchmark eligible: {candidate_id}")
        if "gold_evidence_ids" in row:
            errors.append(f"automated row illegally declares gold: {candidate_id}")
        if not set(row.get("covered_dimensions", [])) <= set(row.get("predicted_dimensions", [])):
            errors.append(f"covered dimensions are not predicted: {candidate_id}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"query shortlist invalid ({len(errors)} errors):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    shortlist = json.loads(SHORTLIST.read_text(encoding="utf-8"))
    print(f"query shortlist valid ({len(shortlist)} automation-only rows; 0 benchmark eligible)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
