#!/usr/bin/env python3
"""Validate the committed end-to-end grounded-answer artifact."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def validate() -> list[str]:
    errors: list[str] = []
    result = json.loads(
        (ROOT / "results" / "grounded_answer_contract.json").read_text(encoding="utf-8")
    )
    rows = result.get("queries", [])
    if len(rows) != 80 or len({row.get("query_id") for row in rows}) != 80:
        errors.append("grounded-answer result must contain 80 unique queries")
    if result.get("external_validity") is not False:
        errors.append("synthetic artifact must explicitly deny external validity")
    if result.get("overall", {}).get("grounded_response_rate") != 1.0:
        errors.append("not every response passes grounding validation")
    if result.get("overall", {}).get("answered_citation_validity") != 1.0:
        errors.append("not every answered response has complete citations")
    if result.get("overall", {}).get("verified_only_rate") != 1.0:
        errors.append("a response exposed non-verified evidence")
    if result.get("overall", {}).get("numeric_claim_grounding_rate") != 1.0:
        errors.append("an answer contains a number absent from its cited evidence")
    if result.get("overall", {}).get("unsafe_answer_count") != 0:
        errors.append("a non-answerable query received a factual answer")
    test = result.get("by_split", {}).get("test", {})
    if test.get("n") != 35:
        errors.append("frozen test denominator is not 35")
    if any(row.get("grounding_errors") for row in rows):
        errors.append("query rows contain grounding errors")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"grounded-answer experiment invalid ({len(errors)} errors):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("grounded-answer experiment valid (80 citation-bound synthetic cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
