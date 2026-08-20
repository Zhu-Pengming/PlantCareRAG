#!/usr/bin/env python3
"""Validate v1.2 annotation examples and claim-to-evidence integrity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT.parent / "dataset_v1" / "data" / "processed"
ALLOWED_BEHAVIORS = {"answer", "abstain", "clarify", "report_conflict"}
ALLOWED_ANSWERABILITY = {
    "answerable",
    "insufficient_evidence",
    "clarification_needed",
    "unsupported",
    "conflicting_evidence",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> list[str]:
    errors: list[str] = []
    queries = load(ROOT / "data" / "query_annotation_seed.json")
    entries = {entry["id"]: entry for entry in load(V1_ROOT / "entries.json")}
    plants = {plant["id"] for plant in load(V1_ROOT / "plants.json")}
    seen = set()

    for query in queries:
        query_id = query.get("query_id")
        if query_id in seen:
            errors.append(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        if query.get("schema_version") != "1.2.0":
            errors.append(f"invalid schema_version: {query_id}")
        expected_hash = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
        if query.get("split_hash") != expected_hash:
            errors.append(f"split_hash mismatch: {query_id}")
        if query.get("benchmark_eligible") is not False:
            errors.append(f"seed case must not be benchmark eligible: {query_id}")
        if query.get("provenance") != "authored_contract_case":
            errors.append(f"seed provenance is not explicit: {query_id}")
        if query.get("answerability") not in ALLOWED_ANSWERABILITY:
            errors.append(f"invalid answerability: {query_id}")
        if query.get("expected_behavior") not in ALLOWED_BEHAVIORS:
            errors.append(f"invalid expected_behavior: {query_id}")
        plant_id = query.get("plant_id")
        if (
            plant_id is not None
            and plant_id not in plants
            and query["answerability"] != "conflicting_evidence"
        ):
            errors.append(f"unknown accepted plant_id: {query_id}")
        for evidence_id in query.get("gold_evidence_ids", []):
            evidence = entries.get(evidence_id)
            if evidence is None:
                errors.append(f"unknown gold evidence: {query_id}/{evidence_id}")
                continue
            if plant_id and evidence["plant_id"] != plant_id:
                errors.append(f"gold evidence plant mismatch: {query_id}/{evidence_id}")
            if evidence["dimension"] not in query.get("expected_dimensions", []):
                errors.append(f"gold evidence dimension mismatch: {query_id}/{evidence_id}")
        if query["answerability"] == "answerable" and not query.get("gold_evidence_ids"):
            errors.append(f"answerable case lacks gold evidence: {query_id}")
        if query["answerability"] != "answerable" and query.get("gold_evidence_ids"):
            errors.append(f"non-answerable case declares sufficient gold: {query_id}")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"v1.2 contract validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    count = len(load(ROOT / "data" / "query_annotation_seed.json"))
    print(f"v1.2 contract is valid ({count} non-benchmark seed cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
