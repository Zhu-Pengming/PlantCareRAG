#!/usr/bin/env python3
"""Evaluate structured routing/retrieval on the synthetic contract benchmark."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_v1.query_engine import QueryEngine, detect_unsupported_dimension, mask_entity
from evaluation_v1_2.answerability import classify_dimensions
from evaluation_v1_2.evidence_store import VerifiedEvidenceStore


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "synthetic_benchmark.json"
OUTPUT = ROOT / "results" / "synthetic_contract_baseline.json"
UNRESOLVED_MAPPINGS = {"plant:monstera", "plant:hoya_wax_plant"}


def run_query(query: dict, linker: QueryEngine, store: VerifiedEvidenceStore) -> dict:
    entity = linker.link_entity(query["raw_text"])
    plant_id = entity["plant_id"] if entity else None
    masked = mask_entity(query["raw_text"], entity["name"] if entity else None)
    unsupported = detect_unsupported_dimension(masked)
    dimensions = [unsupported] if unsupported else classify_dimensions(masked)
    if plant_id in UNRESOLVED_MAPPINGS:
        behavior, evidence_ids = "clarify", []
    elif unsupported:
        behavior, evidence_ids = "abstain", []
    elif plant_id is None:
        behavior, evidence_ids = "clarify", []
    elif not dimensions:
        behavior, evidence_ids = "clarify", []
    else:
        evidence_ids = [
            row["claim_id"]
            for row in store.retrieve(plant_id=plant_id, dimensions=dimensions)
        ]
        behavior = "answer" if evidence_ids else "abstain"
    gold = set(query["gold_evidence_ids"])
    actual = set(evidence_ids)
    return {
        "query_id": query["query_id"],
        "split": "dev" if int(query["split_hash"], 16) % 2 == 0 else "test",
        "expected_plant_id": query["plant_id"],
        "actual_plant_id": plant_id,
        "entity_match": plant_id == query["plant_id"],
        "expected_dimensions": query["expected_dimensions"],
        "actual_dimensions": dimensions,
        "dimension_exact_match": set(dimensions) == set(query["expected_dimensions"]),
        "expected_behavior": query["expected_behavior"],
        "actual_behavior": behavior,
        "behavior_match": behavior == query["expected_behavior"],
        "gold_evidence_ids": sorted(gold),
        "actual_evidence_ids": sorted(actual),
        "evidence_exact_match": actual == gold,
        "gold_recovered": len(actual & gold),
        "gold_count": len(gold),
    }


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    gold_count = sum(row["gold_count"] for row in rows)
    recovered = sum(row["gold_recovered"] for row in rows)
    return {
        "n": n,
        "entity_accuracy": sum(row["entity_match"] for row in rows) / n,
        "dimension_exact_match": sum(row["dimension_exact_match"] for row in rows) / n,
        "behavior_accuracy": sum(row["behavior_match"] for row in rows) / n,
        "evidence_exact_match": sum(row["evidence_exact_match"] for row in rows) / n,
        "evidence_micro_recall": recovered / gold_count if gold_count else 1.0,
    }


def style(query_id: str) -> str:
    suffix = query_id.rsplit(":", 1)[-1]
    if suffix in {"v1", "v2", "v3"}:
        return suffix
    if suffix == "multi":
        return "multi_dimension"
    if suffix == "pet_safety":
        return "unsupported_boundary"
    return "mapping_clarification"


def evaluate() -> dict:
    queries = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    linker = QueryEngine()
    store = VerifiedEvidenceStore()
    rows = [run_query(query, linker, store) for query in queries]
    for row in rows:
        row["generation_style"] = style(row["query_id"])
    return {
        "schema_version": "1.0.0",
        "evaluation_type": "evidence_conditioned_synthetic_contract_benchmark",
        "external_validity": False,
        "warning": "Generated from the evaluated evidence; do not report as real-user QA performance.",
        "overall": metrics(rows),
        "by_split": {
            split: metrics([row for row in rows if row["split"] == split])
            for split in ("dev", "test")
        },
        "by_generation_style": {
            value: metrics([row for row in rows if row["generation_style"] == value])
            for value in sorted({row["generation_style"] for row in rows})
        },
        "failure_counts": dict(
            Counter(
                failure
                for row in rows
                for failure, passed in (
                    ("entity", row["entity_match"]),
                    ("dimension", row["dimension_exact_match"]),
                    ("behavior", row["behavior_match"]),
                    ("evidence", row["evidence_exact_match"]),
                )
                if not passed
            )
        ),
        "queries": rows,
    }


def main() -> int:
    result = evaluate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("overall", "by_split", "by_generation_style", "failure_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
