#!/usr/bin/env python3
"""Evaluate citation-bound answers using the frozen semantic route decisions.

This is a component/regression evaluation, not real-user QA. It deliberately
reuses the already-frozen route output so the generation and citation layer can
be tested without reopening model selection or touching the test split.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_v1_2.rag_engine import (
    GroundedRAGEngine,
    RouteDecision,
    validate_grounded_response,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "synthetic_benchmark.json"
ROUTES = ROOT / "results" / "semantic_dimension_router.json"
OUTPUT = ROOT / "results" / "grounded_answer_contract.json"


def behavior(status: str) -> str:
    if status == "answered":
        return "answer"
    if status == "clarification_needed":
        return "clarify"
    return "abstain"


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    gold_count = sum(row["gold_count"] for row in rows)
    recovered = sum(row["gold_recovered"] for row in rows)
    answered = [row for row in rows if row["actual_behavior"] == "answer"]
    return {
        "n": n,
        "behavior_accuracy": sum(row["behavior_match"] for row in rows) / n,
        "evidence_exact_match": sum(row["evidence_exact_match"] for row in rows) / n,
        "evidence_micro_recall": recovered / gold_count if gold_count else 1.0,
        "grounded_response_rate": sum(not row["grounding_errors"] for row in rows) / n,
        "answered_citation_validity": (
            sum(row["citation_valid"] for row in answered) / len(answered)
            if answered else 1.0
        ),
        "verified_only_rate": sum(row["verified_only"] for row in rows) / n,
        "numeric_claim_grounding_rate": sum(row["numbers_grounded"] for row in rows) / n,
        "unsafe_answer_count": sum(
            row["expected_behavior"] != "answer" and row["actual_behavior"] == "answer"
            for row in rows
        ),
    }


def evaluate() -> dict:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    route_artifact = json.loads(ROUTES.read_text(encoding="utf-8"))
    routes = {row["query_id"]: row for row in route_artifact["queries"]}
    engine = GroundedRAGEngine()
    rows = []
    for query in benchmark:
        frozen = routes[query["query_id"]]
        decision = RouteDecision(
            tuple(frozen["actual_dimensions"]),
            "frozen_" + frozen["route"],
            frozen.get("semantic_scores"),
        )
        response = engine.response(query["raw_text"], route_decision=decision)
        actual_ids = {item["claim_id"] for item in response["evidence"]}
        gold_ids = set(query["gold_evidence_ids"])
        grounding_errors = validate_grounded_response(response)
        cited_text = " ".join(item["claim_text"] for item in response["evidence"])
        answer_without_citations = re.sub(
            r"\[claim:[^\]]+\]", "", response["answer"]
        )
        answer_numbers = numeric_tokens(answer_without_citations)
        rows.append(
            {
                "query_id": query["query_id"],
                "split": "dev" if int(query["split_hash"], 16) % 2 == 0 else "test",
                "expected_behavior": query["expected_behavior"],
                "actual_behavior": behavior(response["status"]),
                "behavior_match": behavior(response["status"]) == query["expected_behavior"],
                "expected_dimensions": query["expected_dimensions"],
                "actual_dimensions": frozen["actual_dimensions"],
                "gold_evidence_ids": sorted(gold_ids),
                "actual_evidence_ids": sorted(actual_ids),
                "evidence_exact_match": actual_ids == gold_ids,
                "gold_recovered": len(actual_ids & gold_ids),
                "gold_count": len(gold_ids),
                "grounding_errors": grounding_errors,
                "citation_valid": not grounding_errors and all(
                    item["citation"].get("url")
                    and item["citation"].get("section")
                    and item["citation"].get("paragraph")
                    for item in response["evidence"]
                ),
                "verified_only": all(
                    item["review_status"] == "human_verified"
                    for item in response["evidence"]
                ),
                "numbers_grounded": answer_numbers <= numeric_tokens(cited_text),
                "response": response,
            }
        )

    result = {
        "schema_version": "1.0.0",
        "evaluation_type": "frozen_route_grounded_answer_contract",
        "external_validity": False,
        "warning": (
            "Synthetic questions were generated from the evaluated evidence. "
            "This measures answer/citation contract behavior, not real-user QA quality."
        ),
        "routing_artifact": "evaluation_v1_2/results/semantic_dimension_router.json",
        "routing_artifact_revision": route_artifact["resolved_model_revision"],
        "overall": metrics(rows),
        "by_split": {
            split: metrics([row for row in rows if row["split"] == split])
            for split in ("dev", "test")
        },
        "status_counts": dict(
            sorted(Counter(row["response"]["status"] for row in rows).items())
        ),
        "failure_count": sum(
            not row["behavior_match"]
            or not row["evidence_exact_match"]
            or bool(row["grounding_errors"])
            for row in rows
        ),
        "queries": rows,
    }
    return result


def main() -> int:
    result = evaluate()
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "overall": result["overall"],
                "by_split": result["by_split"],
                "status_counts": result["status_counts"],
                "failure_count": result["failure_count"],
            },
            indent=2,
        )
    )
    return 0 if result["overall"]["grounded_response_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
