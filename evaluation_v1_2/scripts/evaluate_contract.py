#!/usr/bin/env python3
"""Run authored contract cases without presenting them as a benchmark."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_v1_2.answerability import AnswerabilityEngine


ROOT = Path(__file__).resolve().parents[1]
STATUS_BY_LABEL = {
    "answerable": "answered",
    "insufficient_evidence": "insufficient_evidence",
    "clarification_needed": "clarification_needed",
    "unsupported": "unsupported",
    "conflicting_evidence": "conflicting_evidence",
}


def evaluate() -> dict:
    queries = json.loads(
        (ROOT / "data" / "query_annotation_seed.json").read_text(encoding="utf-8")
    )
    engine = AnswerabilityEngine()
    rows = []
    for query in queries:
        response = engine.response(query["raw_text"])
        evidence_ids = {
            evidence["entry_id"] for evidence in response.get("evidence", [])
        }
        gold = set(query["gold_evidence_ids"])
        rows.append(
            {
                "query_id": query["query_id"],
                "expected_status": STATUS_BY_LABEL[query["answerability"]],
                "actual_status": response["status"],
                "status_match": response["status"]
                == STATUS_BY_LABEL[query["answerability"]],
                "evidence_match": evidence_ids == gold,
            }
        )
    counts = Counter(
        "pass" if row["status_match"] and row["evidence_match"] else "fail"
        for row in rows
    )
    return {
        "schema_version": "1.2.0",
        "evaluation_type": "authored_contract_smoke_test",
        "benchmark_eligible": False,
        "warning": "Do not report these results as real-user QA performance.",
        "summary": {"n": len(rows), "pass": counts["pass"], "fail": counts["fail"]},
        "queries": rows,
    }


def main() -> int:
    result = evaluate()
    print(json.dumps(result, indent=2))
    return 0 if result["summary"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
