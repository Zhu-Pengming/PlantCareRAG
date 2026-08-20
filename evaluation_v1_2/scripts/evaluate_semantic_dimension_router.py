#!/usr/bin/env python3
"""Evaluate the dev-tuned semantic fallback on the frozen benchmark."""

from __future__ import annotations

from collections import Counter
from datetime import date
import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_v1.query_engine import QueryEngine, detect_unsupported_dimension, mask_entity
from evaluation_v1_2.evidence_store import VerifiedEvidenceStore
from evaluation_v1_2.semantic_dimension_router import SemanticDimensionRouter
from evaluation_v1_2.scripts.evaluate_synthetic_benchmark import metrics, style


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "synthetic_benchmark.json"
CONFIG = ROOT / "config" / "semantic_dimension_router.json"
LEXICAL_RESULT = ROOT / "results" / "synthetic_contract_baseline.json"
OUTPUT = ROOT / "results" / "semantic_dimension_router.json"
UNRESOLVED_MAPPINGS = {"plant:monstera", "plant:hoya_wax_plant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen semantic fallback on dev and test."
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Explicitly reproduce the already-recorded test evaluation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("test_accessed") and not args.reproduce:
        print(
            "test result is already frozen; use --reproduce only for an explicit reproduction",
            file=sys.stderr,
        )
        return 2
    queries = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    linker = QueryEngine()
    store = VerifiedEvidenceStore()
    router = SemanticDimensionRouter(
        threshold=config["threshold"],
        model_name=config["model"],
    )
    rows = []
    for query in queries:
        entity = linker.link_entity(query["raw_text"])
        plant_id = entity["plant_id"] if entity else None
        masked = mask_entity(query["raw_text"], entity["name"] if entity else None)
        unsupported = detect_unsupported_dimension(masked)
        if unsupported:
            dimensions = [unsupported]
            route = {"route": "unsupported_gate", "semantic_scores": {}}
        else:
            dimensions, route = router.predict(
                query["raw_text"],
                entity["name"] if entity else None,
            )
        if plant_id in UNRESOLVED_MAPPINGS:
            behavior, evidence_ids = "clarify", []
        elif unsupported:
            behavior, evidence_ids = "abstain", []
        elif plant_id is None or not dimensions:
            behavior, evidence_ids = "clarify", []
        else:
            evidence_ids = [
                row["claim_id"]
                for row in store.retrieve(plant_id=plant_id, dimensions=dimensions)
            ]
            behavior = "answer" if evidence_ids else "abstain"
        gold = set(query["gold_evidence_ids"])
        actual = set(evidence_ids)
        rows.append(
            {
                "query_id": query["query_id"],
                "split": "dev" if int(query["split_hash"], 16) % 2 == 0 else "test",
                "generation_style": style(query["query_id"]),
                "route": route["route"],
                "semantic_scores": route.get("semantic_scores", {}),
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
        )

    lexical = json.loads(LEXICAL_RESULT.read_text(encoding="utf-8"))
    overall = metrics(rows)
    by_split = {
        split: metrics([row for row in rows if row["split"] == split])
        for split in ("dev", "test")
    }
    result = {
        "schema_version": "1.0.0",
        "evaluation_type": "semantic_fallback_dimension_routing",
        "benchmark_sha256": json.loads(
            (ROOT / "reports" / "synthetic_benchmark_stats.json").read_text(encoding="utf-8")
        )["benchmark_sha256"],
        "model": config["model"],
        "resolved_model_repo": config["resolved_model_repo"],
        "resolved_model_revision": config["resolved_model_revision"],
        "threshold": config["threshold"],
        "threshold_selected_on": "dev",
        "overall": overall,
        "by_split": by_split,
        "by_generation_style": {
            value: metrics([row for row in rows if row["generation_style"] == value])
            for value in sorted({row["generation_style"] for row in rows})
        },
        "route_counts": dict(sorted(Counter(row["route"] for row in rows).items())),
        "improvement_vs_lexical": {
            metric: overall[metric] - lexical["overall"][metric]
            for metric in (
                "dimension_exact_match",
                "behavior_accuracy",
                "evidence_exact_match",
                "evidence_micro_recall",
            )
        },
        "test_improvement_vs_lexical": {
            metric: by_split["test"][metric] - lexical["by_split"]["test"][metric]
            for metric in (
                "dimension_exact_match",
                "behavior_accuracy",
                "evidence_exact_match",
                "evidence_micro_recall",
            )
        },
        "failures": [
            row
            for row in rows
            if not (
                row["entity_match"]
                and row["dimension_exact_match"]
                and row["behavior_match"]
                and row["evidence_exact_match"]
            )
        ],
        "queries": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    config.update(
        test_accessed=True,
        test_evaluated_on=date.today().isoformat(),
        test_result_file="evaluation_v1_2/results/semantic_dimension_router.json",
        test_dimension_exact_match=by_split["test"]["dimension_exact_match"],
    )
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall": overall,
        "by_split": by_split,
        "by_generation_style": result["by_generation_style"],
        "route_counts": result["route_counts"],
        "improvement_vs_lexical": result["improvement_vs_lexical"],
        "test_improvement_vs_lexical": result["test_improvement_vs_lexical"],
        "failure_count": len(result["failures"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
