#!/usr/bin/env python3
"""Measure the upper bound from manually annotated query dimensions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        EntityNormalizer,
        QUESTION_COLUMNS,
        load_documents,
        load_json,
        safe_ratio,
        summarize_retrieval_population,
    )
except ModuleNotFoundError:  # Support direct execution: python3 scripts/evaluate_oracle.py
    from evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        EntityNormalizer,
        QUESTION_COLUMNS,
        load_documents,
        load_json,
        safe_ratio,
        summarize_retrieval_population,
    )


DEFAULT_REPORT = EVAL_ROOT / "results" / "oracle_dimensions.json"


def evaluate_scenario(
    name: str,
    bm25: BM25,
    normalizer: EntityNormalizer,
    queries: list[dict[str, Any]],
    known_plant_ids: set[str],
    k: int,
    use_gold_entity: bool,
    use_oracle_dimensions: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        gold = set(query["gold_entry_ids"])
        if not gold:
            continue
        if use_gold_entity and query.get("plant_id") in known_plant_ids:
            plant_id = query["plant_id"]
        else:
            plant_id = normalizer.detect(query["raw_text"])
        expected_dimensions = set(query["expected_dimensions"])
        ranked = bm25.rank(query["raw_text"], plant_id=plant_id)
        if use_oracle_dimensions:
            ranked = [
                (document, score)
                for document, score in ranked
                if document.dimension in expected_dimensions
            ]
        ranked_ids = [document.entry_id for document, _ in ranked]
        top_ids = ranked_ids[:k]
        gold_size = len(gold)
        matched = gold.intersection(top_ids)
        first_rank = next(
            (index + 1 for index, entry_id in enumerate(ranked_ids) if entry_id in gold),
            None,
        )
        rows.append(
            {
                "query_id": query["query_id"],
                "question_column": QUESTION_COLUMNS[query["question_type"]],
                "gold_size": gold_size,
                f"hit_at_{k}": int(bool(matched)),
                f"recall_at_{k}": safe_ratio(len(matched), gold_size),
                "recall_at_5": safe_ratio(len(gold.intersection(ranked_ids[:5])), gold_size),
                "recall_at_gold": safe_ratio(
                    len(gold.intersection(ranked_ids[:gold_size])), gold_size
                ),
                "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
                "expected_dimensions": sorted(expected_dimensions),
                "entity_filter_plant_id": plant_id,
                "top_k": top_ids,
            }
        )

    single_turn_queries = [
        query for query in queries if not query.get("excluded_from_single_turn", False)
    ]
    return {
        "name": name,
        "assumptions": {
            "dimensions": "manual expected_dimensions" if use_oracle_dimensions else "unfiltered",
            "entity": "manual plant_id" if use_gold_entity else "current lexical entity linker",
        },
        "retrieval": summarize_retrieval_population(rows, queries, k),
        "retrieval_single_turn": summarize_retrieval_population(
            rows, single_turn_queries, k
        ),
    }


def build_oracle_report(k: int = 3) -> dict[str, Any]:
    documents = load_documents()
    document_by_id = {document.entry_id: document for document in documents}
    bm25 = BM25(documents)
    normalizer = EntityNormalizer(load_json(DATA_ROOT / "entity_aliases.json"))
    queries = load_json(MAIN_PATH)
    known_plant_ids = {document.plant_id for document in documents}

    label_violations: list[dict[str, Any]] = []
    for query in queries:
        expected_dimensions = set(query["expected_dimensions"])
        gold_dimensions = {
            document_by_id[entry_id].dimension for entry_id in query["gold_entry_ids"]
        }
        if not gold_dimensions.issubset(expected_dimensions):
            label_violations.append(
                {
                    "query_id": query["query_id"],
                    "gold_dimensions_missing_from_expected": sorted(
                        gold_dimensions - expected_dimensions
                    ),
                }
            )

    scenarios = [
        evaluate_scenario(
            "oracle_dimensions_current_entity",
            bm25,
            normalizer,
            queries,
            known_plant_ids,
            k,
            use_gold_entity=False,
            use_oracle_dimensions=True,
        ),
        evaluate_scenario(
            "oracle_entity_only",
            bm25,
            normalizer,
            queries,
            known_plant_ids,
            k,
            use_gold_entity=True,
            use_oracle_dimensions=False,
        ),
        evaluate_scenario(
            "oracle_dimensions_and_entity",
            bm25,
            normalizer,
            queries,
            known_plant_ids,
            k,
            use_gold_entity=True,
            use_oracle_dimensions=True,
        ),
    ]
    return {
        "schema_version": "1.0.0",
        "configuration": {
            "k": k,
            "documents": len(documents),
            "queries": len(queries),
            "queries_with_gold": sum(bool(query["gold_entry_ids"]) for query in queries),
            "single_turn_queries": sum(
                not query.get("excluded_from_single_turn", False) for query in queries
            ),
            "excluded_thread_context_queries": sum(
                query.get("excluded_from_single_turn", False) for query in queries
            ),
        },
        "label_integrity": {
            "gold_dimensions_are_covered_by_expected_dimensions": not label_violations,
            "violations": label_violations,
        },
        "scenarios": scenarios,
        "notes": [
            "expected_dimensions are existing human annotations; this script does not generate or modify labels.",
            "The primary oracle keeps the current entity linker fixed and isolates dimension-filter headroom.",
            "The entity-only oracle isolates losses from context-dependent entity linking without dimension filtering.",
            "The full-routing oracle uses both annotated plant_id and expected_dimensions.",
            "Scores use the same global BM25 index; filters remove candidates without recalculating IDF.",
            "Every scenario also reports a uniform single-turn view that excludes all 10 thread-context queries.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    report = build_oracle_report(args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Oracle report: {args.output}")
    print(
        "Label integrity: "
        f"{report['label_integrity']['gold_dimensions_are_covered_by_expected_dimensions']}"
    )
    for scenario in report["scenarios"]:
        aggregate = scenario["retrieval"]["aggregate"]
        compound = scenario["retrieval"]["by_question_column"]["D_compound"]
        print(
            f"{scenario['name']} (N={aggregate['eligible_queries']}): "
            f"Hit@{args.k}={aggregate[f'hit_at_{args.k}']['value']:.3f} "
            f"Recall@5={aggregate['recall_at_5_macro']['value']:.3f} "
            f"Recall@|gold|={aggregate['recall_at_gold_macro']['value']:.3f}; "
            f"D Hit@{args.k}={compound[f'hit_at_{args.k}']['hits']}/{compound[f'hit_at_{args.k}']['n']} "
            f"D Recall@|gold|={compound['recall_at_gold_macro']['value']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
