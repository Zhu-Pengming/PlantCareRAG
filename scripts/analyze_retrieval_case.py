#!/usr/bin/env python3
"""Inspect ranks and lexical overlap for one retrieval failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.classify_dimensions import validate_predictions
    from scripts.evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        EntityNormalizer,
        load_documents,
        load_json,
        tokenize,
    )
except ModuleNotFoundError:  # Support direct execution
    from classify_dimensions import validate_predictions
    from evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        EntityNormalizer,
        load_documents,
        load_json,
        tokenize,
    )


DEFAULT_PREDICTIONS = EVAL_ROOT / "results" / "dimension_predictions_recall.json"


def build_case_report(
    query_id: str,
    prediction_payload: dict[str, Any],
    top_n: int = 10,
) -> dict[str, Any]:
    queries = load_json(MAIN_PATH)
    validate_predictions(prediction_payload, queries)
    query = next((item for item in queries if item["query_id"] == query_id), None)
    if query is None:
        raise ValueError(f"unknown query_id: {query_id}")
    predictions = {
        prediction["query_id"]: set(prediction["dimensions"])
        for prediction in prediction_payload["predictions"]
    }

    documents = load_documents()
    document_by_id = {document.entry_id: document for document in documents}
    bm25 = BM25(documents)
    normalizer = EntityNormalizer(load_json(DATA_ROOT / "entity_aliases.json"))
    detected_plant_id = normalizer.detect(query["raw_text"])
    known_plant_ids = {document.plant_id for document in documents}
    annotated_plant_id = (
        query["plant_id"] if query.get("plant_id") in known_plant_ids else None
    )
    gold = set(query["gold_entry_ids"])

    scenarios = [
        ("bm25_raw", None, None),
        ("bm25_entity", detected_plant_id, None),
        (
            "bm25_entity_predicted_dimensions",
            detected_plant_id,
            predictions[query_id],
        ),
        (
            "bm25_entity_oracle_dimensions",
            detected_plant_id,
            set(query["expected_dimensions"]),
        ),
        ("bm25_oracle_entity", annotated_plant_id, None),
        (
            "bm25_oracle_entity_oracle_dimensions",
            annotated_plant_id,
            set(query["expected_dimensions"]),
        ),
    ]
    rankings: list[dict[str, Any]] = []
    for name, plant_filter, dimension_filter in scenarios:
        ranked = bm25.rank(query["raw_text"], plant_id=plant_filter)
        if dimension_filter is not None:
            ranked = [
                (document, score)
                for document, score in ranked
                if document.dimension in dimension_filter
            ]
        ranked_ids = [document.entry_id for document, _ in ranked]
        gold_ranks = {
            entry_id: ranked_ids.index(entry_id) + 1 if entry_id in ranked_ids else None
            for entry_id in sorted(gold)
        }
        rankings.append(
            {
                "name": name,
                "plant_filter": plant_filter,
                "dimension_filter": sorted(dimension_filter) if dimension_filter else None,
                "first_gold_rank": min(
                    (rank for rank in gold_ranks.values() if rank is not None),
                    default=None,
                ),
                "gold_ranks": gold_ranks,
                "top": [
                    {
                        "rank": index,
                        "entry_id": document.entry_id,
                        "plant_id": document.plant_id,
                        "dimension": document.dimension,
                        "score": round(score, 6),
                        "is_gold": document.entry_id in gold,
                    }
                    for index, (document, score) in enumerate(ranked[:top_n], start=1)
                ],
            }
        )

    query_tokens = set(tokenize(query["raw_text"]))
    lexical_overlap: list[dict[str, Any]] = []
    for entry_id in sorted(gold):
        document = document_by_id[entry_id]
        document_tokens = set(tokenize(document.text))
        overlap = query_tokens & document_tokens
        union = query_tokens | document_tokens
        lexical_overlap.append(
            {
                "entry_id": entry_id,
                "plant_id": document.plant_id,
                "dimension": document.dimension,
                "overlap_tokens": sorted(overlap),
                "overlap_count": len(overlap),
                "query_token_coverage": round(len(overlap) / max(len(query_tokens), 1), 6),
                "jaccard": round(len(overlap) / max(len(union), 1), 6),
            }
        )

    return {
        "schema_version": "1.0.0",
        "query": {
            "query_id": query_id,
            "raw_text": query["raw_text"],
            "plant_id": query["plant_id"],
            "detected_plant_id": detected_plant_id,
            "expected_dimensions": query["expected_dimensions"],
            "predicted_dimensions": sorted(predictions[query_id]),
            "gold_entry_ids": query["gold_entry_ids"],
            "answer_shape": query["answer_shape"],
            "context_requirement": query["context_requirement"],
            "excluded_from_single_turn": query.get("excluded_from_single_turn", False),
            "status": query["status"],
        },
        "query_tokens": sorted(query_tokens),
        "rankings": rankings,
        "gold_lexical_overlap": lexical_overlap,
        "notes": [
            "Ranks use the same global BM25 index as the main ablation.",
            "Lexical overlap uses the baseline tokenizer over the query and indexed entry text.",
            "This report records evidence only; lexical-gap versus ranking-gap interpretation remains a human judgment.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be positive")
    output = args.output or EVAL_ROOT / "results" / f"case_{args.query_id.replace(':', '_')}.json"
    report = build_case_report(
        args.query_id,
        load_json(args.predictions),
        top_n=args.top_n,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Case report: {output}")
    print(
        f"Query: {report['query']['query_id']} "
        f"answer_shape={report['query']['answer_shape']} "
        f"detected_plant={report['query']['detected_plant_id']}"
    )
    for scenario in report["rankings"]:
        print(
            f"{scenario['name']}: first_gold_rank={scenario['first_gold_rank']} "
            f"gold_ranks={scenario['gold_ranks']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
