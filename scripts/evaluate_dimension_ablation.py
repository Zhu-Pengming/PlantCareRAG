#!/usr/bin/env python3
"""Evaluate multi-label dimension predictions and the four-stage retrieval ablation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.classify_dimensions import ALLOWED_DIMENSIONS, DEFAULT_OUTPUT, validate_predictions
    from scripts.evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        EntityNormalizer,
        QUESTION_COLUMNS,
        build_report,
        load_documents,
        load_json,
        ratio_metric,
        safe_ratio,
        summarize_retrieval_population,
    )
    from scripts.evaluate_oracle import build_oracle_report
except ModuleNotFoundError:  # Support direct execution
    from classify_dimensions import ALLOWED_DIMENSIONS, DEFAULT_OUTPUT, validate_predictions
    from evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        EntityNormalizer,
        QUESTION_COLUMNS,
        build_report,
        load_documents,
        load_json,
        ratio_metric,
        safe_ratio,
        summarize_retrieval_population,
    )
    from evaluate_oracle import build_oracle_report


DEFAULT_REPORT = EVAL_ROOT / "results" / "dimension_ablation.json"


def classification_metrics(
    queries: list[dict[str, Any]], predictions: dict[str, set[str]]
) -> dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact = 0
    errors: list[dict[str, Any]] = []
    label_counts: dict[str, Counter[str]] = {
        label: Counter() for label in ALLOWED_DIMENSIONS
    }

    for query in queries:
        query_id = query["query_id"]
        gold = set(query["expected_dimensions"])
        predicted = predictions[query_id]
        tp = len(gold & predicted)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        exact += int(gold == predicted)
        for label in ALLOWED_DIMENSIONS:
            label_counts[label]["tp"] += int(label in gold and label in predicted)
            label_counts[label]["fp"] += int(label not in gold and label in predicted)
            label_counts[label]["fn"] += int(label in gold and label not in predicted)
            label_counts[label]["support"] += int(label in gold)
        if gold != predicted:
            errors.append(
                {
                    "query_id": query_id,
                    "question_column": QUESTION_COLUMNS[query["question_type"]],
                    "gold": sorted(gold),
                    "predicted": sorted(predicted),
                    "missing": sorted(gold - predicted),
                    "extra": sorted(predicted - gold),
                }
            )

    precision = safe_ratio(total_tp, total_tp + total_fp)
    recall = safe_ratio(total_tp, total_tp + total_fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    per_label: dict[str, Any] = {}
    for label, counts in label_counts.items():
        label_precision = safe_ratio(counts["tp"], counts["tp"] + counts["fp"])
        label_recall = safe_ratio(counts["tp"], counts["tp"] + counts["fn"])
        label_f1 = safe_ratio(
            2 * label_precision * label_recall,
            label_precision + label_recall,
        )
        per_label[label] = {
            "precision": round(label_precision, 6),
            "recall": round(label_recall, 6),
            "f1": round(label_f1, 6),
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "support": counts["support"],
        }
    return {
        "micro": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "exact_match": ratio_metric(exact, len(queries), "exact"),
        "gold_label_count": sum(len(query["expected_dimensions"]) for query in queries),
        "predicted_label_count": sum(len(predictions[query["query_id"]]) for query in queries),
        "per_label": per_label,
        "errors": errors,
    }


def evaluate_predicted_dimensions(
    queries: list[dict[str, Any]],
    predictions: dict[str, set[str]],
    k: int,
) -> dict[str, Any]:
    documents = load_documents()
    bm25 = BM25(documents)
    normalizer = EntityNormalizer(load_json(DATA_ROOT / "entity_aliases.json"))
    rows: list[dict[str, Any]] = []

    for query in queries:
        gold = set(query["gold_entry_ids"])
        if not gold:
            continue
        plant_id = normalizer.detect(query["raw_text"])
        predicted_dimensions = predictions[query["query_id"]]
        ranked = [
            (document, score)
            for document, score in bm25.rank(query["raw_text"], plant_id=plant_id)
            if document.dimension in predicted_dimensions
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
                "predicted_dimensions": sorted(predicted_dimensions),
                "entity_filter_plant_id": plant_id,
                "top_k": top_ids,
            }
        )

    return summarize_retrieval_population(rows, queries, k)


def build_dimension_ablation(
    prediction_payload: dict[str, Any], k: int = 3
) -> dict[str, Any]:
    queries = load_json(MAIN_PATH)
    validate_predictions(prediction_payload, queries)
    predictions = {
        prediction["query_id"]: set(prediction["dimensions"])
        for prediction in prediction_payload["predictions"]
    }
    baseline = build_report(k=k, threshold=1.0)
    baseline_modes = {mode["name"]: mode for mode in baseline["modes"]}
    predicted_retrieval = evaluate_predicted_dimensions(queries, predictions, k)
    single_turn_queries = [
        query for query in queries if not query.get("excluded_from_single_turn", False)
    ]
    predicted_retrieval_single_turn = summarize_retrieval_population(
        predicted_retrieval["per_query"], single_turn_queries, k
    )
    oracle = build_oracle_report(k=k)
    oracle_scenarios = {scenario["name"]: scenario for scenario in oracle["scenarios"]}

    stages = [
        {
            "name": "bm25_raw",
            "entity_filter": "none",
            "dimension_filter": "none",
            "retrieval": baseline_modes["bm25_raw"]["retrieval"],
            "retrieval_single_turn": baseline_modes["bm25_raw"]["retrieval_single_turn"],
        },
        {
            "name": "bm25_entity",
            "entity_filter": "current lexical entity linker",
            "dimension_filter": "none",
            "retrieval": baseline_modes["bm25_entity"]["retrieval"],
            "retrieval_single_turn": baseline_modes["bm25_entity"]["retrieval_single_turn"],
        },
        {
            "name": "bm25_entity_predicted_dimensions",
            "entity_filter": "current lexical entity linker",
            "dimension_filter": "few-shot LLM predictions",
            "retrieval": predicted_retrieval,
            "retrieval_single_turn": predicted_retrieval_single_turn,
        },
        {
            "name": "bm25_entity_oracle_dimensions",
            "entity_filter": "current lexical entity linker",
            "dimension_filter": "manual expected_dimensions",
            "retrieval": oracle_scenarios["oracle_dimensions_current_entity"]["retrieval"],
            "retrieval_single_turn": oracle_scenarios["oracle_dimensions_current_entity"]
            ["retrieval_single_turn"],
        },
    ]
    classifier_metadata = {
        key: value
        for key, value in prediction_payload.items()
        if key != "predictions"
    }
    return {
        "schema_version": "1.0.0",
        "configuration": {
            "k": k,
            "documents": baseline["configuration"]["documents"],
            "queries": len(queries),
            "queries_with_gold": baseline["configuration"]["queries_with_gold"],
            "single_turn_queries": len(single_turn_queries),
            "excluded_thread_context_queries": len(queries) - len(single_turn_queries),
        },
        "classifier": classifier_metadata,
        "dimension_classification": classification_metrics(queries, predictions),
        "retrieval_ablation": stages,
        "notes": [
            "All four retrieval stages use the same 40 v0.2 queries and gold labels.",
            "Each stage also reports retrieval_single_turn over the uniform N=30 non-thread-context population.",
            "The LLM prompt contains synthetic examples but no expected_dimensions from the evaluation set.",
            "Predicted and oracle dimension stages keep the same lexical entity linker fixed.",
            "D-column metrics include absolute Hit counts: full N=9 and single-turn N=6.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    payload = load_json(args.predictions)
    report = build_dimension_ablation(payload, args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    micro = report["dimension_classification"]["micro"]
    exact = report["dimension_classification"]["exact_match"]
    print(f"Dimension ablation: {args.output}")
    print(
        f"Classifier: micro-F1={micro['f1']:.3f} "
        f"P={micro['precision']:.3f} R={micro['recall']:.3f} "
        f"exact={exact['exact']}/{exact['n']}"
    )
    for stage in report["retrieval_ablation"]:
        aggregate = stage["retrieval"]["aggregate"]
        compound = stage["retrieval"]["by_question_column"]["D_compound"]
        print(
            f"{stage['name']}: R@|gold|={aggregate['recall_at_gold_macro']['value']:.3f} "
            f"R@5={aggregate['recall_at_5_macro']['value']:.3f}; "
            f"D Hit@{args.k}={compound[f'hit_at_{args.k}']['hits']}/{compound[f'hit_at_{args.k}']['n']} "
            f"D R@5={compound['recall_at_5_macro']['value']:.3f} "
            f"D R@|gold|={compound['recall_at_gold_macro']['value']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
