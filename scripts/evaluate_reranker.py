#!/usr/bin/env python3
"""Rerank dimension-filtered candidates with an MS MARCO cross-encoder."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from scripts.classify_dimensions import validate_predictions
    from scripts.evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        Document,
        EntityNormalizer,
        QUESTION_COLUMNS,
        load_documents,
        load_json,
        safe_ratio,
        summarize_retrieval_population,
    )
    from scripts.evaluate_dimension_ablation import build_dimension_ablation
except ModuleNotFoundError:  # Support direct execution
    from classify_dimensions import validate_predictions
    from evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        Document,
        EntityNormalizer,
        QUESTION_COLUMNS,
        load_documents,
        load_json,
        safe_ratio,
        summarize_retrieval_population,
    )
    from evaluate_dimension_ablation import build_dimension_ablation


DEFAULT_PREDICTIONS = EVAL_ROOT / "results" / "dimension_predictions_recall.json"
DEFAULT_REPORT = EVAL_ROOT / "results" / "rerank_ablation.json"
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_CARD_URL = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2"
PairScorer = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def evaluate_rerank(
    queries: list[dict[str, Any]],
    dimensions_by_query: dict[str, set[str]],
    scorer: PairScorer,
    k: int,
) -> dict[str, Any]:
    """Score every candidate left by entity and dimension filters."""
    documents = load_documents()
    bm25 = BM25(documents)
    normalizer = EntityNormalizer(load_json(DATA_ROOT / "entity_aliases.json"))
    rows: list[dict[str, Any]] = []

    for query in queries:
        gold = set(query["gold_entry_ids"])
        if not gold:
            continue
        plant_id = normalizer.detect(query["raw_text"])
        dimensions = dimensions_by_query[query["query_id"]]
        candidates = [
            (document, bm25_score)
            for document, bm25_score in bm25.rank(query["raw_text"], plant_id=plant_id)
            if document.dimension in dimensions
        ]
        pairs = [(query["raw_text"], document.text) for document, _ in candidates]
        cross_scores = [float(score) for score in scorer(pairs)] if pairs else []
        if len(cross_scores) != len(candidates):
            raise ValueError(
                f"{query['query_id']}: scorer returned {len(cross_scores)} scores "
                f"for {len(candidates)} candidates"
            )
        scored: list[tuple[Document, float, float]] = [
            (document, cross_score, bm25_score)
            for (document, bm25_score), cross_score in zip(candidates, cross_scores)
        ]
        scored.sort(key=lambda item: (-item[1], -item[2], item[0].entry_id))
        ranked_ids = [document.entry_id for document, _, _ in scored]
        top_ids = ranked_ids[:k]
        gold_size = len(gold)
        matched = gold.intersection(top_ids)
        first_rank = next(
            (index + 1 for index, entry_id in enumerate(ranked_ids) if entry_id in gold),
            None,
        )
        gold_ranks = {
            entry_id: (ranked_ids.index(entry_id) + 1 if entry_id in ranked_ids else None)
            for entry_id in sorted(gold)
        }
        rows.append(
            {
                "query_id": query["query_id"],
                "question_column": QUESTION_COLUMNS[query["question_type"]],
                "gold_size": gold_size,
                f"hit_at_{k}": int(bool(matched)),
                f"recall_at_{k}": safe_ratio(len(matched), gold_size),
                "recall_at_5": safe_ratio(
                    len(gold.intersection(ranked_ids[:5])), gold_size
                ),
                "recall_at_gold": safe_ratio(
                    len(gold.intersection(ranked_ids[:gold_size])), gold_size
                ),
                "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
                "entity_filter_plant_id": plant_id,
                "dimensions": sorted(dimensions),
                "candidate_count": len(scored),
                "top_k": top_ids,
                "top_5": ranked_ids[:5],
                "gold_ranks": gold_ranks,
                "ranking": [
                    {
                        "entry_id": document.entry_id,
                        "cross_encoder_score": round(cross_score, 6),
                        "bm25_score": round(bm25_score, 6),
                    }
                    for document, cross_score, bm25_score in scored
                ],
            }
        )

    single_turn_queries = [
        query for query in queries if not query.get("excluded_from_single_turn", False)
    ]
    return {
        "retrieval": summarize_retrieval_population(rows, queries, k),
        "retrieval_single_turn": summarize_retrieval_population(
            rows, single_turn_queries, k
        ),
    }


def build_rerank_report(
    prediction_payload: dict[str, Any],
    scorer: PairScorer,
    model_metadata: dict[str, Any],
    k: int = 3,
) -> dict[str, Any]:
    queries = load_json(MAIN_PATH)
    validate_predictions(prediction_payload, queries)
    predicted_dimensions = {
        prediction["query_id"]: set(prediction["dimensions"])
        for prediction in prediction_payload["predictions"]
    }
    oracle_dimensions = {
        query["query_id"]: set(query["expected_dimensions"]) for query in queries
    }
    existing = build_dimension_ablation(prediction_payload, k=k)
    existing_stages = {stage["name"]: stage for stage in existing["retrieval_ablation"]}
    predicted_rerank = evaluate_rerank(queries, predicted_dimensions, scorer, k)
    oracle_rerank = evaluate_rerank(queries, oracle_dimensions, scorer, k)

    stages = [
        existing_stages["bm25_raw"],
        existing_stages["bm25_entity"],
        existing_stages["bm25_entity_predicted_dimensions"],
        existing_stages["bm25_entity_oracle_dimensions"],
        {
            "name": "cross_encoder_entity_predicted_dimensions",
            "entity_filter": "current lexical entity linker",
            "dimension_filter": "recall-first few-shot LLM predictions",
            "ranker": model_metadata["model_id"],
            **predicted_rerank,
        },
        {
            "name": "cross_encoder_entity_oracle_dimensions",
            "entity_filter": "current lexical entity linker",
            "dimension_filter": "manual expected_dimensions",
            "ranker": model_metadata["model_id"],
            **oracle_rerank,
        },
    ]
    excluded_ids = [
        query["query_id"] for query in queries if query.get("excluded_from_single_turn", False)
    ]
    return {
        "schema_version": "1.0.0",
        "configuration": {
            "k": k,
            "documents": len(load_documents()),
            "full_queries": len(queries),
            "single_turn_queries": len(queries) - len(excluded_ids),
            "excluded_thread_context_queries": len(excluded_ids),
            "excluded_query_ids": excluded_ids,
            "full_compound_queries": sum(
                query["question_type"] == "compound" for query in queries
            ),
            "single_turn_compound_queries": sum(
                query["question_type"] == "compound"
                and not query.get("excluded_from_single_turn", False)
                for query in queries
            ),
            "candidate_policy": (
                "score every document remaining after current entity linking and "
                "predicted/oracle dimension filtering; do not truncate by BM25 before reranking"
            ),
        },
        "model": model_metadata,
        "dimension_classification": existing["dimension_classification"],
        "retrieval_ablation": stages,
        "notes": [
            "The first four rows preserve the frozen lexical ablation, including the pre-rerank oracle-dimension row.",
            "Every row reports both the full N=40 population and the uniformly defined N=30 single-turn population.",
            "Single-turn excludes every and only query with context_requirement=thread_context; it is not outcome-dependent.",
            "Seven same-post image-context queries remain in both views; this is not a text-only evaluation.",
            "The reranker changes ordering only; entity and dimension filters define candidate membership.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    try:
        from sentence_transformers import CrossEncoder
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "sentence-transformers is required; install requirements-rerank.txt"
        ) from exc

    model = CrossEncoder(args.model, device="cpu")

    def scorer(pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return model.predict(
            list(pairs), batch_size=args.batch_size, show_progress_bar=False
        )

    backbone = getattr(model, "model", None)
    commit_hash = getattr(getattr(backbone, "config", None), "_commit_hash", None)
    model_metadata = {
        "model_id": args.model,
        "model_revision": commit_hash,
        "model_card_url": MODEL_CARD_URL,
        "model_card_accessed": "2026-08-11",
        "device": "cpu",
        "batch_size": args.batch_size,
        "python": platform.python_version(),
        "packages": {
            "sentence-transformers": package_version("sentence-transformers"),
            "transformers": package_version("transformers"),
            "torch": package_version("torch"),
        },
    }
    report = build_rerank_report(
        load_json(args.predictions), scorer, model_metadata, k=args.k
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Rerank report: {args.output}")
    for stage in report["retrieval_ablation"]:
        full = stage["retrieval"]
        single = stage["retrieval_single_turn"]
        full_agg = full["aggregate"]
        single_agg = single["aggregate"]
        full_d = full["by_question_column"]["D_compound"]
        single_d = single["by_question_column"]["D_compound"]
        print(
            f"{stage['name']}: full R@|gold|={full_agg['recall_at_gold_macro']['value']:.3f} "
            f"R@5={full_agg['recall_at_5_macro']['value']:.3f} "
            f"D Hit@{args.k}={full_d[f'hit_at_{args.k}']['hits']}/{full_d[f'hit_at_{args.k}']['n']}; "
            f"single R@|gold|={single_agg['recall_at_gold_macro']['value']:.3f} "
            f"R@5={single_agg['recall_at_5_macro']['value']:.3f} "
            f"D Hit@{args.k}={single_d[f'hit_at_{args.k}']['hits']}/{single_d[f'hit_at_{args.k}']['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
