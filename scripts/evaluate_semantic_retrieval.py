#!/usr/bin/env python3
"""Evaluate content-only BM25, multilingual embeddings, and fixed RRF on one split."""

from __future__ import annotations

import argparse
import gc
import json
import platform
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from scripts.create_evaluation_split import query_ids_for_split
    from scripts.evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        Document,
        QUESTION_COLUMNS,
        load_documents,
        safe_ratio,
        summarize_retrieval,
        summarize_retrieval_population,
    )
    from scripts.validate_kb import load_json
except ModuleNotFoundError:  # Support direct execution
    from create_evaluation_split import query_ids_for_split
    from evaluate_baseline import (
        DATA_ROOT,
        EVAL_ROOT,
        MAIN_PATH,
        BM25,
        Document,
        QUESTION_COLUMNS,
        load_documents,
        safe_ratio,
        summarize_retrieval,
        summarize_retrieval_population,
    )
    from validate_kb import load_json


DEFAULT_EXPERIMENT = EVAL_ROOT / "experiments" / "semantic_retrieval_v1.json"
DEFAULT_SPLIT = EVAL_ROOT / "splits" / "main_sha256_20_20_v1.json"
DEFAULT_OUTPUT = EVAL_ROOT / "results" / "semantic_retrieval_dev.json"
EmbeddingRanker = Callable[[list[dict[str, Any]], list[Document]], dict[str, list[str]]]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def validate_run_scope(
    split: str, confirm_test: bool, model_ids: list[str] | None
) -> None:
    if split == "test" and not confirm_test:
        raise ValueError("test is gated until the dev choice is frozen")
    if split == "test" and (not model_ids or len(model_ids) != 1):
        raise ValueError("test must run exactly one explicitly selected model")


def load_content_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted((DATA_ROOT / "entries").glob("*.json")):
        for entry in load_json(path):
            documents.append(
                Document(
                    entry_id=entry["id"],
                    plant_id=entry["plant_id"],
                    dimension=entry["dimension"],
                    text=entry["content"],
                )
            )
    return documents


def ranked_ids_from_bm25(
    queries: list[dict[str, Any]], documents: list[Document]
) -> dict[str, list[str]]:
    index = BM25(documents)
    return {
        query["query_id"]: [
            document.entry_id for document, _ in index.rank(query["raw_text"])
        ]
        for query in queries
    }


def positive_ranked_ids_from_bm25(
    queries: list[dict[str, Any]], documents: list[Document]
) -> dict[str, list[str]]:
    index = BM25(documents)
    return {
        query["query_id"]: [
            document.entry_id
            for document, score in index.rank(query["raw_text"])
            if score > 0
        ]
        for query in queries
    }


def reciprocal_rank_fusion(
    rankings: Sequence[list[str]],
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[str]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("RRF weights must match the number of rankings")
    if any(weight <= 0 for weight in weights):
        raise ValueError("RRF weights must be positive")
    scores: defaultdict[str, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, entry_id in enumerate(ranking, start=1):
            scores[entry_id] += weight / (rrf_k + rank)
    return sorted(scores, key=lambda entry_id: (-scores[entry_id], entry_id))


def evaluate_rankings(
    queries: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    k: int = 3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        gold = set(query["gold_entry_ids"])
        if not gold:
            continue
        ranked_ids = rankings[query["query_id"]]
        top_ids = ranked_ids[:k]
        gold_size = len(gold)
        matched = gold.intersection(top_ids)
        first_rank = next(
            (rank for rank, entry_id in enumerate(ranked_ids, start=1) if entry_id in gold),
            None,
        )
        rows.append(
            {
                "query_id": query["query_id"],
                "question_column": QUESTION_COLUMNS[query["question_type"]],
                "plant_id": query.get("plant_id"),
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
                "gold_hits_at_5": len(gold.intersection(ranked_ids[:5])),
                "top_k": top_ids,
                "top_5": ranked_ids[:5],
                "gold_ranks": {
                    entry_id: (
                        ranked_ids.index(entry_id) + 1 if entry_id in ranked_ids else None
                    )
                    for entry_id in sorted(gold)
                },
            }
        )

    single_turn_queries = [
        query for query in queries if not query.get("excluded_from_single_turn", False)
    ]

    def summarize_view(view_queries: list[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_retrieval_population(rows, view_queries, k)
        included = {query["query_id"] for query in view_queries}
        view_rows = [row for row in rows if row["query_id"] in included]
        total_pairs = sum(row["gold_size"] for row in view_rows)
        pair_hits_at_5 = sum(row["gold_hits_at_5"] for row in view_rows)
        summary["query_gold_pair_recall_at_5"] = {
            "value": round(safe_ratio(pair_hits_at_5, total_pairs), 6),
            "hits": pair_hits_at_5,
            "n": total_pairs,
        }
        by_plant: dict[str, Any] = {}
        for plant_id in sorted({query.get("plant_id") or "null" for query in view_queries}):
            plant_rows = [
                row for row in view_rows if (row.get("plant_id") or "null") == plant_id
            ]
            by_plant[plant_id] = summarize_retrieval(plant_rows, k)
        summary["by_plant_diagnostic_only"] = by_plant
        return summary

    return {
        "retrieval": summarize_view(queries),
        "retrieval_single_turn": summarize_view(single_turn_queries),
    }


def model_revision(model: Any) -> str | None:
    for module in model:
        auto_model = getattr(module, "auto_model", None)
        revision = getattr(getattr(auto_model, "config", None), "_commit_hash", None)
        if revision:
            return revision
    return None


def sentence_transformer_ranker(
    model: Any,
    model_spec: dict[str, Any],
    batch_size: int,
) -> EmbeddingRanker:
    def rank(
        queries: list[dict[str, Any]], documents: list[Document]
    ) -> dict[str, list[str]]:
        import numpy as np

        query_texts = [
            model_spec["query_prefix"] + query["raw_text"] for query in queries
        ]
        passage_texts = [model_spec["passage_prefix"] + document.text for document in documents]
        query_embeddings = model.encode(
            query_texts,
            batch_size=batch_size,
            normalize_embeddings=model_spec["normalize_embeddings"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        passage_embeddings = model.encode(
            passage_texts,
            batch_size=batch_size,
            normalize_embeddings=model_spec["normalize_embeddings"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embeddings = query_embeddings.astype(np.float64, copy=False)
        passage_embeddings = passage_embeddings.astype(np.float64, copy=False)
        if not np.isfinite(query_embeddings).all() or not np.isfinite(
            passage_embeddings
        ).all():
            raise ValueError(f"{model_spec['model_id']}: non-finite embedding values")
        similarities = np.einsum(
            "ik,jk->ij", query_embeddings, passage_embeddings, optimize=False
        )
        if not np.isfinite(similarities).all():
            raise ValueError(f"{model_spec['model_id']}: non-finite similarity values")
        output: dict[str, list[str]] = {}
        for query, scores in zip(queries, similarities):
            order = sorted(
                range(len(documents)),
                key=lambda index: (-float(scores[index]), documents[index].entry_id),
            )
            output[query["query_id"]] = [documents[index].entry_id for index in order]
        return output

    return rank


def selection_key(result: dict[str, Any]) -> tuple[float, float]:
    retrieval = result["evaluation"]["retrieval"]
    return (
        retrieval["aggregate"]["recall_at_5_macro"]["value"],
        retrieval["query_gold_pair_recall_at_5"]["value"],
    )


def gold_pairs_at_5(evaluation: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in evaluation["retrieval"]["per_query"]:
        for entry_id, rank in row["gold_ranks"].items():
            if rank is not None and rank <= 5:
                pairs.add((row["query_id"], entry_id))
    return pairs


def build_report(
    queries: list[dict[str, Any]],
    experiment: dict[str, Any],
    embedding_results: list[dict[str, Any]],
    k: int = 3,
) -> dict[str, Any]:
    content_documents = load_content_documents()
    full_documents = load_documents()
    content_bm25_rankings = ranked_ids_from_bm25(queries, content_documents)
    content_bm25_positive_rankings = positive_ranked_ids_from_bm25(
        queries, content_documents
    )
    mixed_bm25_rankings = ranked_ids_from_bm25(queries, full_documents)
    content_bm25 = evaluate_rankings(queries, content_bm25_rankings, k)
    content_bm25_positive = evaluate_rankings(
        queries, content_bm25_positive_rankings, k
    )
    mixed_bm25 = evaluate_rankings(queries, mixed_bm25_rankings, k)

    selected = max(embedding_results, key=selection_key)
    selected_rankings = selected["rankings"]
    hybrid_weights = experiment["hybrid"].get("weights", [1.0, 1.0])
    hybrid_rankings = {
        query["query_id"]: reciprocal_rank_fusion(
            [mixed_bm25_rankings[query["query_id"]], selected_rankings[query["query_id"]]],
            rrf_k=experiment["hybrid"]["rrf_k"],
            weights=hybrid_weights,
        )
        for query in queries
    }
    hybrid = evaluate_rankings(queries, hybrid_rankings, k)
    mixed_pairs = gold_pairs_at_5(mixed_bm25)
    embedding_pairs = gold_pairs_at_5(selected["evaluation"])
    hybrid_pairs = gold_pairs_at_5(hybrid)
    for result in embedding_results:
        result.pop("rankings", None)

    return {
        "schema_version": "1.0.0",
        "experiment": experiment,
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                "sentence-transformers": package_version("sentence-transformers"),
                "transformers": package_version("transformers"),
                "torch": package_version("torch"),
                "numpy": package_version("numpy"),
            },
            "device": "cpu",
            "similarity": "cosine via normalized embeddings and deterministic float64 einsum",
        },
        "population": {
            "split": "dev",
            "queries": len(queries),
            "single_turn_queries": sum(
                not query.get("excluded_from_single_turn", False) for query in queries
            ),
            "documents": len(content_documents),
            "test_queries_evaluated_or_encoded": False,
        },
        "baselines": {
            "content_only_bm25_forced_topk": content_bm25,
            "content_only_bm25_positive_score_only": content_bm25_positive,
            "metadata_plus_content_bm25": mixed_bm25,
        },
        "embedding_candidates": embedding_results,
        "selection": {
            "selected_model_id": selected["model"]["model_id"],
            "criterion": experiment["selection"],
        },
        "hybrid_selected_embedding_rrf": {
            "model_id": selected["model"]["model_id"],
            "rrf_k": experiment["hybrid"]["rrf_k"],
            "weights": hybrid_weights,
            "evaluation": hybrid,
        },
        "channel_complementarity_at_5": {
            "metadata_plus_content_bm25_pairs": len(mixed_pairs),
            "content_embedding_pairs": len(embedding_pairs),
            "intersection_pairs": len(mixed_pairs & embedding_pairs),
            "bm25_only_pairs": len(mixed_pairs - embedding_pairs),
            "embedding_only_pairs": len(embedding_pairs - mixed_pairs),
            "union_pairs": len(mixed_pairs | embedding_pairs),
            "fixed_rrf_pairs": len(hybrid_pairs),
            "total_gold_pairs": sum(len(query["gold_entry_ids"]) for query in queries),
        },
        "notes": [
            "Content-only BM25 and dense retrieval use only entry.content and no entity or dimension filters.",
            "Forced Top-K retains zero-score BM25 ties; positive-score-only isolates observed lexical signal.",
            "Every approved knowledge entry is one chunk; no secondary splitting is applied.",
            "Per-plant metrics are diagnostic only because dev plant counts are too small for model comparison.",
            "The lexical-overlap count 4/91 and semantic Top-K pair recall are different metrics and are not placed on one numeric scale.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        validate_run_scope(args.split, args.confirm_test, args.models)
    except ValueError as exc:
        parser.error(str(exc))
    if args.k < 1 or args.batch_size < 1:
        parser.error("--k and --batch-size must be positive")

    experiment = load_json(args.experiment)
    manifest = load_json(args.split_manifest)
    allowed_models = {
        model_spec["model_id"]: model_spec
        for model_spec in experiment["embedding_candidates"]
    }
    model_ids = args.models or list(allowed_models)
    unknown = [model_id for model_id in model_ids if model_id not in allowed_models]
    if unknown:
        parser.error(f"models not registered before the run: {unknown}")

    all_queries = load_json(MAIN_PATH)
    included_ids = query_ids_for_split(manifest, args.split)
    queries = [query for query in all_queries if query["query_id"] in included_ids]
    if len(queries) != manifest["populations"][args.split]["n"]:
        raise SystemExit("split manifest and query data disagree")

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "sentence-transformers is required; install requirements-rerank.txt"
        ) from exc

    content_documents = load_content_documents()
    embedding_results: list[dict[str, Any]] = []
    for model_id in model_ids:
        spec = allowed_models[model_id]
        model = SentenceTransformer(model_id, device="cpu")
        revision = model_revision(model)
        expected_revision = spec.get("model_revision")
        if expected_revision and revision != expected_revision:
            raise SystemExit(
                f"model revision mismatch for {model_id}: "
                f"expected {expected_revision}, got {revision}"
            )
        rankings = sentence_transformer_ranker(model, spec, args.batch_size)(
            queries, content_documents
        )
        embedding_results.append(
            {
                "model": {
                    "model_id": model_id,
                    "model_revision": revision,
                    "model_card": spec["model_card"],
                    "model_card_accessed": "2026-08-11",
                    "query_prefix": spec["query_prefix"],
                    "passage_prefix": spec["passage_prefix"],
                    "normalize_embeddings": spec["normalize_embeddings"],
                },
                "evaluation": evaluate_rankings(queries, rankings, args.k),
                "rankings": rankings,
            }
        )
        del model
        gc.collect()

    report = build_report(queries, experiment, embedding_results, k=args.k)
    report["population"]["split"] = args.split
    report["population"]["test_queries_evaluated_or_encoded"] = args.split == "test"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Semantic retrieval report: {args.output}")
    for name, result in report["baselines"].items():
        retrieval = result["retrieval"]
        print(
            f"{name}: R@5={retrieval['aggregate']['recall_at_5_macro']['value']:.3f} "
            f"R@|gold|={retrieval['aggregate']['recall_at_gold_macro']['value']:.3f} "
            f"pairs@5={retrieval['query_gold_pair_recall_at_5']['hits']}/"
            f"{retrieval['query_gold_pair_recall_at_5']['n']}"
        )
    for result in report["embedding_candidates"]:
        retrieval = result["evaluation"]["retrieval"]
        print(
            f"{result['model']['model_id']}: "
            f"R@5={retrieval['aggregate']['recall_at_5_macro']['value']:.3f} "
            f"R@|gold|={retrieval['aggregate']['recall_at_gold_macro']['value']:.3f} "
            f"pairs@5={retrieval['query_gold_pair_recall_at_5']['hits']}/"
            f"{retrieval['query_gold_pair_recall_at_5']['n']}"
        )
    hybrid = report["hybrid_selected_embedding_rrf"]
    retrieval = hybrid["evaluation"]["retrieval"]
    print(
        f"hybrid({hybrid['model_id']}): "
        f"R@5={retrieval['aggregate']['recall_at_5_macro']['value']:.3f} "
        f"R@|gold|={retrieval['aggregate']['recall_at_gold_macro']['value']:.3f} "
        f"pairs@5={retrieval['query_gold_pair_recall_at_5']['hits']}/"
        f"{retrieval['query_gold_pair_recall_at_5']['n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
