#!/usr/bin/env python3
"""Run the single pre-registered weighted-RRF sweep on the dev split only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.create_evaluation_split import query_ids_for_split
    from scripts.evaluate_baseline import EVAL_ROOT, MAIN_PATH, load_documents
    from scripts.evaluate_semantic_retrieval import (
        evaluate_rankings,
        gold_pairs_at_5,
        load_content_documents,
        model_revision,
        ranked_ids_from_bm25,
        reciprocal_rank_fusion,
        sentence_transformer_ranker,
    )
    from scripts.validate_kb import load_json
except ModuleNotFoundError:  # Support direct execution
    from create_evaluation_split import query_ids_for_split
    from evaluate_baseline import EVAL_ROOT, MAIN_PATH, load_documents
    from evaluate_semantic_retrieval import (
        evaluate_rankings,
        gold_pairs_at_5,
        load_content_documents,
        model_revision,
        ranked_ids_from_bm25,
        reciprocal_rank_fusion,
        sentence_transformer_ranker,
    )
    from validate_kb import load_json


DEFAULT_EXPERIMENT = EVAL_ROOT / "experiments" / "fusion_weight_sweep_v1.json"
DEFAULT_OUTPUT = EVAL_ROOT / "results" / "fusion_weight_sweep_dev.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value: str) -> Path:
    return EVAL_ROOT.parent / value


def selection_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    retrieval = candidate["evaluation"]["retrieval"]
    weight = candidate["embedding_weight"]
    return (
        retrieval["query_gold_pair_recall_at_5"]["hits"],
        retrieval["aggregate"]["recall_at_5_macro"]["value"],
        retrieval["aggregate"]["recall_at_gold_macro"]["value"],
        -abs(weight - 1.0),
        -weight,
    )


def build_sweep_report(
    queries: list[dict[str, Any]],
    experiment: dict[str, Any],
    model_spec: dict[str, Any],
    model_info: dict[str, Any],
    embedding_rankings: dict[str, list[str]],
    k: int = 3,
) -> dict[str, Any]:
    mixed_rankings = ranked_ids_from_bm25(queries, load_documents())
    mixed_evaluation = evaluate_rankings(queries, mixed_rankings, k)
    embedding_evaluation = evaluate_rankings(queries, embedding_rankings, k)
    rrf_k = experiment["frozen_inputs"]["rrf_k"]
    bm25_weight = experiment["only_tunable_parameter"]["bm25_weight"]
    candidates: list[dict[str, Any]] = []

    for embedding_weight in experiment["only_tunable_parameter"][
        "embedding_weight_grid"
    ]:
        rankings = {
            query["query_id"]: reciprocal_rank_fusion(
                [
                    mixed_rankings[query["query_id"]],
                    embedding_rankings[query["query_id"]],
                ],
                rrf_k=rrf_k,
                weights=[bm25_weight, embedding_weight],
            )
            for query in queries
        }
        candidates.append(
            {
                "bm25_weight": bm25_weight,
                "embedding_weight": embedding_weight,
                "rrf_k": rrf_k,
                "evaluation": evaluate_rankings(queries, rankings, k),
            }
        )

    best = max(candidates, key=selection_key)
    threshold = 28
    best_hits = best["evaluation"]["retrieval"][
        "query_gold_pair_recall_at_5"
    ]["hits"]
    stop_condition_met = best_hits <= threshold
    final_weight = 1.0 if stop_condition_met else best["embedding_weight"]
    final_candidate = next(
        candidate
        for candidate in candidates
        if candidate["embedding_weight"] == final_weight
    )

    mixed_pairs = gold_pairs_at_5(mixed_evaluation)
    embedding_pairs = gold_pairs_at_5(embedding_evaluation)
    final_pairs = gold_pairs_at_5(final_candidate["evaluation"])
    return {
        "schema_version": "1.0.0",
        "experiment": experiment,
        "population": {
            "split": "dev",
            "queries": len(queries),
            "gold_pairs": sum(len(query["gold_entry_ids"]) for query in queries),
            "documents": len(load_content_documents()),
            "test_queries_evaluated_or_encoded": False,
        },
        "frozen_model": model_info,
        "channel_baselines": {
            "metadata_plus_content_bm25": mixed_evaluation,
            "content_only_embedding": embedding_evaluation,
        },
        "weight_candidates": candidates,
        "selection_result": {
            "best_observed_embedding_weight": best["embedding_weight"],
            "best_observed_pair_hits_at_5": best_hits,
            "threshold_pair_hits_at_5": threshold,
            "stop_condition_met": stop_condition_met,
            "decision": (
                "accept_original_equal_weight_rrf"
                if stop_condition_met
                else "freeze_selected_weighted_rrf"
            ),
            "final_bm25_weight": bm25_weight,
            "final_embedding_weight": final_weight,
            "final_candidate": final_candidate,
        },
        "pair_complementarity_at_5": {
            "bm25_pairs": len(mixed_pairs),
            "embedding_pairs": len(embedding_pairs),
            "intersection_pairs": len(mixed_pairs & embedding_pairs),
            "union_pairs": len(mixed_pairs | embedding_pairs),
            "final_rrf_pairs": len(final_pairs),
        },
        "notes": [
            "This script has no test-split execution path.",
            "The five weights, selection rule, and stopping rule were registered before this run.",
            "No model, RRF k, tokenizer, candidate-set, or reranker change is part of this sweep.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 1 or args.k < 1:
        parser.error("--batch-size and --k must be positive")

    experiment = load_json(args.experiment)
    if experiment["development_scope"]["split"] != "dev":
        parser.error("fusion weight sweep must be dev-only")
    parent_result = project_path(experiment["parent_dev_result"])
    if file_sha256(parent_result) != experiment["parent_dev_result_sha256"]:
        parser.error("parent dev result checksum differs from pre-registration")
    split_path = project_path(experiment["split_manifest"])
    if file_sha256(split_path) != experiment["split_manifest_sha256"]:
        parser.error("split manifest checksum differs from pre-registration")

    parent_experiment = load_json(project_path(experiment["parent_experiment"]))
    model_id = experiment["frozen_inputs"]["model_id"]
    model_spec = next(
        spec
        for spec in parent_experiment["embedding_candidates"]
        if spec["model_id"] == model_id
    )
    manifest = load_json(split_path)
    dev_ids = query_ids_for_split(manifest, "dev")
    queries = [
        query for query in load_json(MAIN_PATH) if query["query_id"] in dev_ids
    ]
    if len(queries) != experiment["development_scope"]["queries"]:
        parser.error("dev population differs from pre-registration")

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "sentence-transformers is required; install requirements-rerank.txt"
        ) from exc

    model = SentenceTransformer(model_id, device="cpu")
    revision = model_revision(model)
    expected_revision = experiment["frozen_inputs"]["model_revision"]
    if revision != expected_revision:
        raise SystemExit(
            f"model revision mismatch: expected {expected_revision}, got {revision}"
        )
    content_documents = load_content_documents()
    embedding_rankings = sentence_transformer_ranker(
        model, model_spec, args.batch_size
    )(queries, content_documents)
    model_info = {
        "model_id": model_id,
        "model_revision": revision,
        "query_prefix": model_spec["query_prefix"],
        "passage_prefix": model_spec["passage_prefix"],
        "normalize_embeddings": model_spec["normalize_embeddings"],
    }
    del model
    gc.collect()

    report = build_sweep_report(
        queries,
        experiment,
        model_spec,
        model_info,
        embedding_rankings,
        k=args.k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Fusion weight sweep: {args.output}")
    for candidate in report["weight_candidates"]:
        retrieval = candidate["evaluation"]["retrieval"]
        pairs = retrieval["query_gold_pair_recall_at_5"]
        print(
            f"embedding_weight={candidate['embedding_weight']:.2f}: "
            f"pairs@5={pairs['hits']}/{pairs['n']} "
            f"R@5={retrieval['aggregate']['recall_at_5_macro']['value']:.3f} "
            f"R@|gold|={retrieval['aggregate']['recall_at_gold_macro']['value']:.3f}"
        )
    selection = report["selection_result"]
    print(
        f"decision={selection['decision']} "
        f"final_embedding_weight={selection['final_embedding_weight']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
