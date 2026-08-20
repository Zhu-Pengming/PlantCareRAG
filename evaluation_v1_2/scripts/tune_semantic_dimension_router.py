#!/usr/bin/env python3
"""Tune one semantic-fallback threshold on synthetic dev only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_v1.query_engine import QueryEngine, detect_unsupported_dimension, mask_entity
from evaluation_v1_2.answerability import classify_dimensions
from evaluation_v1_2.semantic_dimension_router import SemanticDimensionRouter


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "synthetic_benchmark.json"
OUTPUT = ROOT / "config" / "semantic_dimension_router.json"
GRID = [round(value / 100, 2) for value in range(15, 81)]
RESOLVED_MODEL_REPO = "qdrant/bge-small-en-v1.5-onnx-q"
RESOLVED_MODEL_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"


def main() -> int:
    queries = [
        row
        for row in json.loads(BENCHMARK.read_text(encoding="utf-8"))
        if int(row["split_hash"], 16) % 2 == 0
    ]
    linker = QueryEngine()
    probe = SemanticDimensionRouter(threshold=0.0)
    cached = []
    for query in queries:
        entity = linker.link_entity(query["raw_text"])
        masked = mask_entity(query["raw_text"], entity["name"] if entity else None)
        unsupported = detect_unsupported_dimension(masked)
        lexical = classify_dimensions(masked)
        scores = probe.semantic_scores(masked) if not lexical and not unsupported else {}
        cached.append((query, unsupported, lexical, scores))

    trials = []
    for threshold in GRID:
        exact = 0
        fallback_count = 0
        for query, unsupported, lexical, scores in cached:
            if unsupported:
                predicted = [unsupported]
            elif lexical:
                predicted = lexical
            elif scores:
                fallback_count += 1
                dimension, score = max(scores.items(), key=lambda item: item[1])
                predicted = [dimension] if score >= threshold else []
            else:
                predicted = []
            exact += set(predicted) == set(query["expected_dimensions"])
        trials.append(
            {
                "threshold": threshold,
                "dimension_exact_match": exact / len(queries),
                "exact_count": exact,
                "n": len(queries),
                "semantic_fallback_candidates": fallback_count,
            }
        )
    best = max(trials, key=lambda row: (row["dimension_exact_match"], row["threshold"]))
    result = {
        "schema_version": "1.0.0",
        "model": probe.model_name,
        "resolved_model_repo": RESOLVED_MODEL_REPO,
        "resolved_model_revision": RESOLVED_MODEL_REVISION,
        "fastembed_version": "0.8.0",
        "selection_split": "dev",
        "selection_metric": "dimension_exact_match",
        "tie_break": "highest_threshold",
        "threshold": best["threshold"],
        "dev_result": best,
        "threshold_grid": GRID,
        "test_accessed": False,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
