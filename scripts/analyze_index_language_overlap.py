#!/usr/bin/env python3
"""Audit whether English evaluation queries overlap KB content or index metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_baseline import DATA_ROOT, EVAL_ROOT, MAIN_PATH, flatten_strings, tokenize
    from scripts.validate_kb import load_json
except ModuleNotFoundError:  # Support direct execution
    from evaluate_baseline import DATA_ROOT, EVAL_ROOT, MAIN_PATH, flatten_strings, tokenize
    from validate_kb import load_json


DEFAULT_REPORT = EVAL_ROOT / "results" / "index_language_overlap.json"
METADATA_FIELDS = ("dimension", "topics", "conditions", "retrieval")


def ratio(count: int, total: int) -> dict[str, Any]:
    return {"count": count, "n": total, "rate": round(count / total, 6) if total else 0.0}


def build_overlap_report() -> dict[str, Any]:
    queries = load_json(MAIN_PATH)
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted((DATA_ROOT / "entries").glob("*.json")):
        for entry in load_json(path):
            entries[entry["id"]] = entry

    pairs: list[dict[str, Any]] = []
    queries_with_content_overlap: set[str] = set()
    queries_with_metadata_overlap: set[str] = set()
    for query in queries:
        query_tokens = set(tokenize(query["raw_text"]))
        for entry_id in query["gold_entry_ids"]:
            entry = entries[entry_id]
            content_tokens = set(tokenize(entry["content"]))
            metadata_payload = {field: entry[field] for field in METADATA_FIELDS}
            metadata_tokens = set(tokenize(" ".join(flatten_strings(metadata_payload))))
            content_overlap = sorted(query_tokens & content_tokens)
            metadata_overlap = sorted(query_tokens & metadata_tokens)
            if content_overlap:
                queries_with_content_overlap.add(query["query_id"])
            if metadata_overlap:
                queries_with_metadata_overlap.add(query["query_id"])
            pairs.append(
                {
                    "query_id": query["query_id"],
                    "entry_id": entry_id,
                    "content_overlap_tokens": content_overlap,
                    "metadata_overlap_tokens": metadata_overlap,
                }
            )

    total = len(pairs)
    content_any = sum(bool(pair["content_overlap_tokens"]) for pair in pairs)
    metadata_any = sum(bool(pair["metadata_overlap_tokens"]) for pair in pairs)
    both = sum(
        bool(pair["content_overlap_tokens"]) and bool(pair["metadata_overlap_tokens"])
        for pair in pairs
    )
    content_only = sum(
        bool(pair["content_overlap_tokens"]) and not pair["metadata_overlap_tokens"]
        for pair in pairs
    )
    metadata_only = sum(
        bool(pair["metadata_overlap_tokens"]) and not pair["content_overlap_tokens"]
        for pair in pairs
    )
    neither = total - both - content_only - metadata_only
    return {
        "schema_version": "1.0.0",
        "population": {
            "queries": len(queries),
            "query_gold_pairs": total,
            "query_language": "en",
            "content_language": "zh-CN",
        },
        "index_partition": {
            "content_fields": ["content"],
            "metadata_fields": list(METADATA_FIELDS),
        },
        "pair_overlap": {
            "content_any": ratio(content_any, total),
            "metadata_any": ratio(metadata_any, total),
            "metadata_only": ratio(metadata_only, total),
            "content_only": ratio(content_only, total),
            "both": ratio(both, total),
            "neither": ratio(neither, total),
            "content_overlap_token_count": sum(
                len(pair["content_overlap_tokens"]) for pair in pairs
            ),
            "metadata_overlap_token_count": sum(
                len(pair["metadata_overlap_tokens"]) for pair in pairs
            ),
        },
        "query_coverage": {
            "with_any_gold_content_overlap": ratio(
                len(queries_with_content_overlap), len(queries)
            ),
            "with_any_gold_metadata_overlap": ratio(
                len(queries_with_metadata_overlap), len(queries)
            ),
        },
        "per_pair": pairs,
        "notes": [
            "Overlap uses the exact tokenizer from the BM25 baseline, including its English stemming behavior.",
            "Counts are unique overlap tokens within each query-gold pair, then summed across pairs.",
            "This is a descriptive lexical audit, not an attribution of BM25 score or a causal language-ablation experiment.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_overlap_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pair_overlap = report["pair_overlap"]
    print(f"Index-language overlap report: {args.output}")
    print(
        f"Pairs={report['population']['query_gold_pairs']} "
        f"metadata_any={pair_overlap['metadata_any']['count']} "
        f"content_any={pair_overlap['content_any']['count']} "
        f"metadata_only={pair_overlap['metadata_only']['count']} "
        f"content_only={pair_overlap['content_only']['count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
