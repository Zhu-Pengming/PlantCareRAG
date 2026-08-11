#!/usr/bin/env python3
"""Create a deterministic label-blind dev/test split from evaluation query IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_baseline import EVAL_ROOT, MAIN_PATH
    from scripts.validate_kb import load_json
except ModuleNotFoundError:  # Support direct execution
    from evaluate_baseline import EVAL_ROOT, MAIN_PATH
    from validate_kb import load_json


DEFAULT_OUTPUT = EVAL_ROOT / "splits" / "main_sha256_20_20_v1.json"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(queries: list[dict[str, Any]]) -> dict[str, Any]:
    def counts(field: str, null_label: str = "null") -> dict[str, int]:
        return dict(
            sorted(Counter(query.get(field) or null_label for query in queries).items())
        )

    return {
        "question_type": counts("question_type"),
        "plant_id": counts("plant_id"),
        "context_requirement": counts("context_requirement"),
        "status": counts("status"),
        "gold_size": dict(
            sorted(Counter(str(len(query["gold_entry_ids"])) for query in queries).items())
        ),
        "excluded_from_single_turn": {
            "true": sum(query.get("excluded_from_single_turn", False) for query in queries),
            "false": sum(
                not query.get("excluded_from_single_turn", False) for query in queries
            ),
        },
    }


def build_split_manifest(
    queries: list[dict[str, Any]], dev_size: int = 20
) -> dict[str, Any]:
    if not 0 < dev_size < len(queries):
        raise ValueError("dev_size must leave at least one query in both splits")
    query_ids = [query["query_id"] for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique before splitting")

    assignments = sorted(
        (
            {"query_id": query_id, "sha256": sha256_text(query_id)}
            for query_id in query_ids
        ),
        key=lambda item: (item["sha256"], item["query_id"]),
    )
    for index, assignment in enumerate(assignments):
        assignment["split"] = "dev" if index < dev_size else "test"

    query_by_id = {query["query_id"]: query for query in queries}
    populations: dict[str, Any] = {}
    for split in ("dev", "test"):
        members = [item["query_id"] for item in assignments if item["split"] == split]
        split_queries = [query_by_id[query_id] for query_id in members]
        populations[split] = {
            "n": len(members),
            "query_ids": members,
            "distribution": distribution(split_queries),
        }

    sorted_ids = sorted(query_ids)
    return {
        "schema_version": "1.0.0",
        "split_id": "main_sha256_20_20_v1",
        "source": {
            "path": "evaluation/data/main_queries.json",
            "query_count": len(queries),
            "source_file_sha256": file_sha256(MAIN_PATH),
            "sorted_query_ids_sha256": sha256_text("\n".join(sorted_ids) + "\n"),
        },
        "algorithm": {
            "hash": "sha256",
            "input": "UTF-8 query_id with no salt",
            "ordering": "ascending hexadecimal digest; query_id is the deterministic tie-breaker",
            "assignment": f"first {dev_size} records are dev; remaining records are test",
            "uses_query_text": False,
            "uses_labels_or_metrics": False,
        },
        "populations": populations,
        "assignments_by_hash": assignments,
        "notes": [
            "The split is deterministic and label-blind; no stratification or manual swaps are applied.",
            "Distribution differences are reported, not repaired after observing labels.",
            "Use dev for model and hyperparameter choices; run the frozen test split once per finalized experiment.",
        ],
    }


def query_ids_for_split(manifest: dict[str, Any], split: str) -> set[str]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    return set(manifest["populations"][split]["query_ids"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_split_manifest(load_json(MAIN_PATH), dev_size=args.dev_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Evaluation split: {args.output}")
    for split in ("dev", "test"):
        population = manifest["populations"][split]
        print(
            f"{split}: N={population['n']} "
            f"question_type={json.dumps(population['distribution']['question_type'], sort_keys=True)} "
            f"thread_context={population['distribution']['context_requirement'].get('thread_context', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
