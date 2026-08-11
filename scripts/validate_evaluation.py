#!/usr/bin/env python3
"""Validate retrieval and abstention evaluation datasets."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from scripts.validate_kb import DATA_ROOT, DIMENSIONS, ROOT, SCHEMA_VERSION, is_http_url, load_json
except ModuleNotFoundError:  # Support direct execution: python3 scripts/validate_evaluation.py
    from validate_kb import DATA_ROOT, DIMENSIONS, ROOT, SCHEMA_VERSION, is_http_url, load_json


EVALUATION_ROOT = ROOT / "evaluation" / "data"
MAIN_PATH = EVALUATION_ROOT / "main_queries.json"
ABSTAIN_PATH = EVALUATION_ROOT / "abstain_queries.json"
EXTRA_DIMENSIONS = {"propagation", "plant_support"}
ANSWER_SHAPES = {"identity", "numeric_request", "conditional", "yes_no", "diagnosis_list", "recommendation"}
ANSWER_SHAPE_SUPPORT = {"supported", "conditional_substitute", "unsupported"}
QUESTION_TYPES = {"direct", "alias_or_coreference", "symptom", "compound", "judgment_or_premise"}


def load_kb_indexes() -> tuple[set[str], dict[str, str]]:
    plants = load_json(DATA_ROOT / "plants.json")
    plant_ids = {plant["id"] for plant in plants}
    entry_to_plant: dict[str, str] = {}
    for path in sorted((DATA_ROOT / "entries").glob("*.json")):
        for entry in load_json(path):
            entry_to_plant[entry["id"]] = entry["plant_id"]
    return plant_ids, entry_to_plant


def validate_common(query: Any, label: str, expected_id_prefix: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(query, dict):
        return [f"{label}: expected an object"]
    if query.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{label}: schema_version must be '{SCHEMA_VERSION}'")
    query_id = query.get("query_id")
    if not isinstance(query_id, str) or not query_id.startswith(expected_id_prefix):
        errors.append(f"{label}: invalid query_id '{query_id}'")
    raw_text = query.get("raw_text")
    if not isinstance(raw_text, str) or len(raw_text.strip()) < 3:
        errors.append(f"{label}: raw_text must be a non-empty user utterance")
    if not is_http_url(query.get("url")):
        errors.append(f"{label}: url must be an absolute HTTP(S) URL")
    if query.get("platform") not in {"reddit", "zhihu", "tieba"}:
        errors.append(f"{label}: unsupported platform")
    if query.get("source_location") not in {"post_title", "post_body", "comment"}:
        errors.append(f"{label}: invalid source_location")
    dimensions = query.get("expected_dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        errors.append(f"{label}: expected_dimensions must be a non-empty array")
    elif any(dimension not in DIMENSIONS | EXTRA_DIMENSIONS for dimension in dimensions):
        errors.append(f"{label}: expected_dimensions contains an unsupported value")
    return errors


def validate_main(queries: Any, plant_ids: set[str], entry_to_plant: dict[str, str]) -> list[str]:
    if not isinstance(queries, list):
        return ["main_queries.json: top-level JSON value must be an array"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, query in enumerate(queries):
        label = f"main_queries.json[{index}]"
        errors.extend(validate_common(query, label, "q:main:"))
        if not isinstance(query, dict):
            continue
        query_id = query.get("query_id")
        if query_id in seen:
            errors.append(f"{label}: duplicate query_id '{query_id}'")
        elif isinstance(query_id, str):
            seen.add(query_id)
        plant_id = query.get("plant_id")
        if plant_id is not None and plant_id not in plant_ids:
            errors.append(f"{label}: unknown plant_id '{plant_id}'")
        gold = query.get("gold_entry_ids")
        if not isinstance(gold, list):
            errors.append(f"{label}: gold_entry_ids must be an array")
            gold = []
        for entry_id in gold:
            if entry_id not in entry_to_plant:
                errors.append(f"{label}: unknown gold entry '{entry_id}'")
            elif plant_id is not None and entry_to_plant[entry_id] != plant_id:
                errors.append(f"{label}: gold entry '{entry_id}' belongs to another plant")
        status = query.get("status")
        if status not in {"covered", "partial", "gap"}:
            errors.append(f"{label}: invalid status '{status}'")
        if status == "gap" and gold:
            errors.append(f"{label}: gap queries cannot have gold entries")
        if status in {"covered", "partial"} and not gold:
            errors.append(f"{label}: {status} queries require at least one gold entry")
        if query.get("answer_shape") not in ANSWER_SHAPES:
            errors.append(f"{label}: invalid answer_shape")
        answer_shape_support = query.get("answer_shape_support")
        if answer_shape_support is not None and answer_shape_support not in ANSWER_SHAPE_SUPPORT:
            errors.append(f"{label}: invalid answer_shape_support")
        if answer_shape_support == "conditional_substitute" and query.get("answer_shape") != "numeric_request":
            errors.append(f"{label}: conditional_substitute requires numeric_request")
        if answer_shape_support == "conditional_substitute" and status != "partial":
            errors.append(f"{label}: conditional_substitute queries must be partial")
        if query.get("question_type") not in QUESTION_TYPES:
            errors.append(f"{label}: invalid question_type")
        if query.get("context_requirement") not in {"none", "image", "thread_context"}:
            errors.append(f"{label}: invalid context_requirement")
        expected_exclusion = query.get("context_requirement") == "thread_context"
        actual_exclusion = query.get("excluded_from_single_turn", False)
        if actual_exclusion is not expected_exclusion:
            errors.append(
                f"{label}: excluded_from_single_turn must be true exactly when "
                "context_requirement is thread_context"
            )
    return errors


def validate_abstain(queries: Any, plant_ids: set[str]) -> list[str]:
    if not isinstance(queries, list):
        return ["abstain_queries.json: top-level JSON value must be an array"]
    errors: list[str] = []
    seen: set[str] = set()
    reasons = {"out_of_scope_plant", "missing_dimension", "insufficient_evidence", "image_required"}
    for index, query in enumerate(queries):
        label = f"abstain_queries.json[{index}]"
        errors.extend(validate_common(query, label, "q:abstain:"))
        if not isinstance(query, dict):
            continue
        query_id = query.get("query_id")
        if query_id in seen:
            errors.append(f"{label}: duplicate query_id '{query_id}'")
        elif isinstance(query_id, str):
            seen.add(query_id)
        plant_id = query.get("plant_id")
        if plant_id is not None and plant_id not in plant_ids:
            errors.append(f"{label}: unknown plant_id '{plant_id}'")
        if query.get("reason") not in reasons:
            errors.append(f"{label}: invalid abstention reason")
    return errors


def validate_evaluation() -> tuple[list[str], dict[str, Any]]:
    try:
        plant_ids, entry_to_plant = load_kb_indexes()
        main_queries = load_json(MAIN_PATH)
        abstain_queries = load_json(ABSTAIN_PATH)
    except Exception as exc:  # load_json already supplies a precise path/message
        return [str(exc)], {}

    errors = validate_main(main_queries, plant_ids, entry_to_plant)
    errors.extend(validate_abstain(abstain_queries, plant_ids))
    stats = {
        "main": len(main_queries) if isinstance(main_queries, list) else 0,
        "abstain": len(abstain_queries) if isinstance(abstain_queries, list) else 0,
        "status": dict(sorted(Counter(query.get("status") for query in main_queries).items())),
        "plants": dict(sorted(Counter(query.get("plant_id") or "ambiguous" for query in main_queries).items())),
        "platforms": dict(sorted(Counter(query.get("platform") for query in main_queries).items())),
        "languages": dict(sorted(Counter(query.get("language") for query in main_queries).items())),
        "context_requirements": dict(
            sorted(Counter(query.get("context_requirement") for query in main_queries).items())
        ),
        "single_turn_included": sum(
            not query.get("excluded_from_single_turn", False) for query in main_queries
        ),
    }
    return errors, stats


def main() -> int:
    errors, stats = validate_evaluation()
    if errors:
        print(f"Evaluation validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Evaluation datasets are valid.")
    print(f"Main queries: {stats['main']}")
    print(f"Abstain queries: {stats['abstain']}")
    for field in ("status", "plants", "platforms", "languages", "context_requirements"):
        print(f"{field.title()}: {json.dumps(stats[field], ensure_ascii=False, sort_keys=True)}")
    print(f"Single-turn included: {stats['single_turn_included']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
