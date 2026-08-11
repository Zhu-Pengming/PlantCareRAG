#!/usr/bin/env python3
"""Validate the rebuilt dataset-v1 corpus and benchmark contracts."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "config" / "datasets.json"
DIMENSIONS = {"growth", "soil", "lighting", "watering", "fertilizer"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path = PROCESSED_ROOT) -> tuple[list[str], dict]:
    errors: list[str] = []
    manifest = load(MANIFEST_PATH)
    plants = load(root / "plants.json")
    entries = load(root / "entries.json")
    queries = load(root / "queries.json")
    contexts = load(root / "context_samples.json")
    excluded = load(ROOT / "reports" / "excluded_conflicts.json")

    accepted_sources = {
        item["id"]
        for item in manifest["datasets"]
        if item["decision"] != "rejected"
    }
    rejected_sources = {
        item["id"]
        for item in manifest["datasets"]
        if item["decision"] == "rejected"
    }
    for item in manifest["datasets"]:
        if item["decision"] != "rejected" and item["license"] == "Unknown":
            errors.append(f"accepted dataset has unknown license: {item['id']}")

    plant_map = {}
    normalized_names = set()
    for plant in plants:
        plant_id = plant.get("id")
        if plant_id in plant_map:
            errors.append(f"duplicate plant id: {plant_id}")
        plant_map[plant_id] = plant
        normalized = plant.get("normalized_name")
        if normalized in normalized_names:
            errors.append(f"duplicate normalized plant name: {normalized}")
        normalized_names.add(normalized)
        if plant.get("source_dataset_id") not in accepted_sources:
            errors.append(f"plant uses unaccepted source: {plant_id}")
        if plant.get("review_status") != "dataset_accepted":
            errors.append(f"plant lacks dataset_accepted status: {plant_id}")

    entry_map = {}
    entries_by_plant = defaultdict(list)
    for entry in entries:
        entry_id = entry.get("id")
        if entry_id in entry_map:
            errors.append(f"duplicate entry id: {entry_id}")
        entry_map[entry_id] = entry
        plant_id = entry.get("plant_id")
        entries_by_plant[plant_id].append(entry)
        if plant_id not in plant_map:
            errors.append(f"entry has unknown plant: {entry_id}")
        if entry.get("dimension") not in DIMENSIONS:
            errors.append(f"entry has invalid dimension: {entry_id}")
        source_id = entry.get("source", {}).get("dataset_id")
        if source_id not in accepted_sources or source_id in rejected_sources:
            errors.append(f"entry uses rejected or unknown source: {entry_id}")
        if not entry.get("source", {}).get("raw_row_numbers"):
            errors.append(f"entry lacks raw row locator: {entry_id}")
    for plant_id, plant_entries in entries_by_plant.items():
        dimensions = Counter(entry["dimension"] for entry in plant_entries)
        if set(dimensions) != DIMENSIONS or any(count != 1 for count in dimensions.values()):
            errors.append(f"plant does not have exactly one entry per dimension: {plant_id}")

    query_ids = set()
    ordered_hashes = sorted(query["split_hash"] for query in queries)
    dev_count = len(queries) // 2
    cutoff = set(ordered_hashes[:dev_count])
    for query in queries:
        query_id = query.get("id")
        if query_id in query_ids:
            errors.append(f"duplicate query id: {query_id}")
        query_ids.add(query_id)
        expected_hash = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
        if query.get("split_hash") != expected_hash:
            errors.append(f"query hash mismatch: {query_id}")
        expected_split = "dev" if expected_hash in cutoff else "test"
        if query.get("split") != expected_split:
            errors.append(f"query split mismatch: {query_id}")
        gold = entry_map.get(query.get("gold_entry_id"))
        if gold is None:
            errors.append(f"query has unknown gold entry: {query_id}")
        elif gold["plant_id"] != query.get("plant_id") or gold["dimension"] != query.get("expected_dimension"):
            errors.append(f"query gold contract mismatch: {query_id}")
        if query.get("benchmark_type") != "template_contract_test":
            errors.append(f"query benchmark type is not explicit: {query_id}")

    for context in contexts:
        pest = context.get("pest_presence")
        severity = context.get("pest_severity")
        if (pest == "None") != (severity == "None"):
            errors.append(f"context has contradictory pest fields: {context.get('id')}")
        if context.get("allowed_use") != "context_stress_test_only":
            errors.append(f"context has unsafe allowed_use: {context.get('id')}")
        if context.get("source", {}).get("dataset_id") not in accepted_sources:
            errors.append(f"context uses unaccepted source: {context.get('id')}")

    excluded_names = {item["plant_name_normalized"] for item in excluded}
    overlap = normalized_names & excluded_names
    if overlap:
        errors.append(f"conflicted plants leaked into KB: {sorted(overlap)[:5]}")

    stats = {
        "plants": len(plants),
        "entries": len(entries),
        "queries": len(queries),
        "contexts": len(contexts),
        "excluded_conflicted_plants": len(excluded),
    }
    return errors, stats


def main() -> int:
    errors, stats = validate()
    if errors:
        print(f"dataset-v1 validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("dataset-v1 is valid")
    for key, value in stats.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
