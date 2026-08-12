#!/usr/bin/env python3
"""Validate the rebuilt dataset-v1 corpus and benchmark contracts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "config" / "datasets.json"
DIMENSIONS = {"growth", "soil", "lighting", "watering", "fertilizer"}
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
PLANT_ID_RE = re.compile(r"^plant:[a-z0-9_]+$")
ENTRY_ID_RE = re.compile(r"^kb:[a-z0-9_]+:(growth|soil|lighting|watering|fertilizer)$")


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
    build_stats = load(ROOT / "reports" / "build_stats.json")

    for name, payload in (
        ("plants", plants),
        ("entries", entries),
        ("queries", queries),
        ("contexts", contexts),
        ("excluded conflicts", excluded),
    ):
        if not isinstance(payload, list):
            return [f"{name} must be a top-level array"], {}

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
        for field in ("archive_sha256", "member_sha256"):
            if not SHA256_RE.fullmatch(item.get(field, "")):
                errors.append(f"dataset has invalid {field}: {item['id']}")
        if not item.get("page_url", "").startswith("https://"):
            errors.append(f"dataset page URL must use HTTPS: {item['id']}")

    plant_map = {}
    normalized_names = set()
    for plant in plants:
        plant_id = plant.get("id")
        if plant_id in plant_map:
            errors.append(f"duplicate plant id: {plant_id}")
        plant_map[plant_id] = plant
        if not isinstance(plant_id, str) or not PLANT_ID_RE.fullmatch(plant_id):
            errors.append(f"invalid plant id: {plant_id}")
        if plant.get("schema_version") != "1.0.0":
            errors.append(f"invalid plant schema_version: {plant_id}")
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
        if not isinstance(entry_id, str) or not ENTRY_ID_RE.fullmatch(entry_id):
            errors.append(f"invalid entry id: {entry_id}")
        if entry.get("schema_version") != "1.0.0":
            errors.append(f"invalid entry schema_version: {entry_id}")
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
        if entry.get("source", {}).get("license") != "CC BY 4.0":
            errors.append(f"entry lacks expected license: {entry_id}")
        if not isinstance(entry.get("value"), str) or not entry["value"]:
            errors.append(f"entry has empty value: {entry_id}")
        if not isinstance(entry.get("content"), str) or len(entry["content"]) < 8:
            errors.append(f"entry has invalid content: {entry_id}")
    for plant_id in plant_map:
        plant_entries = entries_by_plant.get(plant_id, [])
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
        if query.get("schema_version") != "1.0.0":
            errors.append(f"invalid query schema_version: {query_id}")
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

    context_ids = set()
    context_source_rows = set()
    for context in contexts:
        context_id = context.get("id")
        if context_id in context_ids:
            errors.append(f"duplicate context id: {context_id}")
        context_ids.add(context_id)
        if context.get("schema_version") != "1.0.0":
            errors.append(f"invalid context schema_version: {context_id}")
        pest = context.get("pest_presence")
        severity = context.get("pest_severity")
        if (pest == "None") != (severity == "None"):
            errors.append(f"context has contradictory pest fields: {context.get('id')}")
        if context.get("allowed_use") != "context_stress_test_only":
            errors.append(f"context has unsafe allowed_use: {context.get('id')}")
        if context.get("source", {}).get("dataset_id") not in accepted_sources:
            errors.append(f"context uses unaccepted source: {context.get('id')}")
        if context.get("source", {}).get("license") != "CC BY 4.0":
            errors.append(f"context lacks expected license: {context_id}")
        source_row = context.get("source", {}).get("raw_row_number")
        if source_row in context_source_rows:
            errors.append(f"duplicate context source row: {source_row}")
        context_source_rows.add(source_row)
        for field in ("humidity_percent", "soil_moisture_percent"):
            value = context.get(field)
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                errors.append(f"context {field} out of range: {context_id}")
        if context.get("health_score") not in {1, 2, 3, 4, 5}:
            errors.append(f"context health_score out of range: {context_id}")
        temperature = context.get("room_temperature_c")
        if not isinstance(temperature, (int, float)) or not -20 <= temperature <= 60:
            errors.append(f"context temperature out of range: {context_id}")

    excluded_names = {item["plant_name_normalized"] for item in excluded}
    for item in excluded:
        if not item.get("conflicts"):
            errors.append(
                f"excluded entity lacks conflict evidence: {item.get('plant_name')}"
            )
        if not item.get("raw_row_numbers"):
            errors.append(
                f"excluded entity lacks raw row locators: {item.get('plant_name')}"
            )
    overlap = normalized_names & excluded_names
    if overlap:
        errors.append(f"conflicted plants leaked into KB: {sorted(overlap)[:5]}")

    expected_stats = build_stats.get("care_corpus", {})
    actual_counts = {
        "accepted_plants": len(plants),
        "entries": len(entries),
        "queries": len(queries),
        "excluded_conflicted_plants": len(excluded),
    }
    for field, actual in actual_counts.items():
        if expected_stats.get(field) != actual:
            errors.append(
                f"build_stats mismatch for {field}: {expected_stats.get(field)} != {actual}"
            )
    if (
        build_stats.get("context_stress_data", {}).get("accepted_context_rows")
        != len(contexts)
    ):
        errors.append("build_stats mismatch for accepted_context_rows")

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
