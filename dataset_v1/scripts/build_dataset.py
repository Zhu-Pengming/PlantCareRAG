#!/usr/bin/env python3
"""Build a quality-gated atomic PlantCareRAG corpus from frozen datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "datasets.json"
RAW_ROOT = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
REPORT_ROOT = ROOT / "reports"

CARE_ID = "kaggle:aribashafaqat/plants-growth-and-care-recommendations"
CONTEXT_ID = "kaggle:souvikrana17/indoor-plant-health-and-growth-dataset"
CARE_FIELDS = ("Growth", "Soil", "Sunlight", "Watering", "Fertilization Type")
DIMENSION_MAP = {
    "Growth": "growth",
    "Soil": "soil",
    "Sunlight": "lighting",
    "Watering": "watering",
    "Fertilization Type": "fertilizer",
}
QUERY_TEMPLATES = {
    "growth": "How quickly does {plant} grow?",
    "soil": "What potting medium should I use for {plant}?",
    "lighting": "Where should I place {plant} for enough light?",
    "watering": "When should I water {plant}?",
    "fertilizer": "What plant food does {plant} need?",
}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def dataset_by_id(manifest: dict, dataset_id: str) -> dict:
    return next(item for item in manifest["datasets"] if item["id"] == dataset_id)


def raw_member_path(raw_root: Path, dataset: dict) -> Path:
    directory = dataset["id"].split(":", 1)[1].replace("/", "__")
    return raw_root / directory / dataset["member_name"]


def normalize_space(value: str) -> str:
    return " ".join(value.strip().split())


def normalize_value(field: str, value: str) -> str:
    value = normalize_space(value)
    if field == "Watering" and value.casefold() == "regular watering":
        return "Regular watering"
    return value


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized.casefold()).strip("_")
    if not slug:
        slug = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return slug


def load_care_rows(path: Path, encoding: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding=encoding) as handle:
        rows = []
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            cleaned = {field: normalize_value(field, value) for field, value in row.items()}
            cleaned["_row_number"] = str(row_number)
            rows.append(cleaned)
    return rows


def unique_rows(rows: Iterable[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    seen: set[tuple[str, ...]] = set()
    kept = []
    duplicates = 0
    fields = ("Plant Name",) + CARE_FIELDS
    for row in rows:
        signature = tuple(row[field] for field in fields)
        if signature in seen:
            duplicates += 1
            continue
        seen.add(signature)
        kept.append(row)
    return kept, duplicates


def content_for(plant_name: str, dimension: str, value: str) -> str:
    templates = {
        "growth": "{plant} has a {value} growth rate.",
        "soil": "{plant} is listed with {value} soil.",
        "lighting": "{plant} is listed for {value}.",
        "watering": "Watering guidance for {plant}: {value}.",
        "fertilizer": "Fertilizer guidance for {plant}: {value}.",
    }
    return templates[dimension].format(plant=plant_name, value=value)


def build_corpus(rows: list[dict[str, str]], source: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    rows, duplicate_count = unique_rows(rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_space(row["Plant Name"]).casefold()].append(row)

    accepted: list[tuple[str, dict[str, str], list[int]]] = []
    excluded = []
    conflict_fields = Counter()
    for normalized_name, group in sorted(grouped.items()):
        values = {
            field: sorted({row[field] for row in group})
            for field in CARE_FIELDS
        }
        conflicts = {field: items for field, items in values.items() if len(items) > 1}
        row_numbers = sorted(int(row["_row_number"]) for row in group)
        if conflicts:
            conflict_fields.update(conflicts.keys())
            excluded.append(
                {
                    "plant_name": group[0]["Plant Name"],
                    "plant_name_normalized": normalized_name,
                    "raw_row_numbers": row_numbers,
                    "conflicts": conflicts,
                    "decision": "excluded_from_kb",
                    "reason": "The source dataset gives conflicting values for the same normalized plant name."
                }
            )
            continue
        accepted.append(
            (
                group[0]["Plant Name"],
                {field: items[0] for field, items in values.items()},
                row_numbers,
            )
        )

    plants = []
    entries = []
    queries = []
    used_slugs: dict[str, str] = {}
    for plant_name, values, row_numbers in accepted:
        slug = slugify(plant_name)
        if slug in used_slugs and used_slugs[slug] != plant_name.casefold():
            slug = f"{slug}_{hashlib.sha256(plant_name.encode()).hexdigest()[:8]}"
        used_slugs[slug] = plant_name.casefold()
        plant_id = f"plant:{slug}"
        plants.append(
            {
                "schema_version": "1.0.0",
                "id": plant_id,
                "name": plant_name,
                "normalized_name": plant_name.casefold(),
                "source_dataset_id": source["id"],
                "source_row_numbers": row_numbers,
                "review_status": "dataset_accepted",
                "evidence_level": "single_dataset_claim"
            }
        )
        for source_field, dimension in DIMENSION_MAP.items():
            entry_id = f"kb:{slug}:{dimension}"
            value = values[source_field]
            entries.append(
                {
                    "schema_version": "1.0.0",
                    "id": entry_id,
                    "plant_id": plant_id,
                    "plant_name": plant_name,
                    "dimension": dimension,
                    "value": value,
                    "content": content_for(plant_name, dimension, value),
                    "source": {
                        "dataset_id": source["id"],
                        "page_url": source["page_url"],
                        "license": source["license"],
                        "raw_row_numbers": row_numbers
                    },
                    "review_status": "dataset_accepted",
                    "evidence_level": "single_dataset_claim"
                }
            )
            query_id = f"q:{slug}:{dimension}"
            digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
            queries.append(
                {
                    "schema_version": "1.0.0",
                    "id": query_id,
                    "text": QUERY_TEMPLATES[dimension].format(plant=plant_name),
                    "plant_id": plant_id,
                    "expected_dimension": dimension,
                    "gold_entry_id": entry_id,
                    "split_hash": digest,
                    "benchmark_type": "template_contract_test"
                }
            )

    ordered = sorted(queries, key=lambda item: item["split_hash"])
    dev_count = len(ordered) // 2
    for index, query in enumerate(ordered):
        query["split"] = "dev" if index < dev_count else "test"
    queries.sort(key=lambda item: item["id"])

    stats = {
        "source_rows": len(rows) + duplicate_count,
        "exact_duplicate_rows_removed": duplicate_count,
        "unique_normalized_plant_names": len(grouped),
        "accepted_plants": len(plants),
        "excluded_conflicted_plants": len(excluded),
        "entries": len(entries),
        "queries": len(queries),
        "dev_queries": sum(query["split"] == "dev" for query in queries),
        "test_queries": sum(query["split"] == "test" for query in queries),
        "conflict_fields": dict(sorted(conflict_fields.items())),
    }
    return plants, entries, queries, {"excluded": excluded, "stats": stats}


def build_contexts(path: Path, source: dict) -> tuple[list[dict], dict]:
    with path.open(newline="", encoding=source["encoding"]) as handle:
        rows = list(csv.DictReader(handle))
    accepted = []
    contradictions = Counter()
    for row_number, row in enumerate(rows, start=2):
        pest = normalize_space(row["Pest_Presence"])
        severity = normalize_space(row["Pest_Severity"])
        if pest == "None" and severity != "None":
            contradictions["no_pest_with_severity"] += 1
            continue
        if pest != "None" and severity == "None":
            contradictions["pest_without_severity"] += 1
            continue
        accepted.append(
            {
                "schema_version": "1.0.0",
                "id": f"context:{row_number - 1:04d}",
                "plant_scientific_name": normalize_space(row["Plant_ID"]),
                "health_note": normalize_space(row["Health_Notes"]),
                "watering_amount_ml": int(row["Watering_Amount_ml"]),
                "watering_frequency_days": int(row["Watering_Frequency_days"]),
                "sunlight_exposure": normalize_space(row["Sunlight_Exposure"]),
                "room_temperature_c": float(row["Room_Temperature_C"]),
                "humidity_percent": float(row["Humidity_%"]),
                "pest_presence": pest,
                "pest_severity": severity,
                "soil_moisture_percent": float(row["Soil_Moisture_%"]),
                "soil_type": normalize_space(row["Soil_Type"]),
                "health_score": int(row["Health_Score"]),
                "source": {
                    "dataset_id": source["id"],
                    "page_url": source["page_url"],
                    "license": source["license"],
                    "raw_row_number": row_number
                },
                "allowed_use": "context_stress_test_only",
                "prohibited_use": "ground_truth_or_medical_safety_claim"
            }
        )
    stats = {
        "source_rows": len(rows),
        "accepted_context_rows": len(accepted),
        "excluded_contradictory_rows": sum(contradictions.values()),
        "contradictions": dict(sorted(contradictions.items())),
    }
    return accepted, stats


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()

    manifest = load_manifest()
    care_source = dataset_by_id(manifest, CARE_ID)
    context_source = dataset_by_id(manifest, CONTEXT_ID)
    care_path = raw_member_path(args.raw_root, care_source)
    context_path = raw_member_path(args.raw_root, context_source)
    if not care_path.exists() or not context_path.exists():
        raise SystemExit("raw datasets are missing; run download_datasets.py first")

    care_rows = load_care_rows(care_path, care_source["encoding"])
    plants, entries, queries, corpus_report = build_corpus(care_rows, care_source)
    contexts, context_stats = build_contexts(context_path, context_source)

    write_json(args.output_root / "plants.json", plants)
    write_json(args.output_root / "entries.json", entries)
    write_json(args.output_root / "queries.json", queries)
    write_json(args.output_root / "context_samples.json", contexts)
    write_json(args.report_root / "excluded_conflicts.json", corpus_report["excluded"])
    write_json(
        args.report_root / "build_stats.json",
        {
            "schema_version": "1.0.0",
            "built_from_manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "care_corpus": corpus_report["stats"],
            "context_stress_data": context_stats,
        },
    )
    print(json.dumps({"care_corpus": corpus_report["stats"], "context_stress_data": context_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
