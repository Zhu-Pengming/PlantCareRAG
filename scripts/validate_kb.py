#!/usr/bin/env python3
"""Validate the Plant RAG knowledge base without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
KB_ROOT = ROOT / "knowledge_base"
DATA_ROOT = KB_ROOT / "data"
SCHEMA_VERSION = "1.0.0"

DIMENSIONS = {
    "taxonomy",
    "plant_attribute",
    "lighting",
    "watering",
    "humidity",
    "temperature",
    "soil",
    "fertilizer",
    "repotting",
    "symptom",
    "disease",
    "pest",
    "pet_safety",
}
CLAIM_TYPES = {"fact", "recommendation", "diagnostic_association", "safety"}
REVIEW_STATUSES = {"draft", "approved", "needs_review"}
INTENTS = {"identify", "learn", "care", "diagnose", "safety"}
ANSWER_SHAPES = {
    "identity",
    "numeric_request",
    "conditional",
    "yes_no",
    "diagnosis_list",
    "recommendation",
}
SOURCE_TYPES = {
    "taxonomy_database",
    "university_extension",
    "botanical_garden",
    "veterinary_reference",
    "veterinary_poison_database",
    "horticultural_guidance",
}
ID_PATTERNS = {
    "plant": re.compile(r"^plant:[a-z0-9_]+$"),
    "source": re.compile(r"^source:[a-z0-9_]+$"),
    "entry": re.compile(r"^kb:[a-z0-9_]+$"),
    "alias": re.compile(r"^alias:[a-z0-9_]+$"),
}


class ValidationFailure(Exception):
    """Raised when a JSON file cannot be loaded."""


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{path}: {exc}") from exc


def is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def is_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def require_fields(item: dict[str, Any], required: set[str], label: str) -> list[str]:
    missing = sorted(required - item.keys())
    return [f"{label}: missing field '{field}'" for field in missing]


def validate_list_file(value: Any, label: str) -> list[str]:
    return [] if isinstance(value, list) else [f"{label}: top-level JSON value must be an array"]


def validate_schema_version(item: dict[str, Any], label: str) -> list[str]:
    value = item.get("schema_version")
    if value != SCHEMA_VERSION:
        return [f"{label}: schema_version must be '{SCHEMA_VERSION}', got '{value}'"]
    return []


def validate_sources(sources: Any) -> list[str]:
    errors = validate_list_file(sources, "sources.json")
    if errors:
        return errors

    required = {
        "schema_version",
        "id",
        "organization",
        "title",
        "url",
        "source_type",
        "authority_tier",
        "accessed_at",
        "copyright_status",
        "usage",
    }
    seen: set[str] = set()
    for index, source in enumerate(sources):
        label = f"sources.json[{index}]"
        if not isinstance(source, dict):
            errors.append(f"{label}: expected an object")
            continue
        errors.extend(require_fields(source, required, label))
        errors.extend(validate_schema_version(source, label))
        source_id = source.get("id")
        if not isinstance(source_id, str) or not ID_PATTERNS["source"].fullmatch(source_id):
            errors.append(f"{label}: invalid source id '{source_id}'")
        elif source_id in seen:
            errors.append(f"{label}: duplicate source id '{source_id}'")
        else:
            seen.add(source_id)
        if not is_http_url(source.get("url")):
            errors.append(f"{label}: url must be an absolute HTTP(S) URL")
        if not is_iso_date(source.get("accessed_at")):
            errors.append(f"{label}: accessed_at must be an ISO date")
        if source.get("authority_tier") not in {1, 2, 3}:
            errors.append(f"{label}: authority_tier must be 1, 2, or 3")
        if source.get("source_type") not in SOURCE_TYPES:
            errors.append(f"{label}: invalid source_type '{source.get('source_type')}'")
    return errors


def validate_plants(plants: Any, source_ids: set[str]) -> list[str]:
    errors = validate_list_file(plants, "plants.json")
    if errors:
        return errors

    required = {
        "schema_version",
        "id",
        "scientific_name",
        "rank",
        "taxonomy",
        "names",
        "synonyms",
        "identity_source_ids",
        "review_status",
        "updated_at",
    }
    taxonomy_fields = {"kingdom", "phylum", "class", "order", "family", "genus"}
    seen: set[str] = set()
    for index, plant in enumerate(plants):
        label = f"plants.json[{index}]"
        if not isinstance(plant, dict):
            errors.append(f"{label}: expected an object")
            continue
        errors.extend(require_fields(plant, required, label))
        errors.extend(validate_schema_version(plant, label))
        plant_id = plant.get("id")
        if not isinstance(plant_id, str) or not ID_PATTERNS["plant"].fullmatch(plant_id):
            errors.append(f"{label}: invalid plant id '{plant_id}'")
        elif plant_id in seen:
            errors.append(f"{label}: duplicate plant id '{plant_id}'")
        else:
            seen.add(plant_id)
        taxonomy = plant.get("taxonomy")
        if not isinstance(taxonomy, dict):
            errors.append(f"{label}: taxonomy must be an object")
        else:
            errors.extend(require_fields(taxonomy, taxonomy_fields, f"{label}.taxonomy"))
        names = plant.get("names")
        if not isinstance(names, list) or not names:
            errors.append(f"{label}: names must be a non-empty array")
        refs = plant.get("identity_source_ids")
        if not isinstance(refs, list) or not refs:
            errors.append(f"{label}: identity_source_ids must be a non-empty array")
        else:
            for source_id in refs:
                if source_id not in source_ids:
                    errors.append(f"{label}: unknown identity source '{source_id}'")
        if plant.get("review_status") not in REVIEW_STATUSES:
            errors.append(f"{label}: invalid review_status")
        if not is_iso_date(plant.get("updated_at")):
            errors.append(f"{label}: updated_at must be an ISO date")
    return errors


def validate_aliases(aliases: Any, plant_ids: set[str], source_ids: set[str]) -> list[str]:
    errors = validate_list_file(aliases, "entity_aliases.json")
    if errors:
        return errors

    required = {
        "schema_version",
        "id",
        "value",
        "normalized_value",
        "language",
        "plant_id",
        "alias_type",
        "automatic",
        "provenance",
        "source_ids",
        "review_status",
    }
    seen_ids: set[str] = set()
    automatic_values: set[str] = set()
    for index, alias in enumerate(aliases):
        label = f"entity_aliases.json[{index}]"
        if not isinstance(alias, dict):
            errors.append(f"{label}: expected an object")
            continue
        errors.extend(require_fields(alias, required, label))
        errors.extend(validate_schema_version(alias, label))
        alias_id = alias.get("id")
        if not isinstance(alias_id, str) or not ID_PATTERNS["alias"].fullmatch(alias_id):
            errors.append(f"{label}: invalid alias id '{alias_id}'")
        elif alias_id in seen_ids:
            errors.append(f"{label}: duplicate alias id '{alias_id}'")
        else:
            seen_ids.add(alias_id)
        value = alias.get("value")
        normalized = alias.get("normalized_value")
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: value must be a non-empty string")
        if not isinstance(normalized, str) or not normalized.strip():
            errors.append(f"{label}: normalized_value must be a non-empty string")
        plant_id = alias.get("plant_id")
        if plant_id is not None and plant_id not in plant_ids:
            errors.append(f"{label}: unknown plant_id '{plant_id}'")
        refs = alias.get("source_ids")
        if not isinstance(refs, list):
            errors.append(f"{label}: source_ids must be an array")
        else:
            for source_id in refs:
                if source_id not in source_ids:
                    errors.append(f"{label}: unknown source '{source_id}'")
        if alias.get("provenance") != "project_curated" and not refs:
            errors.append(f"{label}: non-project aliases require at least one source")
        if alias.get("language") == "zh-CN" and alias.get("evaluation_status") != "untested":
            errors.append(
                f"{label}: Chinese aliases must be marked untested in the English-only evaluation scope"
            )
        if alias.get("automatic") is True:
            if plant_id is None:
                errors.append(f"{label}: automatic aliases require a plant_id")
            if alias.get("review_status") != "approved":
                errors.append(f"{label}: automatic aliases must be approved")
            if isinstance(normalized, str) and normalized in automatic_values:
                errors.append(f"{label}: duplicate automatic normalized_value '{normalized}'")
            elif isinstance(normalized, str):
                automatic_values.add(normalized)
        if alias.get("alias_type") == "ambiguous" and alias.get("automatic") is not False:
            errors.append(f"{label}: ambiguous aliases cannot be automatic")
    return errors


def validate_entries(
    entries: list[tuple[Path, Any]],
    plant_ids: set[str],
    source_map: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "id",
        "plant_id",
        "language",
        "dimension",
        "topics",
        "claim_type",
        "content",
        "conditions",
        "retrieval",
        "source_refs",
        "review_status",
        "updated_at",
    }
    seen: set[str] = set()
    for path, payload in entries:
        if not isinstance(payload, list):
            errors.append(f"{path}: top-level JSON value must be an array")
            continue
        for index, entry in enumerate(payload):
            label = f"{path.name}[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{label}: expected an object")
                continue
            errors.extend(require_fields(entry, required, label))
            errors.extend(validate_schema_version(entry, label))
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or not ID_PATTERNS["entry"].fullmatch(entry_id):
                errors.append(f"{label}: invalid entry id '{entry_id}'")
            elif entry_id in seen:
                errors.append(f"{label}: duplicate entry id '{entry_id}'")
            else:
                seen.add(entry_id)
            if entry.get("plant_id") not in plant_ids:
                errors.append(f"{label}: unknown plant_id '{entry.get('plant_id')}'")
            if entry.get("language") != "zh-CN":
                errors.append(f"{label}: language must be 'zh-CN' in v0.1")
            if entry.get("dimension") not in DIMENSIONS:
                errors.append(f"{label}: invalid dimension '{entry.get('dimension')}'")
            if entry.get("claim_type") not in CLAIM_TYPES:
                errors.append(f"{label}: invalid claim_type '{entry.get('claim_type')}'")
            answer_shape = entry.get("answer_shape")
            if answer_shape is not None and answer_shape not in ANSWER_SHAPES:
                errors.append(f"{label}: invalid answer_shape '{answer_shape}'")
            topics = entry.get("topics")
            if not isinstance(topics, list) or not topics or not all(isinstance(topic, str) and topic for topic in topics):
                errors.append(f"{label}: topics must be a non-empty string array")
            content = entry.get("content")
            if not isinstance(content, str) or not 8 <= len(content) <= 400:
                errors.append(f"{label}: content length must be between 8 and 400 characters")
            if not isinstance(entry.get("conditions"), list):
                errors.append(f"{label}: conditions must be an array")
            retrieval = entry.get("retrieval")
            if not isinstance(retrieval, dict):
                errors.append(f"{label}: retrieval must be an object")
            else:
                for field in ("aliases", "intents", "symptoms"):
                    if not isinstance(retrieval.get(field), list):
                        errors.append(f"{label}: retrieval.{field} must be an array")
                intents = retrieval.get("intents", [])
                if isinstance(intents, list) and any(intent not in INTENTS for intent in intents):
                    errors.append(f"{label}: retrieval.intents contains an unsupported intent")
            refs = entry.get("source_refs")
            if not isinstance(refs, list) or not refs:
                errors.append(f"{label}: source_refs must be a non-empty array")
            else:
                if entry.get("dimension") == "pet_safety" and len(refs) < 2:
                    errors.append(f"{label}: pet_safety entries require at least two source_refs")
                for ref_index, ref in enumerate(refs):
                    if not isinstance(ref, dict):
                        errors.append(f"{label}.source_refs[{ref_index}]: expected an object")
                        continue
                    source_id = ref.get("source_id")
                    if source_id not in source_map:
                        errors.append(f"{label}: unknown source '{source_id}'")
                    elif not is_http_url(source_map[source_id].get("url")):
                        errors.append(f"{label}: source '{source_id}' does not resolve to a valid URL")
                    locator = ref.get("locator")
                    if not isinstance(locator, str) or len(locator.strip()) < 4:
                        errors.append(f"{label}: every source_ref needs a specific locator")
            if entry.get("review_status") not in REVIEW_STATUSES:
                errors.append(f"{label}: invalid review_status")
            if not is_iso_date(entry.get("updated_at")):
                errors.append(f"{label}: updated_at must be an ISO date")
    return errors


def validate_kb(data_root: Path = DATA_ROOT) -> tuple[list[str], dict[str, Any]]:
    try:
        sources = load_json(data_root / "sources.json")
        plants = load_json(data_root / "plants.json")
        aliases = load_json(data_root / "entity_aliases.json")
        entry_files = sorted((data_root / "entries").glob("*.json"))
        entries = [(path, load_json(path)) for path in entry_files]
    except ValidationFailure as exc:
        return [str(exc)], {}

    source_ids = {
        source.get("id") for source in sources if isinstance(source, dict) and isinstance(source.get("id"), str)
    } if isinstance(sources, list) else set()
    source_map = {
        source["id"]: source
        for source in sources
        if isinstance(source, dict) and isinstance(source.get("id"), str)
    } if isinstance(sources, list) else {}
    plant_ids = {
        plant.get("id") for plant in plants if isinstance(plant, dict) and isinstance(plant.get("id"), str)
    } if isinstance(plants, list) else set()

    errors = []
    errors.extend(validate_sources(sources))
    errors.extend(validate_plants(plants, source_ids))
    errors.extend(validate_aliases(aliases, plant_ids, source_ids))
    errors.extend(validate_entries(entries, plant_ids, source_map))

    flat_entries = [entry for _, payload in entries if isinstance(payload, list) for entry in payload if isinstance(entry, dict)]
    stats = {
        "sources": len(sources) if isinstance(sources, list) else 0,
        "plants": len(plants) if isinstance(plants, list) else 0,
        "aliases": len(aliases) if isinstance(aliases, list) else 0,
        "entries": len(flat_entries),
        "dimensions": dict(sorted(Counter(entry.get("dimension") for entry in flat_entries).items())),
    }
    return errors, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Path containing plants.json, sources.json and entries/")
    args = parser.parse_args()
    errors, stats = validate_kb(args.data_root)
    if errors:
        print(f"Knowledge base validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Knowledge base is valid.")
    print(f"Sources: {stats['sources']}")
    print(f"Plants: {stats['plants']}")
    print(f"Aliases: {stats['aliases']}")
    print(f"Entries: {stats['entries']}")
    print("Dimensions:")
    for dimension, count in stats["dimensions"].items():
        print(f"- {dimension}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
