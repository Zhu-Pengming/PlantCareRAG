"""Runtime query engine for the dataset-backed PlantCareRAG v1 corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dataset_v1.scripts.evaluate_baseline import BM25, classify_dimension


ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = ROOT / "data" / "processed"
REPORT_ROOT = ROOT / "reports"
MANIFEST_PATH = ROOT / "config" / "datasets.json"

UNSUPPORTED_DIMENSION_RULES = (
    ("pet_safety", ("toxic", "poison", "cat", "dog", "pet safe", "safe for pets")),
    ("disease_or_symptom", ("disease", "yellow", "brown", "wilt", "rot", "spot", "dying")),
    ("pest", ("pest", "aphid", "mite", "gnat", "whitefl")),
    ("temperature", ("temperature", "degrees", "too cold", "too hot")),
    ("humidity", ("humidity", "humid", "mist")),
    ("taxonomy", ("scientific name", "botanical name", "species", "family")),
)
DISCOVERY_CUES = (
    "which plant",
    "which plants",
    "what plant",
    "what plants",
    "recommend",
    "show me",
    "list plants",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def mask_entity(query: str, entity_name: str | None) -> str:
    if not entity_name:
        return query
    return re.sub(re.escape(entity_name), " ", query, flags=re.IGNORECASE)


def detect_unsupported_dimension(masked_query: str) -> str | None:
    text = normalized_text(masked_query)
    for dimension, phrases in UNSUPPORTED_DIMENSION_RULES:
        if any(contains_phrase(text, phrase) for phrase in phrases):
            return dimension
    return None


def contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def contains_entity_name(text: str, normalized_name: str) -> bool:
    return (
        re.search(
            rf"(?<!\w){re.escape(normalized_name)}(?!\w)",
            text,
        )
        is not None
    )


class QueryEngine:
    """Answer supported care questions and refuse unsupported or conflicted ones."""

    def __init__(
        self,
        processed_root: Path = PROCESSED_ROOT,
        report_root: Path = REPORT_ROOT,
    ) -> None:
        self.plants = load_json(processed_root / "plants.json")
        self.entries = load_json(processed_root / "entries.json")
        self.excluded = load_json(report_root / "excluded_conflicts.json")
        manifest = load_json(MANIFEST_PATH)
        self.care_source = next(
            item
            for item in manifest["datasets"]
            if item["role"] == "knowledge_base_seed"
        )
        self.entry_by_key = {
            (entry["plant_id"], entry["dimension"]): entry for entry in self.entries
        }
        self.accepted_entities = [
            {
                "name": plant["name"],
                "normalized_name": plant["normalized_name"],
                "plant_id": plant["id"],
                "decision": "accepted",
                "record": plant,
            }
            for plant in self.plants
        ]
        self.excluded_entities = [
            {
                "name": item["plant_name"],
                "normalized_name": item["plant_name_normalized"],
                "plant_id": None,
                "decision": "excluded_conflict",
                "record": item,
            }
            for item in self.excluded
        ]
        self.entities = self.accepted_entities + self.excluded_entities
        documents = [
            {
                **entry,
                "index_text": (
                    f"{entry['plant_name']} {entry['dimension']} "
                    f"{entry['value']} {entry['content']}"
                ),
            }
            for entry in self.entries
        ]
        self.documents = documents
        self.bm25 = BM25(documents)

    def link_entity(self, query: str) -> dict | None:
        query_text = normalized_text(query)
        matches = [
            entity
            for entity in self.entities
            if contains_entity_name(query_text, entity["normalized_name"])
        ]
        if not matches:
            return None
        matches.sort(
            key=lambda entity: (
                -len(entity["normalized_name"]),
                entity["decision"] != "accepted",
                entity["normalized_name"],
            )
        )
        return matches[0]

    def citation(self, entry: dict) -> dict:
        source = entry["source"]
        return {
            "dataset_id": source["dataset_id"],
            "dataset_page": source["page_url"],
            "license": source["license"],
            "raw_row_numbers": source["raw_row_numbers"],
        }

    def result_item(self, entry: dict, score: float | None = None) -> dict:
        result = {
            "entry_id": entry["id"],
            "plant_id": entry["plant_id"],
            "plant_name": entry["plant_name"],
            "dimension": entry["dimension"],
            "value": entry["value"],
            "content": entry["content"],
            "citation": self.citation(entry),
            "evidence_level": entry["evidence_level"],
        }
        if score is not None:
            result["score"] = round(score, 6)
        return result

    def response(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> dict:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20")
        query = " ".join(query.split())
        base = {
            "schema_version": "1.0.0",
            "query": query,
            "status": "refused",
            "reason_code": None,
            "linked_entity": None,
            "dimension": None,
            "answer": None,
            "results": [],
            "warnings": [
                "Dataset claims are not independently verified horticultural guidance."
            ],
        }
        if not query:
            base["reason_code"] = "empty_query"
            base["answer"] = "Please provide a plant-care question."
            return base

        entity = self.link_entity(query)
        if entity:
            base["linked_entity"] = {
                "name": entity["name"],
                "plant_id": entity["plant_id"],
                "decision": entity["decision"],
            }
        masked = mask_entity(query, entity["name"] if entity else None)
        unsupported = detect_unsupported_dimension(masked)
        if unsupported:
            base["reason_code"] = "unsupported_dimension"
            base["dimension"] = unsupported
            base["answer"] = (
                f"This dataset-backed v1 has no verified {unsupported} evidence, "
                "so it will not answer this question."
            )
            return base

        dimension = classify_dimension(masked)
        base["dimension"] = dimension
        if not dimension:
            base["reason_code"] = "ambiguous_intent"
            base["answer"] = (
                "Ask about growth, soil, lighting, watering, or fertilizer."
            )
            return base

        if entity and entity["decision"] == "excluded_conflict":
            conflict = entity["record"]
            base["reason_code"] = "conflicted_entity"
            base["answer"] = (
                f"{entity['name']} is quarantined because the source dataset "
                "contains conflicting care values."
            )
            base["conflict"] = {
                "fields": conflict["conflicts"],
                "raw_row_numbers": conflict["raw_row_numbers"],
                "dataset_page": self.care_source["page_url"],
            }
            return base

        if entity:
            entry = self.entry_by_key.get((entity["plant_id"], dimension))
            if entry is None:
                base["reason_code"] = "missing_entry"
                base["answer"] = "The linked plant has no entry for this dimension."
                return base
            base["status"] = "answered"
            base["reason_code"] = "supported_entity_care"
            base["answer"] = entry["content"]
            base["results"] = [self.result_item(entry)]
            return base

        if not any(cue in normalized_text(query) for cue in DISCOVERY_CUES):
            base["reason_code"] = "unknown_entity"
            base["answer"] = (
                "No accepted plant entity was found. Name a plant from the dataset, "
                "or ask a discovery question such as 'Which plants need indirect light?'"
            )
            return base

        candidate_indices = [
            index
            for index, document in enumerate(self.documents)
            if document["dimension"] == dimension
        ]
        ranked = self.bm25.score(query, candidate_indices)
        positive = [(score, index) for score, index in ranked if score > 0][:top_k]
        if not positive:
            base["reason_code"] = "no_positive_match"
            base["answer"] = "No positive lexical match was found in this dimension."
            return base
        base["status"] = "answered"
        base["reason_code"] = "supported_discovery"
        base["answer"] = f"Found {len(positive)} dataset-backed matches."
        base["results"] = [
            self.result_item(self.documents[index], score) for score, index in positive
        ]
        return base
