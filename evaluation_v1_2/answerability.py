"""Deterministic v1.2 answerability gate built on the frozen v1 corpus.

This module deliberately does not generate prose with an LLM. It establishes
the response and evidence contracts that a later retriever/generator must obey.
"""

from __future__ import annotations

import re
from typing import Any

from dataset_v1.query_engine import (
    DISCOVERY_CUES,
    QueryEngine,
    detect_unsupported_dimension,
    mask_entity,
    normalized_text,
)


DIMENSION_RULES = (
    ("watering", ("water", "watering", "thirsty", "dry soil", "wet soil")),
    ("fertilizer", ("fertilizer", "fertilize", "plant food", "feed", "feeding")),
    ("soil", ("soil", "potting medium", "potting mix")),
    ("lighting", ("light", "sun", "shade", "window", "place")),
    ("growth", ("grow", "growth", "quickly", "fast")),
)
QUALIFIER_RULES = {
    "season": {
        "winter": ("winter",),
        "spring": ("spring",),
        "summer": ("summer",),
        "autumn": ("autumn", "fall"),
    },
}
VAGUE_ENTITY_PHRASES = (
    "my plant",
    "this plant",
    "the plant",
    "it ",
    "it's ",
    "its ",
)


def contains_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def classify_dimensions(masked_query: str) -> list[str]:
    """Return every supported care dimension expressed by the query."""
    text = normalized_text(masked_query)
    return [
        dimension
        for dimension, phrases in DIMENSION_RULES
        if any(contains_phrase(text, phrase) for phrase in phrases)
    ]


def extract_qualifiers(masked_query: str) -> dict[str, str]:
    """Extract qualifiers that require evidence more specific than v1 stores."""
    text = normalized_text(masked_query)
    qualifiers: dict[str, str] = {}
    for qualifier_type, values in QUALIFIER_RULES.items():
        for value, phrases in values.items():
            if any(contains_phrase(text, phrase) for phrase in phrases):
                qualifiers[qualifier_type] = value
                break
    return qualifiers


class AnswerabilityEngine:
    """Classify answerability and attach every returned claim to evidence."""

    def __init__(self, query_engine: QueryEngine | None = None) -> None:
        self.v1 = query_engine or QueryEngine()

    @staticmethod
    def _base(query: str) -> dict[str, Any]:
        return {
            "schema_version": "1.2.0",
            "query": query,
            "status": "clarification_needed",
            "reason_code": "unclassified",
            "linked_entity": None,
            "dimensions": [],
            "qualifiers": {},
            "answer": "",
            "claims": [],
            "evidence": [],
            "limitations": [],
        }

    def response(self, query: str, *, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20")

        query = " ".join(query.split())
        result = self._base(query)
        if not query:
            result.update(
                reason_code="empty_query",
                answer="Please provide a plant-care question.",
            )
            return result

        entity = self.v1.link_entity(query)
        if entity:
            result["linked_entity"] = {
                "name": entity["name"],
                "plant_id": entity["plant_id"],
                "decision": entity["decision"],
            }
        masked = mask_entity(query, entity["name"] if entity else None)
        dimensions = classify_dimensions(masked)
        qualifiers = extract_qualifiers(masked)
        result["dimensions"] = dimensions
        result["qualifiers"] = qualifiers

        padded_query = f"{normalized_text(query)} "
        if entity is None and any(phrase in padded_query for phrase in VAGUE_ENTITY_PHRASES):
            result.update(
                status="clarification_needed",
                reason_code="missing_entity_context",
                answer="Which plant are you asking about?",
                limitations=["The plant entity is not present in this single-turn query."],
            )
            return result

        unsupported = detect_unsupported_dimension(masked)
        if unsupported:
            result["dimensions"] = [unsupported]
            result.update(
                status="unsupported",
                reason_code="unsupported_dimension",
                answer=(
                    f"The current evidence corpus has no verified {unsupported} "
                    "evidence, so this question cannot be answered safely."
                ),
                limitations=[f"Unsupported evidence dimension: {unsupported}."],
            )
            return result

        if entity and entity["decision"] == "excluded_conflict":
            conflict = entity["record"]
            result.update(
                status="conflicting_evidence",
                reason_code="source_row_conflict",
                answer=(
                    f"{entity['name']} is quarantined because the source dataset "
                    "contains conflicting care values."
                ),
                limitations=[
                    "No authority policy can resolve conflicts inside the single seed dataset."
                ],
                conflict={
                    "fields": conflict["conflicts"],
                    "raw_row_numbers": conflict["raw_row_numbers"],
                    "dataset_page": self.v1.care_source["page_url"],
                },
            )
            return result

        if not dimensions:
            result.update(
                status="clarification_needed",
                reason_code="missing_care_dimension",
                answer="Ask about growth, soil, lighting, watering, or fertilizer.",
            )
            return result

        if entity is None:
            if any(cue in normalized_text(query) for cue in DISCOVERY_CUES) and len(dimensions) == 1:
                v1_response = self.v1.response(query, top_k=top_k)
                if v1_response["status"] == "answered":
                    return self._from_v1_discovery(result, v1_response)
            result.update(
                status="clarification_needed",
                reason_code="unknown_entity",
                answer="Name a plant from the accepted dataset before asking for care advice.",
            )
            return result

        if qualifiers:
            qualifier_text = ", ".join(
                f"{key}={value}" for key, value in sorted(qualifiers.items())
            )
            result.update(
                status="insufficient_evidence",
                reason_code="missing_qualified_evidence",
                answer=(
                    "The corpus has general care evidence but no reviewed evidence "
                    f"for the requested qualifier ({qualifier_text})."
                ),
                limitations=[
                    "Generic evidence must not be promoted into a qualified recommendation."
                ],
            )
            return result

        entries = []
        for dimension in dimensions:
            entry = self.v1.entry_by_key.get((entity["plant_id"], dimension))
            if entry is None:
                result.update(
                    status="insufficient_evidence",
                    reason_code="missing_dimension_evidence",
                    answer=f"No evidence is available for {dimension}.",
                    limitations=[f"Missing plant/dimension pair: {dimension}."],
                )
                return result
            entries.append(entry)

        evidence = [self.v1.result_item(entry) for entry in entries]
        claims = [
            {"text": entry["content"], "evidence_ids": [entry["id"]]}
            for entry in entries
        ]
        result.update(
            status="answered",
            reason_code=(
                "supported_multi_dimension_care"
                if len(dimensions) > 1
                else "supported_entity_care"
            ),
            answer=" ".join(entry["content"] for entry in entries),
            claims=claims,
            evidence=evidence,
            limitations=[
                "Evidence is a single-dataset claim and has not been independently verified."
            ],
        )
        return result

    @staticmethod
    def _from_v1_discovery(result: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
        evidence = response["results"]
        claims = [
            {"text": item["content"], "evidence_ids": [item["entry_id"]]}
            for item in evidence
        ]
        result.update(
            status="answered",
            reason_code="supported_discovery",
            dimensions=[response["dimension"]],
            answer=response["answer"],
            claims=claims,
            evidence=evidence,
            limitations=["Discovery ranking is lexical and based on a single dataset."],
        )
        return result
