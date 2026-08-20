"""Citation-first PlantCareRAG answer pipeline over verified evidence.

The default generator is deliberately extractive: every factual sentence is a
reviewed claim and carries its claim ID. This provides an auditable generation
floor before an optional LLM paraphraser is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from dataset_v1.query_engine import (
    QueryEngine,
    detect_unsupported_dimension,
    mask_entity,
)
from evaluation_v1_2.answerability import classify_dimensions, extract_qualifiers
from evaluation_v1_2.evidence_store import VerifiedEvidenceStore


ROOT = Path(__file__).resolve().parent
SEMANTIC_CONFIG = ROOT / "config" / "semantic_dimension_router.json"
UNRESOLVED_COMMON_NAME_MAPPINGS = frozenset(
    {"plant:monstera", "plant:hoya_wax_plant"}
)


@dataclass(frozen=True)
class RouteDecision:
    dimensions: tuple[str, ...]
    route: str
    scores: dict[str, float] | None = None


class DimensionRouter(Protocol):
    def route(self, query: str, entity_name: str | None) -> RouteDecision: ...


class LexicalDimensionRouter:
    def route(self, query: str, entity_name: str | None) -> RouteDecision:
        masked = mask_entity(query, entity_name)
        return RouteDecision(tuple(classify_dimensions(masked)), "lexical")


class LiveSemanticDimensionRouter:
    """Lexical-first router with the frozen dense fallback configuration."""

    def __init__(self) -> None:
        from evaluation_v1_2.semantic_dimension_router import SemanticDimensionRouter

        config = json.loads(SEMANTIC_CONFIG.read_text(encoding="utf-8"))
        self._router = SemanticDimensionRouter(
            threshold=config["threshold"],
            model_name=config["model"],
        )

    def route(self, query: str, entity_name: str | None) -> RouteDecision:
        dimensions, metadata = self._router.predict(query, entity_name)
        return RouteDecision(
            tuple(dimensions),
            metadata["route"],
            metadata.get("semantic_scores"),
        )


class ExtractiveGroundedGenerator:
    """Render evidence without introducing unsupported factual language."""

    @staticmethod
    def generate(evidence: list[dict]) -> tuple[str, list[dict]]:
        claims = [
            {
                "text": record["claim_text"],
                "evidence_ids": [record["claim_id"]],
            }
            for record in evidence
        ]
        answer = "\n".join(
            f"- {record['claim_text']} [{record['claim_id']}]"
            for record in evidence
        )
        return answer, claims


class GroundedRAGEngine:
    """Route, retrieve verified claims, and return a citation-bound answer."""

    def __init__(
        self,
        *,
        router: DimensionRouter | None = None,
        query_engine: QueryEngine | None = None,
        evidence_store: VerifiedEvidenceStore | None = None,
    ) -> None:
        self.linker = query_engine or QueryEngine()
        self.store = evidence_store or VerifiedEvidenceStore()
        self.router = router or LexicalDimensionRouter()
        self.generator = ExtractiveGroundedGenerator()

    @staticmethod
    def _base(query: str) -> dict:
        return {
            "schema_version": "1.3.0",
            "query": query,
            "status": "clarification_needed",
            "reason_code": "unclassified",
            "linked_entity": None,
            "dimensions": [],
            "qualifiers": {},
            "route": "none",
            "answer": "",
            "claims": [],
            "evidence": [],
            "limitations": [],
        }

    @staticmethod
    def _evidence_item(record: dict) -> dict:
        locator = record["locator"]
        return {
            "claim_id": record["claim_id"],
            "plant_id": record["plant_id"],
            "dimension": record["dimension"],
            "qualifiers": record.get("qualifiers", {}),
            "claim_text": record["claim_text"],
            "review_status": record["review_status"],
            "citation": {
                "source_id": record["source_id"],
                "source_title": record["source_title"],
                "source_type": record["source_type"],
                "url": locator["url"],
                "section": locator["section"],
                "paragraph": locator["paragraph"],
                "accessed_at": record["accessed_at"],
            },
        }

    def response(
        self,
        query: str,
        *,
        route_decision: RouteDecision | None = None,
    ) -> dict:
        query = " ".join(query.split())
        result = self._base(query)
        if not query:
            result.update(reason_code="empty_query", answer="Please provide a plant-care question.")
            return result

        entity = self.linker.link_entity(query)
        if entity:
            result["linked_entity"] = {
                "name": entity["name"],
                "plant_id": entity["plant_id"],
                "decision": entity["decision"],
            }
        entity_name = entity["name"] if entity else None
        masked = mask_entity(query, entity_name)
        qualifiers = extract_qualifiers(masked)
        result["qualifiers"] = qualifiers

        unsupported = detect_unsupported_dimension(masked)
        if unsupported:
            result.update(
                status="unsupported",
                reason_code="unsupported_dimension",
                dimensions=[unsupported],
                route="unsupported_gate",
                answer=(
                    f"The verified evidence corpus has no {unsupported} evidence, "
                    "so this question cannot be answered safely."
                ),
                limitations=[f"Unsupported evidence dimension: {unsupported}."],
            )
            return result

        decision = route_decision or self.router.route(query, entity_name)
        result["dimensions"] = list(decision.dimensions)
        result["route"] = decision.route

        if entity is None:
            result.update(
                reason_code="unknown_entity",
                answer="Name one of the plants covered by the verified evidence corpus.",
            )
            return result
        if entity["decision"] != "accepted":
            result.update(
                status="conflicting_evidence",
                reason_code="conflicted_entity",
                answer="The linked plant is quarantined because its source records conflict.",
            )
            return result
        if entity["plant_id"] in UNRESOLVED_COMMON_NAME_MAPPINGS:
            result.update(
                reason_code="unresolved_common_name_mapping",
                answer=(
                    "The common name is broader than the reviewed species-level source. "
                    "Please provide the scientific name before using this guidance."
                ),
                limitations=["Species-level evidence cannot be promoted to an ambiguous common name."],
            )
            return result
        if not decision.dimensions:
            result.update(
                reason_code="missing_care_dimension",
                answer="Ask about lighting, watering, soil, or fertilizer.",
            )
            return result

        records = self.store.retrieve(
            plant_id=entity["plant_id"],
            dimensions=decision.dimensions,
            qualifiers=qualifiers,
        )
        if not records:
            result.update(
                status="insufficient_evidence",
                reason_code="missing_verified_evidence",
                answer="No human-verified evidence matches this plant, dimension, and qualifier.",
                limitations=["No verified claim passed all structured retrieval gates."],
            )
            return result

        recovered_dimensions = {record["dimension"] for record in records}
        missing_dimensions = set(decision.dimensions) - recovered_dimensions
        if missing_dimensions:
            result.update(
                status="insufficient_evidence",
                reason_code="partial_dimension_coverage",
                answer="The verified corpus does not cover every requested care dimension.",
                limitations=[
                    "Missing verified dimensions: " + ", ".join(sorted(missing_dimensions)) + "."
                ],
            )
            return result

        evidence = [self._evidence_item(record) for record in records]
        answer, claims = self.generator.generate(records)
        result.update(
            status="answered",
            reason_code="verified_evidence_answer",
            answer=answer,
            claims=claims,
            evidence=evidence,
            limitations=[
                "The answer is extractive and limited to individually reviewed source claims."
            ],
        )
        return result


def validate_grounded_response(response: dict) -> list[str]:
    """Check that a response contains no unbound or unverified factual claim."""
    errors: list[str] = []
    evidence = {item.get("claim_id"): item for item in response.get("evidence", [])}
    claims = response.get("claims", [])
    if response.get("status") != "answered" and (evidence or claims):
        errors.append("non-answered response exposes claims or evidence")
    for claim_id, item in evidence.items():
        if not claim_id:
            errors.append("evidence item lacks claim_id")
        if item.get("review_status") != "human_verified":
            errors.append(f"non-verified evidence exposed: {claim_id}")
        citation = item.get("citation", {})
        if not citation.get("url") or not citation.get("section") or not citation.get("paragraph"):
            errors.append(f"incomplete citation: {claim_id}")
    referenced_ids: set[str] = set()
    for claim in claims:
        ids = claim.get("evidence_ids", [])
        if not ids:
            errors.append("answer claim lacks evidence_ids")
            continue
        for claim_id in ids:
            referenced_ids.add(claim_id)
            item = evidence.get(claim_id)
            if item is None:
                errors.append(f"claim references absent evidence: {claim_id}")
            elif claim.get("text") != item.get("claim_text"):
                errors.append(f"claim text differs from evidence: {claim_id}")
            if f"[{claim_id}]" not in response.get("answer", ""):
                errors.append(f"answer omits inline citation: {claim_id}")
    if response.get("status") == "answered":
        if referenced_ids != set(evidence):
            errors.append("claims and evidence do not have one-to-one coverage")
        expected_answer = "\n".join(
            f"- {item['claim_text']} [{item['claim_id']}]"
            for item in response.get("evidence", [])
        )
        if response.get("answer") != expected_answer:
            errors.append("answer contains text outside the cited extractive claims")
    return errors
