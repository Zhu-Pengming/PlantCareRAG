"""Citation-first PlantCareRAG answer pipeline over verified evidence.

The default generator is deliberately extractive: every factual sentence is a
reviewed claim and carries its claim ID. This provides an auditable generation
floor before an optional LLM paraphraser is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
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
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
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


class GroundedGenerator(Protocol):
    def generate(
        self, query: str, evidence: list[dict]
    ) -> tuple[str, list[dict], dict]: ...


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
    def generate(
        query: str, evidence: list[dict]
    ) -> tuple[str, list[dict], dict]:
        claims = [
            {
                "text": record["claim_text"],
                "evidence_ids": [record["claim_id"]],
            }
            for record in evidence
        ]
        return render_claims(claims), claims, {
            "mode": "extractive",
            "provider": "local",
            "model": None,
            "fallback": False,
        }


class DeepSeekJSONGenerator:
    """Optional DeepSeek JSON Output generator with local post-validation."""

    def __init__(self, *, model: str, client=None, api_key: str | None = None) -> None:
        if not model or not model.strip():
            raise ValueError("an explicit DeepSeek model ID is required")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "DeepSeek generation requires evaluation_v1_2/requirements-llm.txt"
                ) from exc
            resolved_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not resolved_key:
                raise RuntimeError("DeepSeek client requires DEEPSEEK_API_KEY")
            if len(resolved_key) < 12 or resolved_key == "sk-your-key-here":
                raise RuntimeError("DEEPSEEK_API_KEY is still a placeholder or is too short")
            try:
                client = OpenAI(api_key=resolved_key, base_url=DEEPSEEK_BASE_URL)
            except Exception as exc:
                raise RuntimeError(
                    "DeepSeek client initialization failed; check DEEPSEEK_API_KEY"
                ) from exc
        if not (
            hasattr(client, "chat")
            and hasattr(client.chat, "completions")
            and hasattr(client.chat.completions, "create")
        ):
            raise RuntimeError("client does not support chat.completions.create")
        self.client = client
        self.model = model.strip()

    def generate(
        self, query: str, evidence: list[dict]
    ) -> tuple[str, list[dict], dict]:
        evidence_payload = [
            {
                "claim_id": record["claim_id"],
                "claim_text": record["claim_text"],
                "dimension": record["dimension"],
                "qualifiers": record.get("qualifiers", {}),
            }
            for record in evidence
        ]
        schema_example = {
            "sentences": [
                {
                    "text": "A concise sentence supported only by the cited claims.",
                    "evidence_ids": ["claim:example:dimension:001"],
                }
            ]
        }
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return JSON only. Answer the plant-care question using only "
                        "the supplied evidence. Use every claim_id exactly once. Each "
                        "sentence must list all claim_ids that support it. Do not add "
                        "facts, numbers, frequencies, diagnoses, or safety claims absent "
                        "from those claims. The required JSON shape is: "
                        + json.dumps(schema_example)
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": query, "evidence": evidence_payload},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        if not response.choices:
            raise RuntimeError("DeepSeek response contained no choices")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek JSON Output returned empty content")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("DeepSeek response was not valid JSON") from exc
        claims = payload.get("sentences") or []
        if not isinstance(claims, list) or not claims:
            raise RuntimeError("DeepSeek response contained no answer sentences")
        if any(
            not isinstance(claim, dict)
            or not isinstance(claim.get("text"), str)
            or not isinstance(claim.get("evidence_ids"), list)
            or not all(isinstance(value, str) for value in claim["evidence_ids"])
            for claim in claims
        ):
            raise RuntimeError("DeepSeek response did not match the local answer schema")
        return render_claims(claims), claims, {
            "mode": "llm_json",
            "provider": "deepseek",
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "fallback": False,
        }


def render_claims(claims: list[dict]) -> str:
    return "\n".join(
        "- " + claim["text"] + " " + " ".join(
            f"[{claim_id}]" for claim_id in claim["evidence_ids"]
        )
        for claim in claims
    )


def numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


class GroundedRAGEngine:
    """Route, retrieve verified claims, and return a citation-bound answer."""

    def __init__(
        self,
        *,
        router: DimensionRouter | None = None,
        query_engine: QueryEngine | None = None,
        evidence_store: VerifiedEvidenceStore | None = None,
        generator: GroundedGenerator | None = None,
    ) -> None:
        self.linker = query_engine or QueryEngine()
        self.store = evidence_store or VerifiedEvidenceStore()
        self.router = router or LexicalDimensionRouter()
        self.generator = generator or ExtractiveGroundedGenerator()

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
            "generation": {
                "mode": "none",
                "provider": "local",
                "model": None,
                "fallback": False,
            },
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
        try:
            answer, claims, generation = self.generator.generate(query, records)
        except Exception as exc:
            generation_error = type(exc).__name__
            answer, claims, generation = ExtractiveGroundedGenerator.generate(query, records)
            generation.update(
                fallback=True,
                requested_mode="llm_json",
                fallback_reason=generation_error,
            )
        result.update(
            status="answered",
            reason_code="verified_evidence_answer",
            answer=answer,
            claims=claims,
            evidence=evidence,
            generation=generation,
            limitations=[
                (
                    "The answer is extractive and limited to individually reviewed source claims."
                    if generation["mode"] == "extractive"
                    else "LLM wording passed structural, citation, and numeric checks; automated checks do not prove semantic entailment."
                )
            ],
        )
        validation_errors = validate_grounded_response(result)
        if validation_errors and generation["mode"] != "extractive":
            answer, claims, fallback = ExtractiveGroundedGenerator.generate(query, records)
            fallback.update(
                fallback=True,
                requested_mode="llm_json",
                fallback_reason="; ".join(validation_errors),
            )
            result.update(
                answer=answer,
                claims=claims,
                generation=fallback,
                limitations=[
                    "LLM output failed grounding checks; returned the extractive safety fallback."
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
    reference_count = 0
    generation_mode = response.get("generation", {}).get("mode", "extractive")
    for claim in claims:
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            errors.append("answer claim has empty text")
        ids = claim.get("evidence_ids", [])
        if not ids:
            errors.append("answer claim lacks evidence_ids")
            continue
        for claim_id in ids:
            reference_count += 1
            referenced_ids.add(claim_id)
            item = evidence.get(claim_id)
            if item is None:
                errors.append(f"claim references absent evidence: {claim_id}")
            elif generation_mode == "extractive" and claim.get("text") != item.get("claim_text"):
                errors.append(f"claim text differs from evidence: {claim_id}")
            if f"[{claim_id}]" not in response.get("answer", ""):
                errors.append(f"answer omits inline citation: {claim_id}")
    if response.get("status") == "answered":
        if referenced_ids != set(evidence):
            errors.append("claims and evidence do not have one-to-one coverage")
        if reference_count != len(evidence):
            errors.append("an evidence ID is missing or cited more than once")
        for claim in claims:
            cited_text = " ".join(
                evidence[claim_id]["claim_text"]
                for claim_id in claim.get("evidence_ids", [])
                if claim_id in evidence
            )
            if not numeric_tokens(claim.get("text", "")) <= numeric_tokens(cited_text):
                errors.append("generated claim adds a number absent from cited evidence")
        expected_answer = render_claims(claims)
        if response.get("answer") != expected_answer:
            errors.append("answer contains text outside the structured cited claims")
    return errors
