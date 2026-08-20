"""Embedding fallback for care-dimension routing.

Lexical routing remains authoritative when it returns at least one dimension.
Embeddings are used only when lexical routing is empty, limiting regressions
and making the semantic contribution directly measurable.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from dataset_v1.query_engine import mask_entity
from evaluation_v1_2.answerability import classify_dimensions


ROOT = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_CACHE = ROOT / "models"
EVIDENCE = ROOT / "data" / "evidence_overlay.json"


class SemanticDimensionRouter:
    """Use verified claim text as dimension prototypes."""

    def __init__(
        self,
        *,
        threshold: float,
        model_name: str = MODEL_NAME,
        cache_dir: Path = MODEL_CACHE,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is required; install evaluation_v1_2/requirements-embedding.txt"
            ) from exc
        self.threshold = threshold
        self.model_name = model_name
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        verified = [
            row
            for row in json.loads(EVIDENCE.read_text(encoding="utf-8"))
            if row["review_status"] == "human_verified"
        ]
        claims_by_dimension: dict[str, list[str]] = defaultdict(list)
        for row in verified:
            claims_by_dimension[row["dimension"]].append(row["claim_text"])
        self.dimensions = sorted(claims_by_dimension)
        self.prototype_dimensions = [
            dimension
            for dimension in self.dimensions
            for _ in claims_by_dimension[dimension]
        ]
        passages = [
            "passage: " + claim
            for dimension in self.dimensions
            for claim in sorted(claims_by_dimension[dimension])
        ]
        self.prototype_vectors = np.asarray(list(self.model.embed(passages)))

    def semantic_scores(self, masked_query: str) -> dict[str, float]:
        query_vector = np.asarray(
            list(self.model.embed(["query: " + masked_query]))[0]
        )
        similarities = self.prototype_vectors @ query_vector
        scores = {dimension: float("-inf") for dimension in self.dimensions}
        for dimension, similarity in zip(self.prototype_dimensions, similarities):
            scores[dimension] = max(scores[dimension], float(similarity))
        return scores

    def predict(self, query: str, entity_name: str | None) -> tuple[list[str], dict]:
        masked = mask_entity(query, entity_name)
        lexical = classify_dimensions(masked)
        if lexical:
            return lexical, {
                "route": "lexical",
                "semantic_scores": {},
                "threshold": self.threshold,
            }
        scores = self.semantic_scores(masked)
        best_dimension, best_score = max(scores.items(), key=lambda item: item[1])
        predicted = [best_dimension] if best_score >= self.threshold else []
        return predicted, {
            "route": "semantic_fallback" if predicted else "no_dimension",
            "semantic_scores": dict(sorted(scores.items())),
            "threshold": self.threshold,
        }
