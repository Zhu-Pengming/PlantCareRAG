#!/usr/bin/env python3
"""Run a lexical baseline and a deterministic entity+dimension contract stage."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = ROOT / "data" / "processed"
RESULT_ROOT = ROOT / "results"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document["index_text"]) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        document_frequencies = Counter()
        for tokens in self.tokens:
            document_frequencies.update(set(tokens))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequencies.items()
        }

    def score(self, query: str, candidate_indices: list[int] | None = None) -> list[tuple[float, int]]:
        indices = candidate_indices if candidate_indices is not None else list(range(len(self.documents)))
        query_terms = tokenize(query)
        scores = []
        for index in indices:
            frequency = self.term_frequencies[index]
            length = self.lengths[index]
            score = 0.0
            for term in query_terms:
                count = frequency.get(term, 0)
                if not count:
                    continue
                denominator = count + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                score += self.idf.get(term, 0.0) * count * (self.k1 + 1) / denominator
            scores.append((score, index))
        return sorted(scores, key=lambda item: (-item[0], self.documents[item[1]]["id"]))


def link_plant(query: str, plants: list[dict]) -> str | None:
    query_folded = query.casefold()
    matches = [
        plant
        for plant in plants
        if plant["name"].casefold() in query_folded
    ]
    if not matches:
        return None
    matches.sort(key=lambda plant: (-len(plant["name"]), plant["id"]))
    return matches[0]["id"]


def classify_dimension(query: str, linked_plant_name: str | None = None) -> str | None:
    text = query.casefold()
    if linked_plant_name:
        text = text.replace(linked_plant_name.casefold(), " ")
    rules = (
        ("watering", ("water",)),
        ("fertilizer", ("fertilizer", "plant food", "feed")),
        ("soil", ("soil", "potting medium")),
        ("lighting", ("light", "place")),
        ("growth", ("grow", "quickly")),
    )
    for dimension, phrases in rules:
        if any(phrase in text for phrase in phrases):
            return dimension
    return None


def metrics(rows: list[dict], ranking_field: str) -> dict:
    hit1 = hit3 = 0
    reciprocal_rank = 0.0
    for row in rows:
        ranking = row[ranking_field]
        gold = row["gold_entry_id"]
        if ranking and ranking[0] == gold:
            hit1 += 1
        if gold in ranking[:3]:
            hit3 += 1
        if gold in ranking:
            reciprocal_rank += 1 / (ranking.index(gold) + 1)
    count = len(rows)
    return {
        "n": count,
        "hit_at_1": hit1 / count if count else 0.0,
        "hit_at_3": hit3 / count if count else 0.0,
        "mrr": reciprocal_rank / count if count else 0.0,
    }


def evaluate(processed_root: Path = PROCESSED_ROOT) -> dict:
    plants = json.loads((processed_root / "plants.json").read_text(encoding="utf-8"))
    entries = json.loads((processed_root / "entries.json").read_text(encoding="utf-8"))
    queries = json.loads((processed_root / "queries.json").read_text(encoding="utf-8"))
    documents = [
        {
            **entry,
            "index_text": f"{entry['plant_name']} {entry['dimension']} {entry['value']} {entry['content']}"
        }
        for entry in entries
    ]
    bm25 = BM25(documents)
    rows = []
    for query in queries:
        raw = bm25.score(query["text"])
        linked_plant = link_plant(query["text"], plants)
        linked_name = next(
            (plant["name"] for plant in plants if plant["id"] == linked_plant),
            None,
        )
        predicted_dimension = classify_dimension(query["text"], linked_name)
        candidates = [
            index
            for index, document in enumerate(documents)
            if document["plant_id"] == linked_plant
            and document["dimension"] == predicted_dimension
        ]
        structured = bm25.score(query["text"], candidates)
        rows.append(
            {
                "query_id": query["id"],
                "split": query["split"],
                "gold_entry_id": query["gold_entry_id"],
                "linked_plant_id": linked_plant,
                "predicted_dimension": predicted_dimension,
                "raw_ranking": [documents[index]["id"] for _, index in raw[:5]],
                "structured_ranking": [documents[index]["id"] for _, index in structured[:5]],
            }
        )
    result = {
        "schema_version": "1.0.0",
        "benchmark_scope": "Template-derived retrieval contract test; not a real-user QA benchmark.",
        "config": {"bm25_k1": 1.5, "bm25_b": 0.75, "top_k": 5},
        "metrics": {},
        "queries": rows,
    }
    for split in ("all", "dev", "test"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        result["metrics"][split] = {
            "raw_bm25": metrics(subset, "raw_ranking"),
            "entity_dimension_bm25": metrics(subset, "structured_ranking"),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULT_ROOT / "baseline.json")
    args = parser.parse_args()
    result = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
