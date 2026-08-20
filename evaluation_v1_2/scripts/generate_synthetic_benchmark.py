#!/usr/bin/env python3
"""Generate a deterministic evidence-conditioned synthetic benchmark."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "data" / "evidence_overlay.json"
PLANTS = ROOT / "data" / "overlay_plants.json"
OUTPUT = ROOT / "data" / "synthetic_benchmark.json"
REPORT = ROOT / "reports" / "synthetic_benchmark_stats.json"

TEMPLATES = {
    "lighting": (
        "What light does {plant} need?",
        "Where should I place my {plant} so it gets suitable light?",
        "Will {plant} be okay in a dim corner?",
    ),
    "watering": (
        "How should I water {plant}?",
        "What watering routine is suitable for {plant}?",
        "What should I know before watering my {plant} again?",
    ),
    "soil": (
        "What soil should I use for {plant}?",
        "What kind of potting soil is suitable for {plant}?",
        "What should I fill the pot with for {plant}?",
    ),
    "fertilizer": (
        "How should I fertilize {plant}?",
        "What fertilizer routine is suitable for {plant}?",
        "Does {plant} need extra nutrients, and how often?",
    ),
}
DIMENSION_PHRASES = {
    "lighting": "light",
    "watering": "watering",
    "soil": "soil",
    "fertilizer": "fertilizer",
}


def query_id(*parts: str) -> str:
    return "qv12:synthetic:" + ":".join(parts)


def make_row(
    *,
    identifier: str,
    text: str,
    plant_id: str | None,
    dimensions: list[str],
    answerability: str,
    gold: list[str],
    behavior: str,
) -> dict:
    return {
        "schema_version": "1.2.0",
        "query_id": identifier,
        "raw_text": text,
        "provenance": "synthetic_generated_case",
        "source_url": None,
        "benchmark_eligible": True,
        "plant_id": plant_id,
        "expected_dimensions": dimensions,
        "qualifiers": {},
        "answerability": answerability,
        "gold_evidence_ids": sorted(gold),
        "expected_behavior": behavior,
        "split_hash": hashlib.sha256(identifier.encode("utf-8")).hexdigest(),
    }


def build() -> list[dict]:
    plant_rows = json.loads(PLANTS.read_text(encoding="utf-8"))
    names = {row["plant_id"]: row["plant_name"] for row in plant_rows}
    verified = [
        row
        for row in json.loads(EVIDENCE.read_text(encoding="utf-8"))
        if row["review_status"] == "human_verified"
    ]
    by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    dimensions_by_plant: dict[str, set[str]] = defaultdict(set)
    for claim in verified:
        key = (claim["plant_id"], claim["dimension"])
        by_pair[key].append(claim["claim_id"])
        dimensions_by_plant[claim["plant_id"]].add(claim["dimension"])

    rows: list[dict] = []
    for (plant_id, dimension), gold in sorted(by_pair.items()):
        short = plant_id.removeprefix("plant:")
        for variant, template in enumerate(TEMPLATES[dimension], 1):
            rows.append(
                make_row(
                    identifier=query_id(short, dimension, f"v{variant}"),
                    text=template.format(plant=names[plant_id]),
                    plant_id=plant_id,
                    dimensions=[dimension],
                    answerability="answerable",
                    gold=gold,
                    behavior="answer",
                )
            )

    for plant_id, dimensions in sorted(dimensions_by_plant.items()):
        chosen = sorted(dimensions)[:2]
        gold = [
            claim_id
            for dimension in chosen
            for claim_id in by_pair[(plant_id, dimension)]
        ]
        phrase_a, phrase_b = (DIMENSION_PHRASES[value] for value in chosen)
        rows.append(
            make_row(
                identifier=query_id(plant_id.removeprefix("plant:"), "multi"),
                text=f"What {phrase_a} and {phrase_b} guidance should I follow for {names[plant_id]}?",
                plant_id=plant_id,
                dimensions=chosen,
                answerability="answerable",
                gold=gold,
                behavior="answer",
            )
        )

    for plant_id in (
        "plant:snake_plant",
        "plant:zz_plant",
        "plant:aloe_vera",
        "plant:chinese_evergreen",
    ):
        rows.append(
            make_row(
                identifier=query_id(plant_id.removeprefix("plant:"), "pet_safety"),
                text=f"Is {names[plant_id]} pet safe?",
                plant_id=plant_id,
                dimensions=["pet_safety"],
                answerability="unsupported",
                gold=[],
                behavior="abstain",
            )
        )

    for plant_id, dimension in (
        ("plant:monstera", "lighting"),
        ("plant:hoya_wax_plant", "watering"),
    ):
        rows.append(
            make_row(
                identifier=query_id(plant_id.removeprefix("plant:"), "mapping_clarification"),
                text=TEMPLATES[dimension][0].format(plant=names[plant_id]),
                plant_id=plant_id,
                dimensions=[dimension],
                answerability="clarification_needed",
                gold=[],
                behavior="clarify",
            )
        )
    return sorted(rows, key=lambda row: row["query_id"])


def main() -> int:
    rows = build()
    OUTPUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_counts = Counter(
        "dev" if int(row["split_hash"], 16) % 2 == 0 else "test"
        for row in rows
    )
    report = {
        "schema_version": "1.0.0",
        "evaluation_type": "evidence_conditioned_synthetic_contract_benchmark",
        "external_validity": False,
        "warning": "Generated from the evidence being tested; do not report as real-user performance.",
        "generation_styles": {
            "v1": "direct attribute wording",
            "v2": "novice wording with explicit dimension cue",
            "v3": "implicit paraphrase; frozen before baseline evaluation"
        },
        "query_count": len(rows),
        "answerability_counts": dict(sorted(Counter(row["answerability"] for row in rows).items())),
        "dimension_counts": dict(sorted(Counter(value for row in rows for value in row["expected_dimensions"]).items())),
        "split_counts": dict(sorted(split_counts.items())),
        "benchmark_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} synthetic benchmark queries -> {OUTPUT}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
