#!/usr/bin/env python3
"""Deterministically prelabel and shortlist real-query candidates.

Outputs are automation aids, not benchmark annotations. In particular,
candidate_evidence_ids are never gold_evidence_ids and every record remains
benchmark-ineligible until human review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "query_candidates.json"
EVIDENCE = ROOT / "data" / "evidence_overlay.json"
RULES = ROOT / "config" / "query_triage_rules.json"
TRIAGE = ROOT / "data" / "query_auto_triage.json"
SHORTLIST = ROOT / "data" / "query_shortlist.json"
REPORT = ROOT / "reports" / "query_auto_triage_stats.json"


def contains(text: str, phrase: str) -> bool:
    phrase = phrase.casefold()
    if phrase.endswith("at") or phrase.endswith("l"):
        return re.search(rf"\b{re.escape(phrase)}\w*", text) is not None
    if " " in phrase or "-" in phrase:
        return phrase in text
    return re.search(
        rf"\b{re.escape(phrase)}(?:s|es|ed|ing)?\b",
        text,
    ) is not None


def classify(text: str, rules: dict[str, list[str]]) -> list[str]:
    folded = text.casefold()
    return [
        dimension
        for dimension, phrases in rules.items()
        if any(contains(folded, phrase) for phrase in phrases)
    ]


def verified_index(evidence_rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in evidence_rows:
        if row["review_status"] == "human_verified":
            result[(row["plant_id"], row["dimension"])].append(row["claim_id"])
    return {key: sorted(value) for key, value in result.items()}


def matched_plants(candidate: dict) -> list[str]:
    return sorted({match["plant_id"] for match in candidate["alias_matches"]})


def mapping_risks(candidate: dict, rules: dict) -> list[str]:
    plants = matched_plants(candidate)
    risks: list[str] = []
    if len(plants) > 1:
        risks.append("multi_entity_query")
    for plant_id in plants:
        risky = {value.casefold() for value in rules["mapping_risk_aliases"].get(plant_id, [])}
        used = {
            match["alias"].casefold()
            for match in candidate["alias_matches"]
            if match["plant_id"] == plant_id
        }
        if used and used <= risky:
            risks.append(
                "unresolved_common_name_mapping"
                if plant_id in {"plant:monstera", "plant:hoya_wax_plant"}
                else "species_mismatch_possible"
            )
    return sorted(set(risks))


def context_flags(candidate: dict, dimensions: list[str]) -> list[str]:
    text = candidate["raw_text"].casefold()
    flags: list[str] = []
    if "<img " in candidate["source_body_html"].casefold():
        flags.append("image_present")
    visual_phrases = ("see picture", "see photo", "pictured", "what is this", "these spots", "looks like this")
    if "image_present" in flags and (
        "identification" in dimensions or any(phrase in text for phrase in visual_phrases)
    ):
        flags.append("image_likely_required")
    if len(candidate["raw_body"]) < 40:
        flags.append("little_text_context")
    return flags


def triage_candidate(candidate: dict, rules: dict, evidence: dict) -> dict:
    supported = classify(candidate["raw_text"], rules["supported_dimensions"])
    boundary = classify(candidate["raw_text"], rules["boundary_dimensions"])
    dimensions = supported + [value for value in boundary if value not in supported]
    plants = matched_plants(candidate)
    risks = mapping_risks(candidate, rules)
    flags = context_flags(candidate, dimensions)
    evidence_ids = sorted(
        {
            claim_id
            for plant_id in plants
            for dimension in supported
            for claim_id in evidence.get((plant_id, dimension), [])
        }
    )
    covered_dimensions = sorted(
        {
            dimension
            for dimension in supported
            if any((plant_id, dimension) in evidence for plant_id in plants)
        }
    )
    missing_supported = sorted(set(supported) - set(covered_dimensions))
    if "unresolved_common_name_mapping" in risks:
        answerability = "entity_scope_unresolved"
    elif not dimensions:
        answerability = "needs_manual_intent_review"
    elif evidence_ids and not boundary and not missing_supported:
        answerability = "potentially_answerable"
    elif evidence_ids:
        answerability = "mixed_or_partial"
    else:
        answerability = "unsupported_or_insufficient"

    title_match = any("title" in match["locations"] for match in candidate["alias_matches"])
    metrics = candidate["source_metrics"]
    score = 0
    reasons: list[str] = []
    if title_match:
        score += 30
        reasons.append("entity_alias_in_title:+30")
    else:
        score += 5
        reasons.append("entity_alias_only_in_body:+5")
    if metrics["accepted_answer_id"]:
        score += 8
        reasons.append("accepted_answer_available:+8")
    if metrics["answer_count"] > 0:
        score += 4
        reasons.append("has_answer:+4")
    vote_bonus = max(-3, min(5, metrics["score"]))
    score += vote_bonus
    reasons.append(f"bounded_post_score:{vote_bonus:+d}")
    if supported:
        score += 8
        reasons.append("supported_dimension_signal:+8")
    if boundary:
        score += 3
        reasons.append("boundary_dimension_signal:+3")
    if not dimensions:
        score -= 15
        reasons.append("no_care_intent_signal:-15")
    if "image_likely_required" in flags:
        score -= 8
        reasons.append("image_likely_required:-8")
    if "little_text_context" in flags:
        score -= 8
        reasons.append("little_text_context:-8")
    if "species_mismatch_possible" in risks:
        score -= 6
        reasons.append("species_mismatch_possible:-6")
    return {
        "schema_version": "1.2.0-auto-triage",
        "candidate_id": candidate["candidate_id"],
        "predicted_plant_ids": plants,
        "mapping_risks": risks,
        "predicted_dimensions": dimensions,
        "covered_dimensions": covered_dimensions,
        "missing_supported_dimensions": missing_supported,
        "candidate_evidence_ids": evidence_ids,
        "context_flags": flags,
        "predicted_answerability": answerability,
        "priority_score": score,
        "priority_reasons": reasons,
        "automation_only": True,
        "benchmark_eligible": False,
    }


def make_shortlist(triage: list[dict], per_plant: int, preferred_answerable: int) -> list[dict]:
    by_plant: dict[str, list[dict]] = defaultdict(list)
    for row in triage:
        for plant_id in row["predicted_plant_ids"]:
            by_plant[plant_id].append(row)
    selected_ids: set[str] = set()
    result: list[dict] = []
    for plant_id in sorted(by_plant):
        rows = sorted(
            by_plant[plant_id],
            key=lambda row: (-row["priority_score"], row["candidate_id"]),
        )
        answerable = [
            row for row in rows
            if row["predicted_answerability"] in {"potentially_answerable", "mixed_or_partial"}
        ]
        boundary = [row for row in rows if row not in answerable]
        chosen: list[dict] = []
        for pool, limit in ((answerable, preferred_answerable), (boundary, per_plant)):
            for row in pool:
                if row["candidate_id"] in selected_ids:
                    continue
                chosen.append(row)
                selected_ids.add(row["candidate_id"])
                if len(chosen) >= limit:
                    break
        if len(chosen) < per_plant:
            for row in rows:
                if row["candidate_id"] in selected_ids:
                    continue
                chosen.append(row)
                selected_ids.add(row["candidate_id"])
                if len(chosen) == per_plant:
                    break
        for rank, row in enumerate(chosen[:per_plant], 1):
            result.append({**row, "quota_plant_id": plant_id, "quota_rank": rank})
    return result


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-plant", type=int)
    args = parser.parse_args()
    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    evidence_rows = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    per_plant = args.per_plant or rules["shortlist_per_plant"]
    if not 1 <= per_plant <= 25:
        print("--per-plant must be between 1 and 25", file=sys.stderr)
        return 2
    triage = [triage_candidate(row, rules, verified_index(evidence_rows)) for row in candidates]
    shortlist = make_shortlist(
        triage,
        per_plant,
        min(rules["preferred_potentially_answerable_per_plant"], per_plant),
    )
    write_json(TRIAGE, triage)
    write_json(SHORTLIST, shortlist)
    report = {
        "schema_version": "1.0.0",
        "rules_version": rules["rules_version"],
        "candidate_count": len(triage),
        "shortlist_count": len(shortlist),
        "shortlist_per_plant_target": per_plant,
        "benchmark_eligible_count": 0,
        "triage_answerability_counts": dict(sorted(Counter(row["predicted_answerability"] for row in triage).items())),
        "shortlist_answerability_counts": dict(sorted(Counter(row["predicted_answerability"] for row in shortlist).items())),
        "shortlist_quota_counts": dict(sorted(Counter(row["quota_plant_id"] for row in shortlist).items())),
        "shortlist_dimension_counts": dict(sorted(Counter(dimension for row in shortlist for dimension in row["predicted_dimensions"]).items())),
        "triage_sha256": hashlib.sha256(TRIAGE.read_bytes()).hexdigest(),
        "shortlist_sha256": hashlib.sha256(SHORTLIST.read_bytes()).hexdigest(),
    }
    write_json(REPORT, report)
    print(f"wrote {len(triage)} auto-triage rows and {len(shortlist)} shortlist rows")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
