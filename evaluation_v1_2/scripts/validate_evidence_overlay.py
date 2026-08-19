#!/usr/bin/env python3
"""Validate the reviewed-evidence overlay without network access."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_v1_2.scripts.review_evidence import check, status_counts


ROOT = Path(__file__).resolve().parents[1]
V1_ROOT = ROOT.parent / "dataset_v1" / "data" / "processed"
CLAIM_ID_RE = re.compile(
    r"^claim:[a-z0-9_]+:(growth|soil|lighting|watering|fertilizer|taxonomy|pet_safety):[0-9]{3}$"
)
ALLOWED_SOURCE_HOSTS = {
    "plants.ces.ncsu.edu",
    "extension.psu.edu",
    "extension.umn.edu",
    "www.rhs.org.uk",
    "www.missouribotanicalgarden.org",
    "powo.science.kew.org",
    "www.gbif.org",
    "www.aspca.org",
}
CARE_DIMENSIONS = {"growth", "soil", "lighting", "watering", "fertilizer"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> tuple[list[str], dict]:
    errors: list[str] = []
    accepted = {
        plant["id"]: plant for plant in load(V1_ROOT / "plants.json")
    }
    overlay_plants = load(ROOT / "data" / "overlay_plants.json")
    evidence = load(ROOT / "data" / "evidence_overlay.json")

    if len(overlay_plants) != 10:
        errors.append(f"overlay must contain 10 selected plants, found {len(overlay_plants)}")

    selected: dict[str, dict] = {}
    for plant in overlay_plants:
        plant_id = plant.get("plant_id")
        if plant_id in selected:
            errors.append(f"duplicate selected plant: {plant_id}")
        selected[plant_id] = plant
        original = accepted.get(plant_id)
        if original is None:
            errors.append(f"selected plant is not accepted in v1: {plant_id}")
        elif original["name"] != plant.get("plant_name"):
            errors.append(f"selected plant name mismatch: {plant_id}")
        if plant.get("entity_scope") not in {"species", "genus", "common_name_mapping"}:
            errors.append(f"invalid entity scope: {plant_id}")

    claim_ids = set()
    claims_per_plant = Counter()
    dimensions_per_plant: dict[str, set[str]] = {
        plant_id: set() for plant_id in selected
    }
    source_contracts: dict[str, tuple[str, str, str]] = {}
    claim_keys = set()
    for claim in evidence:
        claim_id = claim.get("claim_id")
        if claim_id in claim_ids:
            errors.append(f"duplicate claim id: {claim_id}")
        claim_ids.add(claim_id)
        if not isinstance(claim_id, str) or not CLAIM_ID_RE.fullmatch(claim_id):
            errors.append(f"invalid claim id: {claim_id}")
        if claim.get("schema_version") != "1.2.0":
            errors.append(f"invalid schema version: {claim_id}")

        plant_id = claim.get("plant_id")
        selected_plant = selected.get(plant_id)
        if selected_plant is None:
            errors.append(f"claim references unselected plant: {claim_id}")
            continue
        for field in ("plant_name", "scientific_name", "entity_scope"):
            if claim.get(field) != selected_plant.get(field):
                errors.append(f"claim {field} mismatch: {claim_id}")
        dimension = claim.get("dimension")
        if dimension not in CARE_DIMENSIONS:
            errors.append(f"first overlay contains non-care dimension: {claim_id}")
        claims_per_plant[plant_id] += 1
        dimensions_per_plant[plant_id].add(dimension)

        qualifiers = claim.get("qualifiers")
        if not isinstance(qualifiers, dict):
            errors.append(f"qualifiers must be an object: {claim_id}")
            qualifiers = {}
        claim_key = (
            plant_id,
            dimension,
            json.dumps(qualifiers, sort_keys=True),
        )
        if claim_key in claim_keys:
            errors.append(f"duplicate plant/dimension/qualifier claim: {claim_id}")
        claim_keys.add(claim_key)

        if not isinstance(claim.get("claim_text"), str) or len(claim["claim_text"]) < 12:
            errors.append(f"claim text is too short: {claim_id}")
        if claim.get("source_type") not in {
            "extension", "rhs", "botanical_garden", "gbif", "kew", "aspca", "veterinary"
        }:
            errors.append(f"overlay uses non-authoritative source type: {claim_id}")
        if "horticultural_care" not in claim.get("authority_for", []):
            errors.append(f"care claim lacks care authority: {claim_id}")
        locator = claim.get("locator", {})
        parsed = urlparse(locator.get("url", ""))
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
            errors.append(f"source URL is not allowlisted HTTPS: {claim_id}")
        if not locator.get("section") or not locator.get("paragraph"):
            errors.append(f"claim lacks section/paragraph locator: {claim_id}")
        if claim.get("accessed_at") != "2026-08-12":
            errors.append(f"unexpected access date: {claim_id}")

        source_id = claim.get("source_id")
        source_contract = (
            claim.get("source_title"),
            claim.get("source_type"),
            locator.get("url"),
        )
        previous = source_contracts.setdefault(source_id, source_contract)
        if previous != source_contract:
            errors.append(f"source metadata drifts across claims: {source_id}")

        if claim.get("entity_scope") in {"genus", "common_name_mapping"} and not claim.get("review_notes"):
            errors.append(f"non-species claim lacks scope warning: {claim_id}")

    for plant_id in selected:
        if claims_per_plant[plant_id] < 2:
            errors.append(f"selected plant has fewer than two claims: {plant_id}")
        if "lighting" not in dimensions_per_plant[plant_id]:
            errors.append(f"selected plant lacks lighting evidence: {plant_id}")

    errors.extend(check(evidence))

    stats = {
        "plants": len(selected),
        "claims": len(evidence),
        "sources": len(source_contracts),
        "review_status_counts": status_counts(evidence),
        "dimensions": dict(sorted(Counter(item["dimension"] for item in evidence).items())),
    }
    return errors, stats


def main() -> int:
    errors, stats = validate()
    if errors:
        print(f"evidence overlay validation failed with {len(errors)} error(s):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("evidence overlay is valid")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
