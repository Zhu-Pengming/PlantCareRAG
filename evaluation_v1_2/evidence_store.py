"""Runtime-safe access to the v1.2 evidence overlay."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OVERLAY = ROOT / "data" / "evidence_overlay.json"
VERIFIED = "human_verified"


class VerifiedEvidenceStore:
    """Expose only claims that a human reviewer explicitly approved."""

    def __init__(self, overlay_path: Path = DEFAULT_OVERLAY) -> None:
        records = json.loads(overlay_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("evidence overlay must be a top-level list")
        self._verified = tuple(
            record for record in records if record.get("review_status") == VERIFIED
        )

    def all(self) -> list[dict]:
        return deepcopy(list(self._verified))

    def retrieve(
        self,
        *,
        plant_id: str,
        dimensions: list[str] | tuple[str, ...] | None = None,
        qualifiers: dict | None = None,
    ) -> list[dict]:
        """Retrieve verified evidence matching structured constraints."""
        allowed_dimensions = set(dimensions or [])
        required_qualifiers = qualifiers or {}
        matches = []
        for record in self._verified:
            if record.get("plant_id") != plant_id:
                continue
            if allowed_dimensions and record.get("dimension") not in allowed_dimensions:
                continue
            record_qualifiers = record.get("qualifiers") or {}
            if any(
                record_qualifiers.get(key) != value
                for key, value in required_qualifiers.items()
            ):
                continue
            matches.append(deepcopy(record))
        return sorted(matches, key=lambda record: record["claim_id"])
