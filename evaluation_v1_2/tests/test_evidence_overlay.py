import json
import unittest
from pathlib import Path

from evaluation_v1_2.scripts.validate_evidence_overlay import validate


ROOT = Path(__file__).resolve().parents[1]


class EvidenceOverlayTests(unittest.TestCase):
    def test_overlay_is_structurally_valid(self):
        errors, stats = validate()
        self.assertEqual(errors, [])
        self.assertEqual(stats["plants"], 10)
        self.assertEqual(stats["claims"], 24)
        self.assertEqual(stats["sources"], 10)

    def test_pending_review_is_not_misrepresented_as_human_verified(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        self.assertTrue(evidence)
        self.assertTrue(
            all(
                item["review_status"]
                == "agent_source_checked_pending_human_review"
                for item in evidence
            )
        )

    def test_qualified_evidence_exists_for_winter_watering(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        matches = [
            item
            for item in evidence
            if item["plant_id"] == "plant:snake_plant"
            and item["dimension"] == "watering"
            and item["qualifiers"] == {"season": "winter"}
        ]
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()

