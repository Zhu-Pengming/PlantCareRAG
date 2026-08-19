import json
import tempfile
import unittest
from pathlib import Path

from evaluation_v1_2.evidence_store import VerifiedEvidenceStore
from evaluation_v1_2.scripts.review_evidence import (
    check,
    record_decision,
    update_checklist,
)
from evaluation_v1_2.scripts.validate_evidence_overlay import validate


ROOT = Path(__file__).resolve().parents[1]


class EvidenceOverlayTests(unittest.TestCase):
    def test_overlay_is_structurally_valid(self):
        errors, stats = validate()
        self.assertEqual(errors, [])
        self.assertEqual(stats["plants"], 10)
        self.assertEqual(stats["claims"], 24)
        self.assertEqual(stats["sources"], 10)

    def test_committed_overlay_contains_no_unreviewed_records(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        self.assertTrue(evidence)
        self.assertTrue(
            all(item["review_status"] != "unreviewed" for item in evidence)
        )

    def test_mixed_state_checker_accepts_pending_verified_and_explained_rejected(self):
        claims = [
            {"claim_id": "a", "review_status": "agent_source_checked_pending_human_review"},
            {"claim_id": "b", "review_status": "human_verified", "locator": {"url": "https://example.org/b"}},
            {"claim_id": "c", "review_status": "rejected", "review_notes": "Source does not support it."},
        ]
        self.assertEqual(check(claims), [])

    def test_rejected_claim_requires_notes(self):
        errors = check([{"claim_id": "a", "review_status": "rejected"}])
        self.assertTrue(errors)

    def test_runtime_store_exposes_only_human_verified(self):
        records = [
            {"claim_id": "claim:a", "plant_id": "plant:a", "dimension": "watering", "qualifiers": {}, "review_status": "agent_source_checked_pending_human_review"},
            {"claim_id": "claim:b", "plant_id": "plant:a", "dimension": "watering", "qualifiers": {"season": "winter"}, "review_status": "human_verified"},
            {"claim_id": "claim:c", "plant_id": "plant:a", "dimension": "lighting", "qualifiers": {}, "review_status": "rejected"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            store = VerifiedEvidenceStore(path)
            self.assertEqual([item["claim_id"] for item in store.all()], ["claim:b"])
            self.assertEqual(
                [item["claim_id"] for item in store.retrieve(plant_id="plant:a", dimensions=["watering"], qualifiers={"season": "winter"})],
                ["claim:b"],
            )

    def test_checklist_ticks_only_after_every_source_claim_is_settled(self):
        url = "https://example.org/source"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checklist.md"
            path.write_text(f"- [ ] [Source]({url})\n", encoding="utf-8")
            claims = [
                {"review_status": "human_verified", "locator": {"url": url}},
                {"review_status": "agent_source_checked_pending_human_review", "locator": {"url": url}},
            ]
            self.assertEqual(update_checklist(path, claims), 0)
            claims[1]["review_status"] = "rejected"
            self.assertEqual(update_checklist(path, claims), 1)
            self.assertIn("- [x]", path.read_text(encoding="utf-8"))

    def test_review_decision_creates_backup_and_audit_entry(self):
        claim = {
            "claim_id": "claim:a",
            "source_id": "source:a",
            "entity_scope": "species",
            "review_status": "agent_source_checked_pending_human_review",
        }
        root = [claim]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            overlay = directory_path / "overlay.json"
            audit = directory_path / "audit.jsonl"
            overlay.write_text(json.dumps(root), encoding="utf-8")
            record_decision(
                claim,
                "human_verified",
                None,
                "test-reviewer",
                root,
                overlay,
                audit,
            )
            self.assertTrue((directory_path / "overlay.json.bak").exists())
            self.assertEqual(
                json.loads(overlay.read_text(encoding="utf-8"))[0]["review_status"],
                "human_verified",
            )
            audit_entry = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(audit_entry["reviewer"], "test-reviewer")
            self.assertEqual(audit_entry["decision"], "human_verified")

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
