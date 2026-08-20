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
from evaluation_v1_2.scripts.validate_evidence_overlay import validate, validate_qualifiers


ROOT = Path(__file__).resolve().parents[1]


class EvidenceOverlayTests(unittest.TestCase):
    def test_overlay_is_structurally_valid(self):
        errors, stats = validate()
        self.assertEqual(errors, [])
        self.assertEqual(stats["plants"], 10)
        self.assertEqual(stats["claims"], 38)
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

    def test_qualifier_vocabulary_rejects_unknown_keys_and_values(self):
        self.assertEqual(validate_qualifiers("claim:a", {"season": "winter"}), [])
        self.assertTrue(validate_qualifiers("claim:a", {"scope": "indoor"}))
        self.assertTrue(validate_qualifiers("claim:a", {"season": "cold_months"}))

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

    def test_review_findings_are_split_by_qualifier_scope(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        aloe_watering = {
            json.dumps(item["qualifiers"], sort_keys=True): item["claim_text"]
            for item in evidence
            if item["plant_id"] == "plant:aloe_vera"
            and item["dimension"] == "watering"
        }
        self.assertEqual(
            aloe_watering["{}"],
            "Allow the soil to dry completely between waterings.",
        )
        self.assertEqual(
            aloe_watering['{"season": "winter"}'],
            "Water less frequently in winter.",
        )

    def test_inferred_snake_plant_indoor_qualifier_was_removed(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        lighting = next(
            item
            for item in evidence
            if item["claim_id"] == "claim:snake_plant:lighting:001"
        )
        self.assertEqual(lighting["qualifiers"], {})
        self.assertNotIn("indoors", lighting["claim_text"].casefold())

    def test_fiddle_leaf_fig_review_findings_are_dimension_atomic(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        claims = {
            item["dimension"]: item
            for item in evidence
            if item["plant_id"] == "plant:fiddle_leaf_fig"
        }
        self.assertEqual(set(claims), {"lighting", "soil", "watering"})
        self.assertEqual(claims["lighting"]["qualifiers"], {})
        self.assertEqual(
            claims["soil"]["claim_text"],
            "Use moist, well-drained, loamy, acidic soil.",
        )
        self.assertNotIn("soil", claims["watering"]["claim_text"].casefold())

    def test_devils_ivy_low_light_cost_is_separate_from_general_preference(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        lighting = [
            item
            for item in evidence
            if item["plant_id"] == "plant:devils_ivy"
            and item["dimension"] == "lighting"
        ]
        self.assertEqual(len(lighting), 2)
        by_qualifier = {
            json.dumps(item["qualifiers"], sort_keys=True): item for item in lighting
        }
        self.assertEqual(by_qualifier["{}"]["locator"]["paragraph"], "Description paragraph 2")
        self.assertIn(
            "loss of leaf variegation",
            by_qualifier['{"condition": "low_light"}']["claim_text"],
        )

    def test_aglaonema_review_findings_are_split_by_scope_and_season(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        claims = [item for item in evidence if item["plant_id"] == "plant:chinese_evergreen"]
        self.assertEqual(len(claims), 7)
        qualifier_keys = {
            (item["dimension"], json.dumps(item["qualifiers"], sort_keys=True))
            for item in claims
        }
        self.assertIn(("lighting", "{}"), qualifier_keys)
        self.assertIn(("lighting", '{"environment": "indoor"}'), qualifier_keys)
        self.assertIn(
            (
                "lighting",
                '{"condition": "low_light", "environment": "indoor"}',
            ),
            qualifier_keys,
        )
        self.assertIn(("watering", '{"season": "spring_to_autumn"}'), qualifier_keys)
        self.assertIn(("watering", '{"season": "winter"}'), qualifier_keys)
        self.assertIn(("watering", "{}"), qualifier_keys)
        self.assertIn(("watering", '{"condition": "cold_water"}'), qualifier_keys)

    def test_peperomia_review_findings_are_dimension_atomic(self):
        evidence = json.loads(
            (ROOT / "data" / "evidence_overlay.json").read_text(encoding="utf-8")
        )
        claims = [item for item in evidence if item["plant_id"] == "plant:peperomia"]
        self.assertEqual(len(claims), 5)
        soil = next(item for item in claims if item["dimension"] == "soil")
        watering = next(item for item in claims if item["dimension"] == "watering")
        direct_sun = next(
            item for item in claims
            if item["qualifiers"] == {"condition": "direct_sun"}
        )
        self.assertIn("loam and sand", soil["claim_text"])
        self.assertNotIn("soil", watering["claim_text"].casefold())
        self.assertIn("scorch", direct_sun["claim_text"].casefold())
        overwatering = next(
            item for item in claims
            if item["qualifiers"] == {"condition": "overwatering"}
        )
        self.assertIn("yellowing or curling", overwatering["claim_text"])

    def test_condition_tag_is_not_used_as_a_hard_filter(self):
        records = [
            {
                "claim_id": "claim:a",
                "plant_id": "plant:a",
                "dimension": "watering",
                "qualifiers": {},
                "review_status": "human_verified",
            },
            {
                "claim_id": "claim:b",
                "plant_id": "plant:a",
                "dimension": "watering",
                "qualifiers": {"condition": "overwatering"},
                "review_status": "human_verified",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            store = VerifiedEvidenceStore(path)
            matches = store.retrieve(
                plant_id="plant:a",
                dimensions=["watering"],
                qualifiers={"condition": "overwatering"},
            )
        self.assertEqual([item["claim_id"] for item in matches], ["claim:a", "claim:b"])


if __name__ == "__main__":
    unittest.main()
