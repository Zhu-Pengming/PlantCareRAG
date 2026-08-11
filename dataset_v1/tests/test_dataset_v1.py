import hashlib
import json
import unittest
from pathlib import Path

from dataset_v1.scripts.evaluate_baseline import classify_dimension, evaluate
from dataset_v1.scripts.validate_dataset import validate


ROOT = Path(__file__).resolve().parents[1]


class DatasetV1Tests(unittest.TestCase):
    def test_committed_artifact_checksums(self):
        lines = (ROOT / "reports" / "artifact_checksums.sha256").read_text(
            encoding="utf-8"
        ).splitlines()
        for line in lines:
            expected, relative_path = line.split("  ", 1)
            actual = hashlib.sha256((ROOT.parent / relative_path).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative_path)

    def test_manifest_rejects_unknown_license_dataset(self):
        manifest = json.loads((ROOT / "config" / "datasets.json").read_text(encoding="utf-8"))
        datasets = {item["id"]: item for item in manifest["datasets"]}
        rejected = datasets[
            "kaggle:prakash27x/indoor-house-plants-dataset-with-care-instructions"
        ]
        self.assertEqual(rejected["license"], "Unknown")
        self.assertEqual(rejected["decision"], "rejected")

    def test_processed_dataset_is_valid(self):
        errors, stats = validate()
        self.assertEqual(errors, [])
        self.assertEqual(
            stats,
            {
                "plants": 263,
                "entries": 1315,
                "queries": 1315,
                "contexts": 650,
                "excluded_conflicted_plants": 124,
            },
        )

    def test_conflicted_target_plants_do_not_leak_into_kb(self):
        plants = json.loads((ROOT / "data" / "processed" / "plants.json").read_text(encoding="utf-8"))
        names = {plant["normalized_name"] for plant in plants}
        self.assertNotIn("peace lily", names)
        self.assertNotIn("spider plant", names)
        self.assertNotIn("pothos", names)
        self.assertIn("snake plant", names)
        self.assertIn("zz plant", names)
        self.assertIn("monstera", names)

    def test_context_data_is_not_ground_truth(self):
        contexts = json.loads(
            (ROOT / "data" / "processed" / "context_samples.json").read_text(encoding="utf-8")
        )
        self.assertTrue(contexts)
        for context in contexts:
            self.assertEqual(context["allowed_use"], "context_stress_test_only")
            self.assertEqual(
                context["prohibited_use"], "ground_truth_or_medical_safety_claim"
            )
            self.assertEqual(
                context["pest_presence"] == "None",
                context["pest_severity"] == "None",
            )

    def test_structured_contract_stage_reaches_all_gold(self):
        result = evaluate()
        for split in ("dev", "test"):
            structured = result["metrics"][split]["entity_dimension_bm25"]
            raw = result["metrics"][split]["raw_bm25"]
            self.assertEqual(structured["hit_at_1"], 1.0)
            self.assertGreater(structured["hit_at_1"], raw["hit_at_1"])

    def test_entity_text_is_masked_before_dimension_routing(self):
        self.assertEqual(
            classify_dimension("How quickly does Water Lily grow?", "Water Lily"),
            "growth",
        )
        self.assertEqual(
            classify_dimension(
                "What potting medium should I use for Water Lily?", "Water Lily"
            ),
            "soil",
        )


if __name__ == "__main__":
    unittest.main()
