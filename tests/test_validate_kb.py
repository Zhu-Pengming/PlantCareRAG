import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_kb import DATA_ROOT, validate_kb


class KnowledgeBaseValidationTests(unittest.TestCase):
    def test_repository_knowledge_base_is_valid(self):
        errors, stats = validate_kb(DATA_ROOT)
        self.assertEqual(errors, [])
        self.assertEqual(stats["sources"], 16)
        self.assertEqual(stats["plants"], 3)
        self.assertEqual(stats["aliases"], 22)
        self.assertEqual(stats["entries"], 31)

    def test_unknown_source_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "entries").mkdir()
            sources = json.loads((DATA_ROOT / "sources.json").read_text(encoding="utf-8"))
            plants = json.loads((DATA_ROOT / "plants.json").read_text(encoding="utf-8"))
            aliases = json.loads((DATA_ROOT / "entity_aliases.json").read_text(encoding="utf-8"))
            entries = json.loads(
                (DATA_ROOT / "entries" / "monstera_deliciosa.json").read_text(encoding="utf-8")
            )
            entries[0]["source_refs"][0]["source_id"] = "source:does_not_exist"
            (root / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
            (root / "plants.json").write_text(json.dumps(plants), encoding="utf-8")
            (root / "entity_aliases.json").write_text(json.dumps(aliases), encoding="utf-8")
            (root / "entries" / "entries.json").write_text(json.dumps(entries), encoding="utf-8")

            errors, _ = validate_kb(root)

            self.assertTrue(any("unknown source 'source:does_not_exist'" in error for error in errors))

    def test_pet_safety_requires_two_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "entries").mkdir()
            sources = json.loads((DATA_ROOT / "sources.json").read_text(encoding="utf-8"))
            plants = json.loads((DATA_ROOT / "plants.json").read_text(encoding="utf-8"))
            aliases = json.loads((DATA_ROOT / "entity_aliases.json").read_text(encoding="utf-8"))
            entries = json.loads(
                (DATA_ROOT / "entries" / "monstera_deliciosa.json").read_text(encoding="utf-8")
            )
            pet_safety = next(entry for entry in entries if entry["dimension"] == "pet_safety")
            pet_safety["source_refs"] = pet_safety["source_refs"][:1]
            (root / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
            (root / "plants.json").write_text(json.dumps(plants), encoding="utf-8")
            (root / "entity_aliases.json").write_text(json.dumps(aliases), encoding="utf-8")
            (root / "entries" / "entries.json").write_text(json.dumps(entries), encoding="utf-8")

            errors, _ = validate_kb(root)

            self.assertTrue(any("pet_safety entries require at least two source_refs" in error for error in errors))

    def test_aspca_pet_sources_record_the_verified_scientific_names(self):
        sources = {
            source["id"]: source
            for source in json.loads((DATA_ROOT / "sources.json").read_text(encoding="utf-8"))
        }
        expected = {
            "source:aspca_monstera_deliciosa": (
                "Monstera deliciosa",
                "/swiss-cheese-plant",
            ),
            "source:aspca_epipremnum_aureum": (
                "Epipremnum aureum",
                "/golden-pothos",
            ),
            "source:aspca_dracaena_trifasciata": (
                "Sansevieria trifasciata",
                "/snake-plant",
            ),
        }
        for source_id, (scientific_name, url_suffix) in expected.items():
            source = sources[source_id]
            self.assertIn(scientific_name, source["title"])
            self.assertIn(f"Scientific Name 明确为 {scientific_name}", source["notes"])
            self.assertTrue(source["url"].endswith(url_suffix))

    def test_snake_plant_accepted_name_and_legacy_synonym_share_an_entity(self):
        plants = json.loads((DATA_ROOT / "plants.json").read_text(encoding="utf-8"))
        snake_plant = next(
            plant for plant in plants if plant["id"] == "plant:dracaena_trifasciata"
        )
        self.assertEqual(snake_plant["scientific_name"], "Dracaena trifasciata")
        self.assertIn("Sansevieria trifasciata", snake_plant["synonyms"])

        aliases = json.loads(
            (DATA_ROOT / "entity_aliases.json").read_text(encoding="utf-8")
        )
        scientific_aliases = {
            alias["value"]: alias["plant_id"]
            for alias in aliases
            if alias["value"] in {"Dracaena trifasciata", "Sansevieria trifasciata"}
        }
        self.assertEqual(
            scientific_aliases,
            {
                "Dracaena trifasciata": "plant:dracaena_trifasciata",
                "Sansevieria trifasciata": "plant:dracaena_trifasciata",
            },
        )


if __name__ == "__main__":
    unittest.main()
