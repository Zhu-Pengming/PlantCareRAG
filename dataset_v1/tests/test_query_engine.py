import unittest

from dataset_v1.query_engine import QueryEngine, detect_unsupported_dimension


class QueryEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = QueryEngine()

    def test_supported_entity_question_returns_citation(self):
        response = self.engine.response("When should I water Snake Plant?")
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["reason_code"], "supported_entity_care")
        self.assertEqual(response["dimension"], "watering")
        self.assertEqual(response["results"][0]["plant_name"], "Snake Plant")
        self.assertEqual(response["results"][0]["citation"]["license"], "CC BY 4.0")
        self.assertTrue(response["results"][0]["citation"]["raw_row_numbers"])

    def test_conflicted_entity_is_refused(self):
        response = self.engine.response("When should I water Peace Lily?")
        self.assertEqual(response["status"], "refused")
        self.assertEqual(response["reason_code"], "conflicted_entity")
        self.assertIn("Watering", response["conflict"]["fields"])

    def test_pet_safety_is_refused_before_conflict_answering(self):
        response = self.engine.response("Is Peace Lily toxic to cats?")
        self.assertEqual(response["status"], "refused")
        self.assertEqual(response["reason_code"], "unsupported_dimension")
        self.assertEqual(response["dimension"], "pet_safety")

    def test_unknown_entity_is_not_replaced_with_arbitrary_search_result(self):
        response = self.engine.response("When should I water my imaginary fern?")
        self.assertEqual(response["status"], "refused")
        self.assertEqual(response["reason_code"], "unknown_entity")
        self.assertEqual(response["results"], [])

    def test_explicit_discovery_question_can_return_multiple_plants(self):
        response = self.engine.response(
            "Which plants need indirect sunlight?", top_k=3
        )
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["reason_code"], "supported_discovery")
        self.assertEqual(response["dimension"], "lighting")
        self.assertEqual(len(response["results"]), 3)
        self.assertTrue(
            all(result["dimension"] == "lighting" for result in response["results"])
        )

    def test_entity_tokens_do_not_leak_into_dimension_routing(self):
        response = self.engine.response("How quickly does Water Lily grow?")
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["dimension"], "growth")
        self.assertEqual(response["results"][0]["plant_name"], "Water Lily")

    def test_empty_query_is_refused(self):
        response = self.engine.response("   ")
        self.assertEqual(response["status"], "refused")
        self.assertEqual(response["reason_code"], "empty_query")

    def test_unsupported_tokens_use_word_boundaries(self):
        self.assertIsNone(
            detect_unsupported_dimension("How should I water a dogwood hybrid?")
        )
        self.assertEqual(
            detect_unsupported_dimension("Is this safe for my dog?"),
            "pet_safety",
        )

    def test_entity_linking_uses_word_boundaries(self):
        self.assertIsNone(
            self.engine.link_entity("What plant food should I use instead?")
        )

    def test_top_k_is_bounded(self):
        with self.assertRaises(ValueError):
            self.engine.response("Which plants need light?", top_k=0)
        with self.assertRaises(ValueError):
            self.engine.response("Which plants need light?", top_k=21)


if __name__ == "__main__":
    unittest.main()
