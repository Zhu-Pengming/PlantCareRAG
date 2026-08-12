import unittest

from evaluation_v1_2.answerability import AnswerabilityEngine, classify_dimensions, extract_qualifiers
from evaluation_v1_2.scripts.evaluate_contract import evaluate
from evaluation_v1_2.scripts.validate_contract import validate


class AnswerabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AnswerabilityEngine()

    def test_multi_dimension_answer_binds_each_claim_to_evidence(self):
        response = self.engine.response(
            "Does Snake Plant need indirect light and frequent watering?"
        )
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["dimensions"], ["watering", "lighting"])
        self.assertEqual(len(response["claims"]), 2)
        self.assertTrue(all(claim["evidence_ids"] for claim in response["claims"]))

    def test_qualified_question_does_not_promote_generic_evidence(self):
        response = self.engine.response(
            "Should I fertilize Snake Plant during winter?"
        )
        self.assertEqual(response["status"], "insufficient_evidence")
        self.assertEqual(response["reason_code"], "missing_qualified_evidence")
        self.assertEqual(response["qualifiers"], {"season": "winter"})
        self.assertEqual(response["claims"], [])

    def test_vague_single_turn_question_requests_entity(self):
        response = self.engine.response("My plant is dying.")
        self.assertEqual(response["status"], "clarification_needed")
        self.assertEqual(response["reason_code"], "missing_entity_context")

    def test_unsupported_safety_question_stays_unsupported(self):
        response = self.engine.response("Is Snake Plant toxic to cats?")
        self.assertEqual(response["status"], "unsupported")
        self.assertEqual(response["dimensions"], ["pet_safety"])
        self.assertEqual(response["claims"], [])

    def test_conflicted_entity_is_reported_not_answered(self):
        response = self.engine.response("When should I water Peace Lily?")
        self.assertEqual(response["status"], "conflicting_evidence")
        self.assertIn("conflict", response)

    def test_dimension_and_qualifier_extractors_are_multilabel(self):
        self.assertEqual(
            classify_dimensions("light and water in winter"),
            ["watering", "lighting"],
        )
        self.assertEqual(
            extract_qualifiers("light and water in winter"),
            {"season": "winter"},
        )

    def test_seed_contract_is_valid_and_passes(self):
        self.assertEqual(validate(), [])
        result = evaluate()
        self.assertFalse(result["benchmark_eligible"])
        self.assertEqual(result["summary"]["fail"], 0)


if __name__ == "__main__":
    unittest.main()
