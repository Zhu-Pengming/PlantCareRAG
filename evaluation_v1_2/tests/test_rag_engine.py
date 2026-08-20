import unittest

from evaluation_v1_2.rag_engine import (
    GroundedRAGEngine,
    RouteDecision,
    validate_grounded_response,
)
from evaluation_v1_2.scripts.evaluate_grounded_answers import evaluate


class GroundedRAGEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = GroundedRAGEngine()

    def test_answer_uses_only_verified_claims_and_inline_citations(self):
        response = self.engine.response("How should I water Snake Plant?")
        self.assertEqual(response["status"], "answered")
        self.assertEqual(validate_grounded_response(response), [])
        self.assertTrue(response["evidence"])
        self.assertTrue(
            all(item["review_status"] == "human_verified" for item in response["evidence"])
        )
        for item in response["evidence"]:
            self.assertIn(f"[{item['claim_id']}]", response["answer"])

    def test_unsupported_safety_query_never_exposes_evidence(self):
        response = self.engine.response("Is Aloe Vera pet safe?")
        self.assertEqual(response["status"], "unsupported")
        self.assertEqual(response["evidence"], [])
        self.assertEqual(response["claims"], [])

    def test_unresolved_common_name_mapping_requests_scientific_name(self):
        response = self.engine.response("What light does Monstera need?")
        self.assertEqual(response["status"], "clarification_needed")
        self.assertEqual(response["reason_code"], "unresolved_common_name_mapping")

    def test_explicit_route_can_exercise_implicit_query_without_model_download(self):
        response = self.engine.response(
            "What should I fill the pot with for Aloe Vera?",
            route_decision=RouteDecision(("soil",), "test_route"),
        )
        self.assertEqual(response["status"], "answered")
        self.assertEqual(response["dimensions"], ["soil"])
        self.assertEqual(validate_grounded_response(response), [])

    def test_response_validator_rejects_unbound_claim(self):
        response = self.engine.response("How should I water Snake Plant?")
        response["claims"][0]["text"] = "Water every day."
        self.assertTrue(validate_grounded_response(response))

    def test_response_validator_rejects_extra_uncited_sentence(self):
        response = self.engine.response("How should I water Snake Plant?")
        response["answer"] += "\nWater every day."
        self.assertIn(
            "answer contains text outside the cited extractive claims",
            validate_grounded_response(response),
        )

    def test_multi_dimension_route_requires_complete_dimension_coverage(self):
        response = self.engine.response(
            "What fertilizer and soil guidance should I follow for Snake Plant?",
            route_decision=RouteDecision(("fertilizer", "soil"), "test_route"),
        )
        self.assertEqual(response["status"], "insufficient_evidence")
        self.assertEqual(response["reason_code"], "partial_dimension_coverage")
        self.assertEqual(response["evidence"], [])

    def test_frozen_routes_produce_fully_grounded_contract_results(self):
        result = evaluate()
        self.assertEqual(result["overall"]["grounded_response_rate"], 1.0)
        self.assertEqual(result["overall"]["answered_citation_validity"], 1.0)
        self.assertEqual(result["overall"]["unsafe_answer_count"], 0)


if __name__ == "__main__":
    unittest.main()
