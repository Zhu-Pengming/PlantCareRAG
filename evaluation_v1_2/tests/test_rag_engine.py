import unittest
import json
from types import SimpleNamespace

from evaluation_v1_2.rag_engine import (
    DeepSeekJSONGenerator,
    GroundedRAGEngine,
    RouteDecision,
    validate_grounded_response,
)
from evaluation_v1_2.scripts.evaluate_grounded_answers import evaluate


class FakeCompletions:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        if self.error:
            raise self.error
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], id="resp_fake"
        )


class FakeDeepSeekClient:
    def __init__(self, payload=None, error=None):
        self.chat = SimpleNamespace(completions=FakeCompletions(payload, error))


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
            "answer contains text outside the structured cited claims",
            validate_grounded_response(response),
        )

    def test_deepseek_json_generator_returns_citation_bound_paraphrase(self):
        client = FakeDeepSeekClient(
            {
                "sentences": [
                    {
                        "text": "Choose very well-drained soil made for succulents.",
                        "evidence_ids": ["claim:aloe_vera:soil:001"],
                    }
                ]
            }
        )
        generator = DeepSeekJSONGenerator(model="test-model", client=client)
        engine = GroundedRAGEngine(generator=generator)
        response = engine.response(
            "What soil should I use for Aloe Vera?",
            route_decision=RouteDecision(("soil",), "test_route"),
        )
        self.assertEqual(response["generation"]["mode"], "llm_json")
        self.assertFalse(response["generation"]["fallback"])
        self.assertEqual(validate_grounded_response(response), [])
        request = client.chat.completions.last_request
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertIn("JSON", request["messages"][0]["content"])

    def test_invalid_llm_number_falls_back_to_extractive_answer(self):
        client = FakeDeepSeekClient(
            {
                "sentences": [
                    {
                        "text": "Water every 2 days.",
                        "evidence_ids": ["claim:aloe_vera:watering:001"],
                    },
                    {
                        "text": "Reduce watering in winter.",
                        "evidence_ids": ["claim:aloe_vera:watering:002"],
                    },
                ]
            }
        )
        engine = GroundedRAGEngine(
            generator=DeepSeekJSONGenerator(model="test-model", client=client)
        )
        response = engine.response(
            "How should I water Aloe Vera?",
            route_decision=RouteDecision(("watering",), "test_route"),
        )
        self.assertEqual(response["generation"]["mode"], "extractive")
        self.assertTrue(response["generation"]["fallback"])
        self.assertIn("adds a number", response["generation"]["fallback_reason"])
        self.assertEqual(validate_grounded_response(response), [])

    def test_deepseek_transport_failure_falls_back_without_error_text_leak(self):
        client = FakeDeepSeekClient(error=RuntimeError("secret transport detail"))
        engine = GroundedRAGEngine(
            generator=DeepSeekJSONGenerator(model="test-model", client=client)
        )
        response = engine.response(
            "What soil should I use for Aloe Vera?",
            route_decision=RouteDecision(("soil",), "test_route"),
        )
        self.assertEqual(response["generation"]["fallback_reason"], "RuntimeError")
        self.assertNotIn("secret transport detail", str(response))
        self.assertEqual(validate_grounded_response(response), [])

    def test_empty_deepseek_json_output_falls_back_to_extractive(self):
        client = FakeDeepSeekClient(payload="")
        engine = GroundedRAGEngine(
            generator=DeepSeekJSONGenerator(model="test-model", client=client)
        )
        response = engine.response(
            "What soil should I use for Aloe Vera?",
            route_decision=RouteDecision(("soil",), "test_route"),
        )
        self.assertEqual(response["generation"]["mode"], "extractive")
        self.assertTrue(response["generation"]["fallback"])
        self.assertEqual(response["generation"]["fallback_reason"], "RuntimeError")

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
