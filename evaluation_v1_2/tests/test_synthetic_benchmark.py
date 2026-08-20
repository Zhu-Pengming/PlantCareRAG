from __future__ import annotations

import unittest

from evaluation_v1_2.scripts.evaluate_synthetic_benchmark import evaluate
from evaluation_v1_2.scripts.validate_synthetic_benchmark import validate


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_generated_benchmark_is_valid(self):
        self.assertEqual(validate(), [])

    def test_frozen_baseline_exposes_implicit_paraphrase_gap(self):
        result = evaluate()
        self.assertEqual(result["overall"]["n"], 80)
        self.assertEqual(result["overall"]["entity_accuracy"], 1.0)
        self.assertEqual(result["failure_counts"], {
            "dimension": 15,
            "behavior": 15,
            "evidence": 15,
        })
        self.assertEqual(
            result["by_generation_style"]["v3"]["dimension_exact_match"],
            7 / 22,
        )


if __name__ == "__main__":
    unittest.main()
