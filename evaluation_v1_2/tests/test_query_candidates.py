from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation_v1_2.scripts.build_query_candidates import alias_pattern, plain_text
from evaluation_v1_2.scripts.screen_query_candidates import validate as validate_screening
from evaluation_v1_2.scripts.validate_query_candidates import validate
from evaluation_v1_2.scripts.validate_query_shortlist import validate as validate_shortlist


ROOT = Path(__file__).resolve().parents[1]


class QueryCandidateTests(unittest.TestCase):
    def test_plain_text_removes_markup(self):
        self.assertEqual(plain_text("<p>Why <strong>yellow</strong>?</p>"), "Why yellow ?")

    def test_alias_pattern_has_word_boundaries(self):
        pattern = alias_pattern("hoya")
        self.assertIsNotNone(pattern.search("My Hoya is yellow"))
        self.assertIsNone(pattern.search("choyangensis"))

    def test_checked_in_candidates_are_valid(self):
        self.assertEqual(validate(), [])

    def test_empty_screening_file_is_valid(self):
        candidates = json.loads((ROOT / "data" / "query_candidates.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_screening(candidates, []), [])

    def test_automated_shortlist_is_balanced_and_non_gold(self):
        self.assertEqual(validate_shortlist(), [])
        rows = json.loads((ROOT / "data" / "query_shortlist.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 100)
        self.assertTrue(all("gold_evidence_ids" not in row for row in rows))
        self.assertTrue(all(row["benchmark_eligible"] is False for row in rows))

    def test_validator_rejects_eligible_unreviewed_row(self):
        rows = json.loads((ROOT / "data" / "query_candidates.json").read_text(encoding="utf-8"))
        row = dict(rows[0])
        row["benchmark_eligible"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps([row]), encoding="utf-8")
            self.assertTrue(any("marked eligible" in error for error in validate(path)))


if __name__ == "__main__":
    unittest.main()
