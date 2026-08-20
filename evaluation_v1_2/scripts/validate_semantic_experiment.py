#!/usr/bin/env python3
"""Validate frozen semantic-routing experiment metadata and results."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def validate() -> list[str]:
    errors: list[str] = []
    config = json.loads((ROOT / "config" / "semantic_dimension_router.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "results" / "semantic_dimension_router.json").read_text(encoding="utf-8"))
    benchmark = json.loads((ROOT / "reports" / "synthetic_benchmark_stats.json").read_text(encoding="utf-8"))
    if config.get("selection_split") != "dev":
        errors.append("threshold was not selected on dev")
    if config.get("test_accessed") is not True:
        errors.append("final test evaluation is not recorded")
    if config.get("threshold") != result.get("threshold"):
        errors.append("result threshold differs from frozen config")
    if config.get("resolved_model_revision") != result.get("resolved_model_revision"):
        errors.append("result model revision differs from frozen config")
    if benchmark.get("benchmark_sha256") != result.get("benchmark_sha256"):
        errors.append("semantic result used a different benchmark revision")
    if result.get("by_split", {}).get("dev", {}).get("n") != 45:
        errors.append("semantic dev denominator is not 45")
    if result.get("by_split", {}).get("test", {}).get("n") != 35:
        errors.append("semantic test denominator is not 35")
    query_ids = [row.get("query_id") for row in result.get("queries", [])]
    if len(query_ids) != 80 or len(set(query_ids)) != 80:
        errors.append("semantic result does not contain 80 unique queries")
    if config.get("test_dimension_exact_match") != result["by_split"]["test"]["dimension_exact_match"]:
        errors.append("config test result does not match result artifact")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        print(f"semantic experiment invalid ({len(errors)} errors):", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("semantic experiment valid (dev-tuned threshold; frozen 35-query test result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
