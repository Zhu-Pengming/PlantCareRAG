#!/usr/bin/env python3
"""Ask the dataset-backed Plant RAG v1 from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset_v1.query_engine import QueryEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Plant-care question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 20:
        parser.error("--top-k must be between 1 and 20")

    response = QueryEngine().response(args.query, top_k=args.top_k)
    if args.as_json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0

    print(f"status: {response['status']} ({response['reason_code']})")
    print(response["answer"])
    for result in response["results"]:
        citation = result["citation"]
        print(
            f"- {result['plant_name']} [{result['dimension']}]: {result['value']} "
            f"(rows {citation['raw_row_numbers']}, {citation['dataset_page']})"
        )
    for warning in response["warnings"]:
        print(f"warning: {warning}")
    return 0 if response["status"] == "answered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
