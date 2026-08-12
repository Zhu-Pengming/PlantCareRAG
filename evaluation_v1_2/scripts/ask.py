#!/usr/bin/env python3
"""Inspect the v1.2 answerability and evidence contract from the CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_v1_2.answerability import AnswerabilityEngine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    response = AnswerabilityEngine().response(args.query, top_k=args.top_k)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response["status"] == "answered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
