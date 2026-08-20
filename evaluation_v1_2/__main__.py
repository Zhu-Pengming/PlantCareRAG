"""Unified CLI for the verified-evidence PlantCareRAG pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest

from evaluation_v1_2.rag_engine import (
    DeepSeekJSONGenerator,
    GroundedRAGEngine,
    LiveSemanticDimensionRouter,
)


ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluation_v1_2")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="answer from human-verified evidence")
    ask.add_argument("query")
    ask.add_argument(
        "--semantic",
        action="store_true",
        help="use the frozen BGE lexical-first semantic fallback",
    )
    ask.add_argument("--json", action="store_true", help="print the full response contract")
    ask.add_argument(
        "--generator",
        choices=("extractive", "deepseek"),
        default="extractive",
        help="answer generator; DeepSeek remains optional",
    )
    ask.add_argument(
        "--model",
        help="DeepSeek model ID (or set DEEPSEEK_MODEL); never inferred by the CLI",
    )
    commands.add_parser("evaluate", help="rebuild the frozen-route answer contract result")
    commands.add_parser("verify", help="run offline validators and unit tests")
    return parser


def ask_command(args: argparse.Namespace) -> int:
    try:
        router = LiveSemanticDimensionRouter() if args.semantic else None
    except (ImportError, RuntimeError) as exc:
        print(f"semantic router unavailable: {exc}", file=sys.stderr)
        print(
            "Install evaluation_v1_2/requirements-embedding.txt in a Python 3.10+ environment.",
            file=sys.stderr,
        )
        return 2
    generator = None
    if args.generator == "deepseek":
        model = args.model or os.environ.get("DEEPSEEK_MODEL")
        if not model:
            print(
                "DeepSeek generation requires --model or DEEPSEEK_MODEL.",
                file=sys.stderr,
            )
            return 2
        try:
            generator = DeepSeekJSONGenerator(model=model)
        except (ImportError, RuntimeError, ValueError) as exc:
            print(f"DeepSeek generator unavailable: {exc}", file=sys.stderr)
            print(
                "Use Python 3.10+ and install evaluation_v1_2/requirements-llm.txt.",
                file=sys.stderr,
            )
            return 2
    response = GroundedRAGEngine(router=router, generator=generator).response(args.query)
    if args.json:
        print(json.dumps(response, ensure_ascii=False, indent=2))
    else:
        print(f"status: {response['status']}")
        print(f"route: {response['route']}")
        print(
            f"generator: {response['generation']['mode']}"
            + (" (fallback)" if response["generation"].get("fallback") else "")
        )
        print(response["answer"])
        if response["evidence"]:
            print("\nSources:")
            seen = set()
            for item in response["evidence"]:
                citation = item["citation"]
                key = (citation["url"], citation["section"], citation["paragraph"])
                if key in seen:
                    continue
                seen.add(key)
                print(
                    f"- {citation['source_title']}: {citation['url']} "
                    f"({citation['section']}; {citation['paragraph']})"
                )
    return 0 if response["status"] == "answered" else 2


def evaluate_command() -> int:
    from evaluation_v1_2.scripts.evaluate_grounded_answers import main

    return main()


def verify_command() -> int:
    from evaluation_v1_2.scripts.validate_evidence_overlay import validate as validate_evidence
    from evaluation_v1_2.scripts.validate_grounded_answer_experiment import validate as validate_answers
    from evaluation_v1_2.scripts.validate_semantic_experiment import validate as validate_semantic
    from evaluation_v1_2.scripts.validate_synthetic_benchmark import validate as validate_benchmark

    validations = {
        "evidence_overlay": validate_evidence()[0],
        "synthetic_benchmark": validate_benchmark(),
        "semantic_experiment": validate_semantic(),
        "grounded_answers": validate_answers(),
    }
    failed = False
    for name, errors in validations.items():
        print(f"{name}: {'PASS' if not errors else 'FAIL'}")
        for error in errors:
            print(f"  - {error}")
        failed |= bool(errors)
    if failed:
        return 1
    evaluation_suite = unittest.TestLoader().discover(
        str(ROOT / "tests"), pattern="test_*.py"
    )
    dataset_suite = unittest.TestLoader().discover(
        str(ROOT.parent / "dataset_v1" / "tests"), pattern="test_*.py"
    )
    suite = unittest.TestSuite((evaluation_suite, dataset_suite))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ask":
        return ask_command(args)
    if args.command == "evaluate":
        return evaluate_command()
    return verify_command()


if __name__ == "__main__":
    raise SystemExit(main())
