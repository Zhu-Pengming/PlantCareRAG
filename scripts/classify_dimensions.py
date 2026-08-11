#!/usr/bin/env python3
"""Classify query dimensions with a leakage-controlled few-shot LLM prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_baseline import MAIN_PATH, ROOT, load_json
except ModuleNotFoundError:  # Support direct execution: python3 scripts/classify_dimensions.py
    from evaluate_baseline import MAIN_PATH, ROOT, load_json


DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "dimension_predictions.json"
DEFAULT_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
ALLOWED_DIMENSIONS = [
    "taxonomy",
    "lighting",
    "watering",
    "humidity",
    "temperature",
    "symptom",
    "disease",
    "pest",
    "pet_safety",
    "repotting",
]
PROMPT_VERSIONS = {
    "conservative": "1.0.0",
    "recall_first": "1.1.0",
}


def build_prompt(
    queries: list[dict[str, Any]], prompt_style: str = "conservative"
) -> str:
    query_payload = [
        {"query_id": query["query_id"], "raw_text": query["raw_text"]}
        for query in queries
    ]
    conservative_prompt = f"""You are a multi-label intent classifier for an English houseplant question retrieval system.

Assign every query one or more dimensions from this closed label set:
- taxonomy: identity, names, synonyms, or comparison between plant identities
- lighting: sun, shade, windows, grow lights, or light placement
- watering: watering amount/frequency/method, wet or dry soil, drainage, or water as a suspected cause
- humidity: humidity, dry air, bathrooms, misting, or humidifiers
- temperature: heat, cold, drafts, heaters, air conditioning, or temperature limits
- symptom: visible damage, discoloration, spots, softness, rot, decline, or diagnosis of what is wrong
- disease: pathogens, fungus, bacterial disease, or disease as a suspected cause
- pest: insects, mites, or pests as a suspected cause
- pet_safety: toxicity or safety for cats, dogs, or other pets
- repotting: changing pots, disturbing roots, or replacing potting mix as a requested action

Label the information needed to answer the query, not every remotely related concept. Preserve multiple labels for compound questions. Do not infer a missing plant identity and do not answer the question. Return exactly one prediction for every query_id.

The examples below are synthetic and are not part of the evaluation set.

Examples:
- "What is the botanical name for this plant?" -> ["taxonomy"]
- "Can it live beside a shaded north-facing window?" -> ["lighting"]
- "Should I water every week or wait until the mix dries?" -> ["watering"]
- "Will hot, dry heater air damage the leaves?" -> ["humidity", "temperature"]
- "The lower leaves are yellow and the soil has stayed wet for days—what is wrong?" -> ["symptom", "watering"]
- "Are these spreading spots caused by fungus or insects?" -> ["symptom", "disease", "pest"]
- "My cat chewed a leaf; is this plant toxic?" -> ["pet_safety"]
- "The roots fill the pot; should I move it to a larger container?" -> ["repotting"]

Queries:
{json.dumps(query_payload, ensure_ascii=False, indent=2)}
"""
    if prompt_style == "conservative":
        return conservative_prompt
    if prompt_style != "recall_first":
        raise ValueError(f"unsupported prompt_style: {prompt_style}")

    recall_policy = """Label every dimension that may be needed to answer the query. This output is a retrieval filter: a missing label permanently removes relevant evidence, while an extra plausible label only widens the candidate set. Optimize for recall. When uncertain whether a borderline dimension is needed, include it. Preserve multiple labels for compound questions. Treat wet/damp soil, slow drying, soggy tissue, or suspected overwatering/underwatering as watering evidence even when the user does not literally say \"water\". Do not infer a missing plant identity and do not answer the question. Return exactly one prediction for every query_id."""
    conservative_policy = """Label the information needed to answer the query, not every remotely related concept. Preserve multiple labels for compound questions. Do not infer a missing plant identity and do not answer the question. Return exactly one prediction for every query_id."""
    borderline_examples = """Additional recall-oriented borderline examples:
- \"The potting mix is still damp and a leaf has collapsed—what is happening?\" -> [\"symptom\", \"watering\"]
- \"The leaves developed pale patches after I moved it away from the window, and the pot now dries very slowly.\" -> [\"lighting\", \"symptom\", \"watering\"]

"""
    return conservative_prompt.replace(conservative_policy, recall_policy).replace(
        "Queries:\n",
        f"{borderline_examples}Queries:\n",
        1,
    )


def output_schema(query_count: int) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["predictions"],
        "properties": {
            "predictions": {
                "type": "array",
                "minItems": query_count,
                "maxItems": query_count,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query_id", "dimensions"],
                    "properties": {
                        "query_id": {"type": "string"},
                        "dimensions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"enum": ALLOWED_DIMENSIONS},
                        },
                    },
                },
            }
        },
    }


def validate_predictions(payload: dict[str, Any], queries: list[dict[str, Any]]) -> None:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("model output is missing predictions[]")
    expected_ids = {query["query_id"] for query in queries}
    predicted_ids = [prediction.get("query_id") for prediction in predictions]
    if len(predicted_ids) != len(set(predicted_ids)):
        raise ValueError("model output contains duplicate query_id values")
    if set(predicted_ids) != expected_ids:
        missing = sorted(expected_ids - set(predicted_ids))
        extra = sorted(set(predicted_ids) - expected_ids)
        raise ValueError(f"model output query IDs do not match input; missing={missing}, extra={extra}")
    allowed = set(ALLOWED_DIMENSIONS)
    for prediction in predictions:
        dimensions = prediction.get("dimensions")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"{prediction.get('query_id')}: dimensions must be non-empty")
        if len(dimensions) != len(set(dimensions)):
            raise ValueError(f"{prediction.get('query_id')}: dimensions must be unique")
        invalid = set(dimensions) - allowed
        if invalid:
            raise ValueError(f"{prediction.get('query_id')}: invalid dimensions {sorted(invalid)}")


def classify_with_codex(
    queries: list[dict[str, Any]],
    model: str,
    reasoning_effort: str,
    codex_binary: Path,
    prompt_style: str,
) -> dict[str, Any]:
    prompt = build_prompt(queries, prompt_style=prompt_style)
    with tempfile.TemporaryDirectory(prefix="plant-dimension-classifier-") as directory:
        temp_root = Path(directory)
        schema_path = temp_root / "output_schema.json"
        result_path = temp_root / "result.json"
        schema_path.write_text(
            json.dumps(output_schema(len(queries)), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        command = [
            str(codex_binary),
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            str(temp_root),
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(result_path),
            "-",
        ]
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()[-2000:]
            raise RuntimeError(f"Codex classifier failed with exit {completed.returncode}: {stderr}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))

    validate_predictions(payload, queries)
    predictions = sorted(payload["predictions"], key=lambda item: item["query_id"])
    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "codex_cli",
        "codex_binary": str(codex_binary),
        "model": model,
        "model_documentation_url": f"https://developers.openai.com/api/docs/models/{model}",
        "model_documentation_accessed_at": date.today().isoformat(),
        "reasoning_effort": reasoning_effort,
        "prompt_style": prompt_style,
        "prompt_version": PROMPT_VERSIONS[prompt_style],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "label_source": "model_prediction_without_expected_dimensions",
        "few_shot_source": "synthetic_examples_not_present_in_evaluation_set",
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument(
        "--prompt-style",
        choices=sorted(PROMPT_VERSIONS),
        default="conservative",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="low",
    )
    parser.add_argument("--codex-binary", type=Path, default=DEFAULT_CODEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.codex_binary.exists():
        parser.error(f"Codex binary not found: {args.codex_binary}")

    queries = load_json(MAIN_PATH)
    result = classify_with_codex(
        queries,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        codex_binary=args.codex_binary,
        prompt_style=args.prompt_style,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dimension predictions: {args.output}")
    print(
        f"Model: {result['model']} effort={result['reasoning_effort']} "
        f"queries={len(result['predictions'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
