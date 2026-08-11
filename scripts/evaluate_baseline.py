#!/usr/bin/env python3
"""Run a dependency-free BM25 baseline over the curated Plant RAG corpus."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.validate_kb import DATA_ROOT, ROOT, load_json
except ModuleNotFoundError:  # Support direct execution: python3 scripts/evaluate_baseline.py
    from validate_kb import DATA_ROOT, ROOT, load_json


EVAL_ROOT = ROOT / "evaluation"
MAIN_PATH = EVAL_ROOT / "data" / "main_queries.json"
ABSTAIN_PATH = EVAL_ROOT / "data" / "abstain_queries.json"
DEFAULT_REPORT = EVAL_ROOT / "results" / "baseline.json"
QUESTION_COLUMNS = {
    "direct": "A_direct",
    "alias_or_coreference": "B_alias_or_coreference",
    "symptom": "C_symptom",
    "compound": "D_compound",
    "judgment_or_premise": "E_judgment_or_premise",
}
DIMENSION_LABELS_ZH = {
    "taxonomy": "分类/名称",
    "lighting": "光照",
    "watering": "浇水",
    "humidity": "湿度",
    "temperature": "温度",
    "symptom": "症状诊断",
    "disease": "病害",
    "pest": "虫害",
    "pet_safety": "宠物安全",
    "repotting": "换盆",
    "fertilizer": "施肥",
    "propagation": "繁殖",
    "plant_support": "植株支撑",
}


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = re.sub(r"[-_/']+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    tokens: list[str] = []
    for word in re.findall(r"[a-z0-9]+", text):
        tokens.append(word)
        stem = stem_english(word)
        if stem != word:
            tokens.append(stem)
    for sequence in re.findall(r"[\u3400-\u9fff]+", text):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


def stem_english(word: str) -> str:
    irregular = {"leaves": "leaf", "wives": "wife"}
    if word in irregular:
        return irregular[word]
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from flatten_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_strings(item)


@dataclass(frozen=True)
class Document:
    entry_id: str
    plant_id: str
    dimension: str
    text: str


class BM25:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(document.text) for document in documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def rank(self, query: str, plant_id: str | None = None) -> list[tuple[Document, float]]:
        query_terms = tokenize(query)
        ranked: list[tuple[Document, float]] = []
        for index, document in enumerate(self.documents):
            if plant_id is not None and document.plant_id != plant_id:
                continue
            score = 0.0
            frequencies = self.term_frequencies[index]
            length = self.lengths[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average_length, 1)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            ranked.append((document, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0].entry_id))


class EntityNormalizer:
    def __init__(self, aliases: list[dict[str, Any]]):
        self.aliases = [
            alias
            for alias in aliases
            if alias.get("automatic") is True and alias.get("review_status") == "approved"
        ]
        self.aliases.sort(key=lambda alias: len(alias["normalized_value"]), reverse=True)
        self.expansions: dict[str, list[str]] = defaultdict(list)
        for alias in self.aliases:
            self.expansions[alias["plant_id"]].append(alias["value"])

    def detect(self, text: str) -> str | None:
        normalized = normalize_text(text)
        matched_plants: set[str] = set()
        for alias in self.aliases:
            alias_value = normalize_text(alias["normalized_value"])
            if re.search(r"[a-z0-9]", alias_value):
                plural = "" if alias_value.endswith("s") else "s?"
                if re.search(rf"(?<![a-z0-9]){re.escape(alias_value)}{plural}(?![a-z0-9])", normalized):
                    matched_plants.add(alias["plant_id"])
            elif alias_value in normalized:
                matched_plants.add(alias["plant_id"])
        return next(iter(matched_plants)) if len(matched_plants) == 1 else None

    def expand(self, text: str, plant_id: str | None) -> str:
        if plant_id is None:
            return text
        return " ".join([text, *self.expansions[plant_id]])


class DimensionNormalizer:
    """Small, inspectable lexical baseline for query-to-dimension linking."""

    RULES = {
        "taxonomy": (
            r"\bdifference\b", r"\breclassif", r"\bnickname\b", r"\bscientific name\b",
            r"区别", r"学名", r"叫什么", r"别名",
        ),
        "lighting": (
            r"\blight\b", r"\bwindow\b", r"\bsun\b", r"\bshade\b", r"光照", r"阳光", r"晒太阳", r"窗边",
        ),
        "watering": (
            r"\bwater", r"\bsoil\b", r"\bdry out\b", r"浇水", r"土干", r"土湿", r"积水",
        ),
        "humidity": (r"\bhumid", r"\bbathroom\b", r"湿度", r"加湿", r"卫生间", r"浴室"),
        "temperature": (r"\btemperature\b", r"\bhot\b", r"\bcold\b", r"温度", r"空调", r"暖气"),
        "symptom": (
            r"\byellow", r"\bbrown", r"\bblack spot", r"\brot", r"\bsoft\b", r"\bfall", r"\bwrong\b",
            r"黄叶", r"发黄", r"黑斑", r"褐斑", r"烂根", r"发软", r"枯", r"怎么了",
        ),
        "disease": (r"\bdisease\b", r"\bfung", r"病害", r"真菌"),
        "pest": (r"\bpest\b", r"\bbug\b", r"虫害", r"虫子"),
        "pet_safety": (r"\bcat", r"\bdog", r"\bpet\b", r"\btoxic\b", r"\bpoison", r"猫", r"狗", r"宠物", r"有毒"),
        "repotting": (r"\brepot", r"换盆", r"移栽"),
        "fertilizer": (r"\bfertiliz", r"\bplant food\b", r"施肥", r"肥料"),
        "propagation": (r"\bpropagat", r"\bcutting\b", r"扦插", r"繁殖"),
        "plant_support": (r"\bstak", r"\bmoss pole\b", r"支撑", r"爬杆"),
    }

    def detect(self, text: str) -> set[str]:
        normalized = normalize_text(text)
        return {
            dimension
            for dimension, patterns in self.RULES.items()
            if any(re.search(pattern, normalized) for pattern in patterns)
        }


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted((DATA_ROOT / "entries").glob("*.json")):
        for entry in load_json(path):
            index_payload = {
                "content": entry["content"],
                "dimension": entry["dimension"],
                "topics": entry["topics"],
                "conditions": entry["conditions"],
                "retrieval": entry["retrieval"],
            }
            documents.append(
                Document(
                    entry_id=entry["id"],
                    plant_id=entry["plant_id"],
                    dimension=entry["dimension"],
                    text=" ".join(flatten_strings(index_payload)),
                )
            )
    return documents


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def ratio_metric(numerator: float, denominator: int, numerator_label: str) -> dict[str, Any]:
    return {
        "value": round(safe_ratio(numerator, denominator), 6),
        numerator_label: round(numerator, 6),
        "n": denominator,
    }


def summarize_retrieval(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    n = len(rows)
    gold_sizes = Counter(row["gold_size"] for row in rows)
    return {
        "eligible_queries": n,
        "gold_size_distribution": {str(size): count for size, count in sorted(gold_sizes.items())},
        f"hit_at_{k}": ratio_metric(sum(row[f"hit_at_{k}"] for row in rows), n, "hits"),
        f"recall_at_{k}_macro": ratio_metric(
            sum(row[f"recall_at_{k}"] for row in rows), n, "sum_query_recall"
        ),
        f"recall_at_{k}_ceiling_macro": ratio_metric(
            sum(min(k, row["gold_size"]) / row["gold_size"] for row in rows),
            n,
            "sum_query_ceiling",
        ),
        "recall_at_5_macro": ratio_metric(
            sum(row["recall_at_5"] for row in rows), n, "sum_query_recall"
        ),
        "recall_at_gold_macro": ratio_metric(
            sum(row["recall_at_gold"] for row in rows), n, "sum_query_recall"
        ),
        "mrr": ratio_metric(sum(row["reciprocal_rank"] for row in rows), n, "sum_reciprocal_rank"),
    }


def summarize_retrieval_population(
    rows: list[dict[str, Any]], queries: list[dict[str, Any]], k: int
) -> dict[str, Any]:
    """Summarize a declared query population without hiding its denominator."""
    included_ids = {query["query_id"] for query in queries}
    included_rows = [row for row in rows if row["query_id"] in included_ids]
    by_column: dict[str, dict[str, Any]] = {}
    for question_type, column in QUESTION_COLUMNS.items():
        column_rows = [row for row in included_rows if row["question_column"] == column]
        column_summary = summarize_retrieval(column_rows, k)
        column_summary["total_queries"] = sum(
            query["question_type"] == question_type for query in queries
        )
        column_summary["queries_without_gold"] = (
            column_summary["total_queries"] - column_summary["eligible_queries"]
        )
        by_column[column] = column_summary
    return {
        "aggregate": summarize_retrieval(included_rows, k),
        "by_question_column": by_column,
        "per_query": included_rows,
    }


def evaluate_structural_gate(
    documents: list[Document],
    normalizer: EntityNormalizer,
    dimension_normalizer: DimensionNormalizer,
    plant_labels: dict[str, str],
    main_queries: list[dict[str, Any]],
    abstain_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    covered_pairs = {(document.plant_id, document.dimension) for document in documents}

    def decision(query: dict[str, Any]) -> dict[str, Any]:
        plant_id = normalizer.detect(query["raw_text"])
        if plant_id is None:
            return {
                "query_id": query["query_id"],
                "abstained": True,
                "reason": "entity_unresolved_or_ambiguous",
                "detected_plant_id": None,
                "detected_dimensions": [],
                "missing_dimensions": [],
                "explanation": "无法唯一识别植物，或该植物不在当前知识库范围内。",
            }
        dimensions = dimension_normalizer.detect(query["raw_text"])
        if not dimensions:
            return {
                "query_id": query["query_id"],
                "abstained": True,
                "reason": "dimension_unresolved",
                "detected_plant_id": plant_id,
                "detected_dimensions": [],
                "missing_dimensions": [],
                "explanation": f"已识别为{plant_labels[plant_id]}，但无法确定问题所属的知识维度。",
            }
        missing = sorted(
            dimension for dimension in dimensions if (plant_id, dimension) not in covered_pairs
        )
        explanation = None
        if missing:
            missing_labels = "、".join(DIMENSION_LABELS_ZH[dimension] for dimension in missing)
            explanation = f"我还没有{plant_labels[plant_id]}的{missing_labels}资料。"
        return {
            "query_id": query["query_id"],
            "abstained": bool(missing),
            "reason": "missing_plant_dimension" if missing else "covered_plant_dimensions",
            "detected_plant_id": plant_id,
            "detected_dimensions": sorted(dimensions),
            "missing_dimensions": missing,
            "explanation": explanation,
        }

    abstain_details = [decision(query) for query in abstain_queries]
    main_details = [decision(query) for query in main_queries]
    fully_covered = [
        detail
        for query, detail in zip(main_queries, main_details)
        if query["status"] == "covered"
    ]
    context_free_covered = [
        detail
        for query, detail in zip(main_queries, main_details)
        if query["status"] == "covered" and query["context_requirement"] == "none"
    ]
    return {
        "method": "lexical_entity_linking_then_lexical_dimension_linking_then_kb_pair_coverage",
        "abstain_set_accuracy": ratio_metric(
            sum(detail["abstained"] for detail in abstain_details),
            len(abstain_details),
            "correct_abstentions",
        ),
        "covered_query_acceptance": ratio_metric(
            sum(not detail["abstained"] for detail in fully_covered),
            len(fully_covered),
            "accepted",
        ),
        "context_free_covered_query_acceptance": ratio_metric(
            sum(not detail["abstained"] for detail in context_free_covered),
            len(context_free_covered),
            "accepted",
        ),
        "abstain_details": abstain_details,
        "main_details": main_details,
        "limitations": [
            "This is a lexical dimension linker, not an oracle label evaluation.",
            "A missing dimension pair can be explained, but an existing pair may still lack the requested answer shape or fact granularity.",
        ],
    }


def evaluate_mode(
    name: str,
    bm25: BM25,
    normalizer: EntityNormalizer,
    queries: list[dict[str, Any]],
    abstain_queries: list[dict[str, Any]],
    k: int,
    threshold: float,
    entity_aware: bool,
) -> dict[str, Any]:
    eligible = [query for query in queries if query["gold_entry_ids"]]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    entity_total = 0
    entity_correct = 0
    entity_context_free_total = 0
    entity_context_free_correct = 0

    for query in queries:
        detected = normalizer.detect(query["raw_text"])
        if query["plant_id"] is not None:
            entity_total += 1
            entity_correct += int(detected == query["plant_id"])
            if query["context_requirement"] == "none":
                entity_context_free_total += 1
                entity_context_free_correct += int(detected == query["plant_id"])

        if not query["gold_entry_ids"]:
            continue
        ranked = bm25.rank(query["raw_text"], plant_id=detected if entity_aware else None)
        ranked_ids = [document.entry_id for document, _ in ranked]
        top_ids = ranked_ids[:k]
        gold = set(query["gold_entry_ids"])
        matched = gold.intersection(top_ids)
        first_rank = next((index + 1 for index, entry_id in enumerate(ranked_ids) if entry_id in gold), None)
        gold_size = len(gold)
        rows.append(
            {
                "query_id": query["query_id"],
                "question_column": QUESTION_COLUMNS[query["question_type"]],
                "gold_size": gold_size,
                f"hit_at_{k}": int(bool(matched)),
                f"recall_at_{k}": safe_ratio(len(matched), gold_size),
                "recall_at_5": safe_ratio(len(gold.intersection(ranked_ids[:5])), gold_size),
                "recall_at_gold": safe_ratio(
                    len(gold.intersection(ranked_ids[:gold_size])), gold_size
                ),
                "reciprocal_rank": 1 / first_rank if first_rank else 0.0,
                f"all_gold_at_{k}": gold.issubset(top_ids),
                "answer_shape_support": query.get("answer_shape_support", "not_audited"),
            }
        )
        if not matched:
            failures.append(
                {
                    "query_id": query["query_id"],
                    "gold": sorted(gold),
                    "top_k": top_ids,
                    "detected_plant_id": detected,
                }
            )

    abstained = 0
    abstain_details: list[dict[str, Any]] = []
    for query in abstain_queries:
        detected = normalizer.detect(query["raw_text"])
        ranked = bm25.rank(query["raw_text"], plant_id=detected if entity_aware else None)
        top_score = ranked[0][1] if ranked else 0.0
        prediction = top_score < threshold
        abstained += int(prediction)
        abstain_details.append(
            {
                "query_id": query["query_id"],
                "abstained": prediction,
                "top_score": round(top_score, 6),
                "top_entry_id": ranked[0][0].entry_id if ranked else None,
                "detected_plant_id": detected,
            }
        )

    single_turn_queries = [
        query for query in queries if not query.get("excluded_from_single_turn", False)
    ]
    shape_gap_rows = [
        row
        for row in rows
        if row["answer_shape_support"] in {"conditional_substitute", "unsupported"}
    ]
    return {
        "name": name,
        "retrieval": summarize_retrieval_population(rows, queries, k),
        "retrieval_single_turn": summarize_retrieval_population(
            rows, single_turn_queries, k
        ),
        "entity_linking": {
            "overall": ratio_metric(entity_correct, entity_total, "correct"),
            "context_free": ratio_metric(
                entity_context_free_correct, entity_context_free_total, "correct"
            ),
        },
        "score_threshold_abstention": {
            "accuracy": ratio_metric(abstained, len(abstain_queries), "correct_abstentions"),
            "details": abstain_details,
        },
        "answer_shape_analysis": {
            "shape_gap_queries": len(shape_gap_rows),
            f"retrieval_complete_at_{k}": ratio_metric(
                sum(row[f"all_gold_at_{k}"] for row in shape_gap_rows),
                len(shape_gap_rows),
                "queries_with_all_gold_retrieved",
            ),
            "query_ids": [row["query_id"] for row in shape_gap_rows],
        },
        f"retrieval_failures_at_{k}": failures,
    }


def build_report(k: int, threshold: float) -> dict[str, Any]:
    documents = load_documents()
    bm25 = BM25(documents)
    aliases = load_json(DATA_ROOT / "entity_aliases.json")
    normalizer = EntityNormalizer(aliases)
    dimension_normalizer = DimensionNormalizer()
    plants = load_json(DATA_ROOT / "plants.json")
    plant_labels = {
        plant["id"]: next(
            name["name"] for name in plant["names"] if name["language"] == "zh-CN"
        )
        for plant in plants
    }
    queries = load_json(MAIN_PATH)
    abstain_queries = load_json(ABSTAIN_PATH)
    coverage = Counter(query["status"] for query in queries)
    modes = [
        evaluate_mode("bm25_raw", bm25, normalizer, queries, abstain_queries, k, threshold, False),
        evaluate_mode("bm25_entity", bm25, normalizer, queries, abstain_queries, k, threshold, True),
    ]
    return {
        "schema_version": "1.1.0",
        "configuration": {
            "k": k,
            "abstain_threshold": threshold,
            "documents": len(documents),
            "main_queries": len(queries),
            "queries_with_gold": sum(bool(query["gold_entry_ids"]) for query in queries),
            "single_turn_queries": sum(
                not query.get("excluded_from_single_turn", False) for query in queries
            ),
            "excluded_thread_context_queries": sum(
                query.get("excluded_from_single_turn", False) for query in queries
            ),
            "abstain_queries": len(abstain_queries),
        },
        "knowledge_coverage": {
            key: {"count": coverage[key], "rate": round(coverage[key] / len(queries), 6)}
            for key in ("covered", "partial", "gap")
        },
        "question_columns": {
            column: question_type for question_type, column in QUESTION_COLUMNS.items()
        },
        "structural_abstention_gate": evaluate_structural_gate(
            documents,
            normalizer,
            dimension_normalizer,
            plant_labels,
            queries,
            abstain_queries,
        ),
        "modes": modes,
        "notes": [
            "Community posts are query sources only; gold entries come from the curated knowledge base.",
            "Gap queries are excluded from Hit/Recall/MRR and retained in coverage analysis.",
            "Every retrieval metric includes its own denominator; aggregate scores exclude queries without gold.",
            "retrieval is the full N=40 view; retrieval_single_turn uniformly excludes all queries marked excluded_from_single_turn.",
            "The single-turn exclusion is defined by thread_context, not by observed model success or failure.",
            "Recall@5 and Recall@|gold| are macro averages over per-query recall.",
            "Score-threshold abstention is retained only as a weak comparison baseline.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--abstain-threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    report = build_report(args.k, args.abstain_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Baseline report: {args.output}")
    print(f"Knowledge coverage: {json.dumps(report['knowledge_coverage'], ensure_ascii=False)}")
    for mode in report["modes"]:
        aggregate = mode["retrieval"]["aggregate"]
        print(
            f"{mode['name']} (N={aggregate['eligible_queries']}): "
            f"Hit@{args.k}={aggregate[f'hit_at_{args.k}']['value']:.3f} "
            f"Recall@{args.k}={aggregate[f'recall_at_{args.k}_macro']['value']:.3f} "
            f"Recall@5={aggregate['recall_at_5_macro']['value']:.3f} "
            f"Recall@|gold|={aggregate['recall_at_gold_macro']['value']:.3f}"
        )
    gate = report["structural_abstention_gate"]
    print(
        "Structural gate: "
        f"Abstain={gate['abstain_set_accuracy']['value']:.3f} "
        f"CoveredAccept={gate['covered_query_acceptance']['value']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
