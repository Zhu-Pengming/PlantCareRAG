#!/usr/bin/env python3
"""Build observed, unlabeled query candidates from the frozen Stack Exchange dump."""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
from html.parser import HTMLParser
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "query_sources"
SOURCE_ID = "stackexchange:gardening:2024-03"
SOURCE_DIR = RAW_ROOT / SOURCE_ID.replace(":", "__")
ALIASES = ROOT / "config" / "query_entity_aliases.json"
OUTPUT = ROOT / "data" / "query_candidates.json"
REPORT = ROOT / "reports" / "query_candidate_stats.json"


class PlainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: str) -> str:
    parser = PlainText()
    parser.feed(html.unescape(value or ""))
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def load_users(path: Path) -> dict[str, str]:
    users: dict[str, str] = {}
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag == "row" and element.attrib.get("Id"):
            users[element.attrib["Id"]] = element.attrib.get("DisplayName", "")
        element.clear()
    return users


def alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(
        r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])",
        flags=re.IGNORECASE,
    )


def parse_tags(value: str) -> list[str]:
    return re.findall(r"<([^>]+)>", value or "")


def build(source_dir: Path) -> list[dict]:
    aliases = json.loads(ALIASES.read_text(encoding="utf-8"))
    patterns = {
        plant_id: [(alias, alias_pattern(alias)) for alias in plant_aliases]
        for plant_id, plant_aliases in aliases.items()
    }
    users = load_users(source_dir / "Users.xml")
    candidates: list[dict] = []
    for _, element in ET.iterparse(source_dir / "Posts.xml", events=("end",)):
        if element.tag != "row" or element.attrib.get("PostTypeId") != "1":
            element.clear()
            continue
        post_id = element.attrib["Id"]
        title = plain_text(element.attrib.get("Title", ""))
        body_html = element.attrib.get("Body", "")
        body = plain_text(body_html)
        lowered = {"title": title.lower(), "body": body.lower()}
        matches = []
        for plant_id, plant_patterns in patterns.items():
            for alias, pattern in plant_patterns:
                locations = [name for name, text in lowered.items() if pattern.search(text)]
                if locations:
                    matches.append({"plant_id": plant_id, "alias": alias, "locations": locations})
        if not matches:
            element.clear()
            continue
        candidate_id = f"qv12:stackexchange_gardening:{post_id}"
        owner = element.attrib.get("OwnerUserId")
        candidates.append(
            {
                "schema_version": "1.2.0-candidate",
                "candidate_id": candidate_id,
                "source_dataset_id": SOURCE_ID,
                "source_record_id": post_id,
                "raw_title": title,
                "raw_body": body,
                "source_body_html": body_html,
                "raw_text": f"{title}\n\n{body}".strip(),
                "source_url": f"https://gardening.stackexchange.com/questions/{post_id}",
                "author": {
                    "owner_user_id": owner,
                    "display_name": users.get(owner) if owner else None,
                },
                "created_at": element.attrib.get("CreationDate", ""),
                "content_license": element.attrib.get("ContentLicense", ""),
                "source_metrics": {
                    "score": int(element.attrib.get("Score", "0")),
                    "view_count": int(element.attrib.get("ViewCount", "0")),
                    "answer_count": int(element.attrib.get("AnswerCount", "0")),
                    "accepted_answer_id": element.attrib.get("AcceptedAnswerId"),
                },
                "tags": parse_tags(element.attrib.get("Tags", "")),
                "alias_matches": matches,
                "annotation_status": "pending_human_annotation",
                "benchmark_eligible": False,
                "split_hash": hashlib.sha256(candidate_id.encode("utf-8")).hexdigest(),
            }
        )
        element.clear()
    return sorted(candidates, key=lambda item: int(item["source_record_id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    try:
        candidates = build(args.source_dir)
    except (OSError, ET.ParseError) as exc:
        print(f"candidate build failed: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    plant_counts: collections.Counter[str] = collections.Counter()
    license_counts: collections.Counter[str] = collections.Counter()
    for candidate in candidates:
        plant_counts.update({match["plant_id"] for match in candidate["alias_matches"]})
        license_counts[candidate["content_license"]] += 1
    report = {
        "schema_version": "1.0.0",
        "source_dataset_id": SOURCE_ID,
        "candidate_count": len(candidates),
        "benchmark_eligible_count": 0,
        "missing_author_count": sum(
            candidate["author"]["display_name"] is None for candidate in candidates
        ),
        "multi_entity_candidate_count": sum(
            len({match["plant_id"] for match in candidate["alias_matches"]}) > 1
            for candidate in candidates
        ),
        "per_plant_alias_match_count": dict(sorted(plant_counts.items())),
        "per_license_count": dict(sorted(license_counts.items())),
        "candidate_file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(candidates)} pending candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
