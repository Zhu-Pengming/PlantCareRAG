#!/usr/bin/env python3
"""Download and verify frozen real-query source archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "query_sources.json"
RAW_ROOT = ROOT / "data" / "raw" / "query_sources"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "PlantCareRAG-v1.2/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def fetch(source: dict, raw_root: Path, force: bool) -> Path:
    source_dir = raw_root / source["id"].replace(":", "__")
    source_dir.mkdir(parents=True, exist_ok=True)
    archive = source_dir / source["archive_name"]
    if force or not archive.exists():
        temporary = archive.with_suffix(archive.suffix + ".part")
        try:
            download(source["download_url"], temporary)
            if digest(temporary, "sha256") != source["archive_sha256"]:
                raise ValueError(f"{source['id']}: downloaded archive SHA-256 mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    for algorithm in ("sha1", "sha256"):
        actual = digest(archive, algorithm)
        if actual != source[f"archive_{algorithm}"]:
            raise ValueError(f"{source['id']}: archive {algorithm} mismatch: {actual}")
    subprocess.run(
        ["bsdtar", "-xf", str(archive), "-C", str(source_dir), *source["members"]],
        check=True,
    )
    for member in source["members"]:
        if not (source_dir / member).is_file():
            raise ValueError(f"{source['id']}: missing extracted member {member}")
    return source_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    try:
        for source in manifest["sources"]:
            if source["decision"].startswith("accepted"):
                print(f"verified {source['id']} -> {fetch(source, args.raw_root, args.force)}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"query-source download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
