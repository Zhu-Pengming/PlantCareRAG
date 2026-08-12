#!/usr/bin/env python3
"""Download frozen public Kaggle archives and verify their checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "datasets.json"
RAW_ROOT = ROOT / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "plant-care-rag-dataset-v1/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def fetch_dataset(dataset: dict, raw_root: Path, force: bool = False) -> Path:
    dataset_dir = raw_root / dataset["id"].split(":", 1)[1].replace("/", "__")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive = dataset_dir / dataset["archive_name"]
    member = dataset_dir / dataset["member_name"]

    if force or not archive.exists():
        temporary = archive.with_suffix(archive.suffix + ".part")
        try:
            download(dataset["download_url"], temporary)
            temporary_hash = sha256(temporary)
            if temporary_hash != dataset["archive_sha256"]:
                raise ValueError(
                    f"{dataset['id']}: downloaded archive SHA-256 mismatch: "
                    f"{temporary_hash}"
                )
            temporary.replace(archive)
        finally:
            if temporary.exists():
                temporary.unlink()
    actual_archive_hash = sha256(archive)
    if actual_archive_hash != dataset["archive_sha256"]:
        raise ValueError(
            f"{dataset['id']}: archive SHA-256 mismatch: {actual_archive_hash}"
        )

    with zipfile.ZipFile(archive) as bundle:
        if dataset["member_name"] not in bundle.namelist():
            raise ValueError(
                f"{dataset['id']}: missing archive member {dataset['member_name']}"
            )
        member.write_bytes(bundle.read(dataset["member_name"]))
    actual_member_hash = sha256(member)
    if actual_member_hash != dataset["member_sha256"]:
        raise ValueError(
            f"{dataset['id']}: member SHA-256 mismatch: {actual_member_hash}"
        )
    return member


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    selected = [
        dataset
        for dataset in manifest["datasets"]
        if dataset["decision"] != "rejected" or args.include_rejected
    ]
    try:
        for dataset in selected:
            member = fetch_dataset(dataset, args.raw_root, args.force)
            print(f"verified {dataset['id']} -> {member}")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
