#!/usr/bin/env python3
"""Snapshot the official Mendeley metadata for the proposed Mehran v2 adapter."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://data.mendeley.com/public-api/datasets/fm6xzxnf36"
LANDING_PAGE = "https://data.mendeley.com/datasets/fm6xzxnf36/2"
SELECTED_PATTERN = re.compile(
    r"^(?:0\.7|0\.9|1\.1|1\.3|1\.5|1\.7)(?:inner|outer)-"
    r"(?:100|200|300)watt(?:-[A-Za-z0-9]+)?\.csv$"
)


def _load_api_payload(source: Path | None) -> dict[str, Any]:
    if source is not None:
        return json.loads(source.read_text(encoding="utf-8"))
    request = urllib.request.Request(API_URL, headers={"User-Agent": "EdgeFault-Bench/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def build_registry(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize volatile API metadata into the immutable registry schema."""

    if payload.get("id") != "fm6xzxnf36" or payload.get("version") != 2:
        raise ValueError("expected Mendeley dataset fm6xzxnf36 version 2")
    if payload.get("doi", {}).get("id") != "10.17632/fm6xzxnf36.2":
        raise ValueError("unexpected Mehran v2 DOI")
    licence = payload.get("data_licence", {})
    if licence.get("short_name") != "CC BY 4.0":
        raise ValueError("Mehran v2 must retain its official CC BY 4.0 license")

    files: list[dict[str, object]] = []
    for item in payload.get("files", []):
        content = item["content_details"]
        selected = SELECTED_PATTERN.fullmatch(item["filename"]) is not None
        files.append(
            {
                "filename": item["filename"],
                "id": item["id"],
                "size_bytes": int(content["size"]),
                "sha256": content["sha256_hash"].lower(),
                "download_url": (
                    "https://data.mendeley.com/public-files/datasets/fm6xzxnf36/files/"
                    f"{item['id']}/file_downloaded"
                ),
                "selected": selected,
                "exclusion_reason": None
                if selected
                else "healthy recording has no stated 100/200/300 W load assignment",
            }
        )
    files.sort(key=lambda item: str(item["filename"]).lower())
    selected_count = sum(bool(item["selected"]) for item in files)
    if len(files) != 38 or selected_count != 36:
        raise ValueError(
            f"expected 38 total and 36 selected files, found {len(files)} and {selected_count}"
        )
    if sum(int(item["size_bytes"]) for item in files) != int(payload["size"]):
        raise ValueError("API file sizes do not sum to the reported dataset size")

    return {
        "schema_version": 1,
        "dataset_id": "mehran-triaxial-bearing-v2",
        "title": payload["name"],
        "repository": "Mendeley Data",
        "landing_page": LANDING_PAGE,
        "api_url": API_URL,
        "doi": payload["doi"]["id"],
        "version": 2,
        "protocol_status": "accepted",
        "published": payload["publish_date"],
        "license": {
            "spdx": "CC-BY-4.0",
            "name": licence["short_name"],
            "url": licence["url"],
        },
        "sampling_rate_hz": 10_000,
        "channels": ["x", "y", "z"],
        "selection": {
            "selected_count": selected_count,
            "excluded_count": len(files) - selected_count,
            "reason": (
                "Select the complete inner/outer, six-defect-size, three-load grid; exclude "
                "healthy files whose source metadata do not assign a load."
            ),
        },
        "files": files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional saved API JSON for offline use")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    registry = build_registry(_load_api_payload(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} | files={len(registry['files'])} | "
        f"selected={registry['selection']['selected_count']}"
    )


if __name__ == "__main__":
    main()
