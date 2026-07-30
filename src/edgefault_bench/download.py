"""Checksum-enforced download command for pinned public dataset files."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypeVar

from edgefault_bench.datasets.checksums import RegisteredFile, verify_registered_file
from edgefault_bench.datasets.hust import load_hust_manifest, verify_hust_file

FileSpecification = TypeVar("FileSpecification", bound=RegisteredFile)


def select_files(
    files: tuple[FileSpecification, ...], names: Iterable[str] | None
) -> tuple[FileSpecification, ...]:
    if names is None:
        return files
    requested = set(names)
    available = {item.filename for item in files}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f"requested files are not in the pinned registry: {unknown}")
    return tuple(item for item in files if item.filename in requested)


def _download_once(
    specification: RegisteredFile, destination: Path, *, timeout: float
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "EdgeFault-Bench/0.1 (+https://github.com/ziyicheng427/edgefault-bench)"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(specification.download_url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", response.getcode())
        append = existing > 0 and status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    verify_registered_file(partial, specification)
    partial.replace(destination)


def download_file(
    specification: RegisteredFile,
    raw_dir: str | Path,
    *,
    repair: bool = False,
    retries: int = 3,
    timeout: float = 120.0,
) -> str:
    """Download one file atomically and require the registry size and SHA-256."""

    directory = Path(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / specification.filename
    if destination.exists():
        try:
            verify_registered_file(destination, specification)
            return "verified"
        except ValueError:
            if not repair:
                raise ValueError(
                    f"existing file failed verification: {destination}; rerun with --repair"
                ) from None
            destination.replace(destination.with_suffix(destination.suffix + ".invalid"))

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_once(specification, destination, timeout=timeout)
            return "downloaded"
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"failed to download {specification.filename} after {retries} attempts: {last_error}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the pinned HUST v3 subset with mandatory checksum validation."
    )
    parser.add_argument("--manifest", default="registry/hust_v3.json")
    parser.add_argument("--raw-dir", default="data/raw/hust_v3")
    parser.add_argument("--files", nargs="*", help="Optional exact MAT filenames; default: all 60")
    parser.add_argument(
        "--repair", action="store_true", help="Quarantine and replace invalid files"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest, all_files = load_hust_manifest(args.manifest)
    files = select_files(all_files, args.files)
    print(
        f"HUST bearing v{manifest['version']} | {manifest['doi']} | "
        f"{manifest['license']['spdx']} | files={len(files)}"
    )
    raw_dir = Path(args.raw_dir)
    if args.verify_only:
        for specification in files:
            verify_hust_file(raw_dir / specification.filename, specification)
            print(f"verified  {specification.filename}")
        return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_file, item, raw_dir, repair=args.repair): item
            for item in files
        }
        completed = 0
        for future in as_completed(futures):
            specification = futures[future]
            status = future.result()
            completed += 1
            print(f"[{completed:02d}/{len(files):02d}] {status:10s} {specification.filename}")


if __name__ == "__main__":
    main()
