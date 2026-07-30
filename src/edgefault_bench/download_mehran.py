"""Checksum-enforced download command for the proposed Mehran v2 selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from edgefault_bench.datasets.mehran import load_mehran_manifest, verify_mehran_file
from edgefault_bench.download import download_file, select_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected Mehran v2 files with mandatory checksum validation."
    )
    parser.add_argument("--manifest", default="registry/mehran_v2.json")
    parser.add_argument("--raw-dir", default="data/raw/mehran_v2")
    parser.add_argument(
        "--files", nargs="*", help="Optional exact CSV filenames; default: all selected 36"
    )
    parser.add_argument(
        "--repair", action="store_true", help="Quarantine and replace invalid files"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest, selected = load_mehran_manifest(args.manifest)
    files = select_files(selected, args.files)
    print(
        f"Mehran triaxial bearing v{manifest['version']} | {manifest['doi']} | "
        f"{manifest['license']['spdx']} | selected_files={len(files)} | "
        f"protocol={manifest['protocol_status']}"
    )
    raw_dir = Path(args.raw_dir)
    if args.verify_only:
        for specification in files:
            verify_mehran_file(raw_dir / specification.filename, specification)
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
