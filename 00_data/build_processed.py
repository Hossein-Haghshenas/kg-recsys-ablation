#!/usr/bin/env python3
"""copy LightKG dumps into processed/ without rewriting splits."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from verify_stats import DATASET_FILES, measure


RAW_ALIASES = {
    "lastfm": ("lastfm", "lastfm"),
    "bookcrossing": ("bookcrossing", "book-crossing"),
    "amazonbook": ("amazonbook", "Amazon-book"),
    "ml1m": ("ml1m", "ml-1m"),
}


def _copy_dataset(raw_root: Path, processed_root: Path, dataset: str) -> None:
    dest_name, src_name = RAW_ALIASES[dataset]
    src = raw_root / dest_name
    if not src.exists():
        src = raw_root / src_name
    dest = processed_root / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file():
            shutil.copy2(path, dest / path.name)
    stats = measure(processed_root, dest_name)
    (dest / "stats.json").write_text(
        __import__("json").dumps(stats, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"copied {src} -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("00_data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("00_data/processed"))
    args = parser.parse_args()
    args.processed.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_FILES:
        _copy_dataset(args.raw, args.processed, dataset)


if __name__ == "__main__":
    main()
