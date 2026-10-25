#!/usr/bin/env python3
"""compare processed LightKG dumps against the statistics reported in LightKG Table 5."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


EXPECTED = {
    "lastfm": {
        "users": 1873,
        "items": 3847,
        "interactions": 21173,
        "entities": 9367,
        "relations": 62,
        "triplets": 15518,
        "kappa": 0.73,
    },
    "bookcrossing": {
        "users": 11018,
        "items": 9059,
        "interactions": 24644,
        "entities": 77904,
        "relations": 27,
        "triplets": 151500,
        "kappa": 6.15,
    },
    "amazonbook": {
        "users": 20347,
        "items": 4230,
        "interactions": 234323,
        "entities": 12230,
        "relations": 21,
        "triplets": 46522,
        "kappa": 0.20,
    },
    "ml1m": {
        "users": 6039,
        "items": 3499,
        "interactions": 573637,
        "entities": 77799,
        "relations": 51,
        "triplets": 378151,
        "kappa": 0.66,
    },
}

DATASET_FILES = {
    "lastfm": "lastfm",
    "bookcrossing": "book-crossing",
    "amazonbook": "Amazon-book",
    "ml1m": "ml-1m",
}


def _read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _count_inter(prefix: Path, stem: str) -> tuple[set[str], set[str], int]:
    users: set[str] = set()
    items: set[str] = set()
    n_rows = 0
    for split in ("train", "valid", "test"):
        rows = _read_table(prefix / f"{stem}.{split}.inter")
        user_key = next(key for key in rows[0] if key.startswith("user_id"))
        item_key = next(key for key in rows[0] if key.startswith("item_id"))
        for row in rows:
            users.add(row[user_key])
            items.add(row[item_key])
            n_rows += 1
    return users, items, n_rows


def _count_kg(path: Path) -> tuple[set[str], set[str], int]:
    rows = _read_table(path)
    head_key = next(key for key in rows[0] if key.startswith("head_id"))
    rel_key = next(key for key in rows[0] if key.startswith("relation_id"))
    tail_key = next(key for key in rows[0] if key.startswith("tail_id"))
    entities: set[str] = set()
    relations: set[str] = set()
    for row in rows:
        entities.add(row[head_key])
        entities.add(row[tail_key])
        relations.add(row[rel_key])
    return entities, relations, len(rows)


def measure(processed_dir: Path, dataset: str) -> dict[str, float]:
    stem = DATASET_FILES[dataset]
    prefix = processed_dir / dataset
    users, items, n_inter = _count_inter(prefix, stem)
    entities, relations, n_triples = _count_kg(prefix / f"{stem}.kg")
    # recbole token tables reserve index 0 as [PAD]; LightKG table 5 reports those padded sizes.
    # relation count is unique relations + [PAD] + interact.
    return {
        "users_unique": len(users),
        "items_unique": len(items),
        "entities_unique": len(entities),
        "relations_unique": len(relations),
        "users": len(users) + 1,
        "items": len(items) + 1,
        "interactions": n_inter,
        "entities": len(entities) + 1,
        "relations": len(relations) + 2,
        "triplets": n_triples,
        "kappa": round(n_triples / n_inter, 2) if n_inter else 0.0,
    }


def _matches(actual: dict[str, float], expected: dict[str, float]) -> list[str]:
    mismatches = []
    for key, want in expected.items():
        got = actual[key]
        if key == "kappa":
            if abs(got - want) > 0.01:
                mismatches.append(f"{key}: got {got}, expected {want}")
        elif got != want:
            mismatches.append(f"{key}: got {got}, expected {want}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("00_data/processed"))
    args = parser.parse_args()

    failed = False
    for dataset, expected in EXPECTED.items():
        stats_path = args.processed / dataset / "stats.json"
        if not (args.processed / dataset / f"{DATASET_FILES[dataset]}.kg").exists():
            print(f"[FAIL] {dataset}: processed files missing")
            failed = True
            continue
        actual = measure(args.processed, dataset)
        stats_path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
        mismatches = _matches(actual, expected)
        if mismatches:
            failed = True
            print(f"[FAIL] {dataset}")
            for line in mismatches:
                print(f"  {line}")
        else:
            print(f"[OK] {dataset} matches Table 5")

    if failed:
        print("dataset check failed: statistics do not match, do not train")
        return 1
    print("dataset check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
