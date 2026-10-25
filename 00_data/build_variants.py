#!/usr/bin/env python3
"""build locked KG variants: none, shuf, dose."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from verify_stats import DATASET_FILES


KG_HEADER = ["head_id:token", "relation_id:token", "tail_id:token"]
SHUF_SEEDS = (0, 1, 2)
DOSE_RATIOS = (25, 50, 75)


def _read_triples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _dose_covers_links(rows: list[dict[str, str]], linked: set[str]) -> set[str]:
    present = {row["head_id:token"] for row in rows} | {row["tail_id:token"] for row in rows}
    return linked - present


def _write_triples(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=KG_HEADER, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in KG_HEADER})


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    mapping = {}
    for key, value in row.items():
        if key.startswith("head_id"):
            mapping["head_id:token"] = value
        elif key.startswith("relation_id"):
            mapping["relation_id:token"] = value
        elif key.startswith("tail_id"):
            mapping["tail_id:token"] = value
    return mapping


def _shuffle(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    heads = [row["head_id:token"] for row in rows]
    tails = [row["tail_id:token"] for row in rows]
    rng.shuffle(heads)
    rng.shuffle(tails)
    out = []
    for rel_row, head, tail in zip(rows, heads, tails):
        out.append(
            {
                "head_id:token": head,
                "relation_id:token": rel_row["relation_id:token"],
                "tail_id:token": tail,
            }
        )
    return out


def _linked_entities(link_path: Path) -> set[str]:
    if not link_path.exists():
        return set()
    rows = _read_triples(link_path)
    if not rows:
        return set()
    key = next(col for col in rows[0] if col.startswith("entity_id"))
    return {row[key] for row in rows if row.get(key)}


def _dose(
    rows: list[dict[str, str]],
    ratio: int,
    seed: int,
    linked: set[str],
) -> list[dict[str, str]]:
    # recbole 1.1.1 _merge_item_and_entity assumes every .link entity still
    # appears in the kg. a raw sample drops those and then either crashes or
    # silently deletes the item's interactions — both invalidate a dose curve.
    rng = random.Random(seed)
    if not rows:
        return []
    keep = max(1, int(len(rows) * ratio / 100))
    by_ent: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_ent.setdefault(row["head_id:token"], []).append(idx)
        by_ent.setdefault(row["tail_id:token"], []).append(idx)
    must: set[int] = set()
    extras: list[dict[str, str]] = []
    fallback_rel = rows[0]["relation_id:token"]
    for ent in sorted(linked):
        idxs = by_ent.get(ent)
        if idxs:
            must.add(rng.choice(idxs))
            continue
        extras.append(
            {
                "head_id:token": ent,
                "relation_id:token": fallback_rel,
                "tail_id:token": ent,
            }
        )
    leftover = [idx for idx in range(len(rows)) if idx not in must]
    rng.shuffle(leftover)
    need = max(0, keep - len(must))
    chosen = [rows[idx] for idx in list(must) + leftover[:need]]
    return chosen + extras


def build_one(processed: Path, variants: Path, dataset: str) -> None:
    stem = DATASET_FILES[dataset]
    src = processed / dataset / f"{stem}.kg"
    rows = [_normalize_row(row) for row in _read_triples(src)]
    linked = _linked_entities(processed / dataset / f"{stem}.link")
    dest = variants / dataset
    _write_triples(dest / "none.txt", [])
    _write_triples(dest / "full.txt", rows)
    for seed in SHUF_SEEDS:
        _write_triples(dest / f"shuf_s{seed}.txt", _shuffle(rows, seed))
    for ratio in DOSE_RATIOS:
        for seed in SHUF_SEEDS:
            sampled = _dose(rows, ratio, seed * 100 + ratio, linked)
            missing = _dose_covers_links(sampled, linked)
            if missing:
                raise RuntimeError(
                    f"{dataset} dose{ratio}_s{seed} dropped {len(missing)} linked entities"
                )
            _write_triples(dest / f"dose{ratio}_s{seed}.txt", sampled)
            print(
                f"  {dataset} dose{ratio}_s{seed}: {len(sampled)} triples "
                f"(target {int(len(rows) * ratio / 100)}, linked {len(linked)})"
            )
    print(f"wrote variants for {dataset}: {len(rows)} triples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("00_data/processed"))
    parser.add_argument("--variants", type=Path, default=Path("00_data/kg_variants"))
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="subset of lastfm/bookcrossing/amazonbook/ml1m; default is every processed dump that exists",
    )
    args = parser.parse_args()
    wanted = args.datasets or list(DATASET_FILES)
    for dataset in wanted:
        src = args.processed / dataset / f"{DATASET_FILES[dataset]}.kg"
        if not src.exists():
            print(f"skip {dataset}: missing {src}")
            continue
        build_one(args.processed, args.variants, dataset)


if __name__ == "__main__":
    main()
