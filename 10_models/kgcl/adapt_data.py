#!/usr/bin/env python3
"""convert LightKG RecBole files into official KGCL train/test/kg format."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_FILES = {
    "lastfm": "lastfm",
    "bookcrossing": "book-crossing",
    "amazonbook": "Amazon-book",
    "ml1m": "ml-1m",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _key(row: dict[str, str], prefix: str) -> str:
    return next(key for key in row if key.startswith(prefix))


def _write_user_items(path: Path, grouped: dict[int, list[int]]) -> None:
    lines = []
    for user in sorted(grouped):
        items = " ".join(str(item) for item in grouped[user])
        lines.append(f"{user} {items}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _entity_to_item(path: Path) -> dict[str, str]:
    rows = _rows(path)
    item_key = _key(rows[0], "item_id")
    entity_key = _key(rows[0], "entity_id")
    return {row[entity_key]: row[item_key] for row in rows}


def convert(dataset: str, condition: str, seed: int) -> Path:
    stem = DATASET_FILES[dataset]
    processed = ROOT / "00_data" / "processed" / dataset
    dest = ROOT / "10_models" / "kgcl" / "code" / "official" / "data" / dataset
    dest.mkdir(parents=True, exist_ok=True)

    splits = {}
    users: set[str] = set()
    items: set[str] = set()
    for split in ("train", "valid", "test"):
        rows = _rows(processed / f"{stem}.{split}.inter")
        user_key = _key(rows[0], "user_id")
        item_key = _key(rows[0], "item_id")
        splits[split] = [(row[user_key], row[item_key]) for row in rows]
        users.update(row[user_key] for row in rows)
        items.update(row[item_key] for row in rows)

    user_map = {token: idx for idx, token in enumerate(sorted(users))}
    item_map = {token: idx for idx, token in enumerate(sorted(items))}

    for split in ("train", "valid", "test"):
        grouped: dict[int, list[int]] = defaultdict(list)
        for user, item in splits[split]:
            grouped[user_map[user]].append(item_map[item])
        _write_user_items(dest / f"{split}.txt", grouped)

    if condition == "full":
        variant_name = "full.txt"
    elif condition == "none":
        variant_name = "none.txt"
    elif condition == "shuf":
        target = ROOT / "00_data" / "kg_variants" / dataset / f"shuf_s{seed}.txt"
        variant_name = f"shuf_s{seed}.txt" if target.exists() else f"shuf_s{seed % 3}.txt"
    elif condition == "relshuf":
        target = ROOT / "00_data" / "kg_variants" / dataset / f"relshuf_s{seed}.txt"
        variant_name = f"relshuf_s{seed}.txt" if target.exists() else f"relshuf_s{seed % 5}.txt"
    else:
        variant_name = f"{condition}_s{seed % 3}.txt" 
    kg_rows = _rows(ROOT / "00_data" / "kg_variants" / dataset / variant_name)
    # kg files use RecBole entity ids; official KGCL TransE looks heads up as items
    link_path = processed / f"{stem}.link"
    entity_map: dict[str, int] = {}
    if link_path.exists():
        for entity_id, item_token in _entity_to_item(link_path).items():
            if item_token in item_map:
                entity_map[entity_id] = item_map[item_token]
    else:
        entity_map = dict(item_map)
    next_id = len(item_map)
    rel_map: dict[str, int] = {}
    triples: list[tuple[int, int, int]] = []
    for row in kg_rows:
        head = row.get("head_id:token") or next(iter(row.values()))
        rel = row.get("relation_id:token") or list(row.values())[1]
        tail = row.get("tail_id:token") or list(row.values())[2]
        for node in (head, tail):
            if node not in entity_map:
                entity_map[node] = next_id
                next_id += 1
        if rel not in rel_map:
            rel_map[rel] = len(rel_map)
        triples.append((entity_map[head], rel_map[rel], entity_map[tail]))
    # official TransE indexes heads with embedding_item; keep or flip so head is an item
    n_items = len(item_map)
    n_rels = max(len(rel_map), 1)
    item_headed: list[tuple[int, int, int]] = []
    for head_id, rel_id, tail_id in triples:
        if head_id < n_items:
            item_headed.append((head_id, rel_id, tail_id))
        elif tail_id < n_items:
            item_headed.append((tail_id, rel_id + n_rels, head_id))
    triples = item_headed
    if not triples:
        triples = [(0, 0, max(n_items, 1))]
    (dest / "kg.txt").write_text(
        "\n".join(f"{h} {r} {t}" for h, r, t in triples) + "\n",
        encoding="utf-8",
    )
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES))
    parser.add_argument("--condition", default="full")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dest = convert(args.dataset, args.condition, args.seed)
    print(f"wrote kgcl files to {dest}")


if __name__ == "__main__":
    main()
