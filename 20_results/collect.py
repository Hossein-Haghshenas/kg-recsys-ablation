#!/usr/bin/env python3
"""fold 10_models/*/runs/*/metrics.json into runs.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


COLUMNS = [
    "run_id",
    "model",
    "dataset",
    "condition",
    "seed",
    "code_source",
    "commit",
    "config_hash",
    "recall@10",
    "recall@20",
    "ndcg@10",
    "ndcg@20",
    "mrr@10",
    "best_epoch",
    "total_epochs",
    "train_time_s",
    "time_per_epoch_s",
    "gpu_mem_mb",
    "n_params",
    "best_valid_score",
    "status",
    "notes",
]


def _load_runs(models_root: Path) -> list[dict[str, object]]:
    rows = []
    for metrics in models_root.glob("*/runs/*/metrics.json"):
        if "runs_protocol_v1" in metrics.parts:
            continue
        if "__hp" in metrics.parent.name:
            continue
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        rows.append({column: payload.get(column, "") for column in COLUMNS})
    rows.sort(key=lambda row: str(row["run_id"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, default=Path("10_models"))
    parser.add_argument("--out", type=Path, default=Path("20_results/runs.csv"))
    args = parser.parse_args()
    rows = _load_runs(args.models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} runs to {args.out}")


if __name__ == "__main__":
    main()
