#!/usr/bin/env python3
"""read 20_results/execution_plan.yaml: which hyper-parameter searches are finished."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "20_results" / "execution_plan.yaml"

# SHUF and DOSE reuse the winning FULL configuration
REUSE_FULL = {"shuf", "dose25", "dose50", "dose75"}


def load_plan(path: Path | None = None) -> dict:
    plan_path = path or PLAN_PATH
    if not plan_path.exists():
        return {}
    return yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}


def cell_key(model: str, dataset: str, condition: str) -> str:
    return f"{model}__{dataset}__{condition}"


def hp_search_complete(plan: dict, model: str, dataset: str, condition: str) -> bool:
    hp = plan.get("hp_search") or {}
    entry = hp.get(cell_key(model, dataset, condition))
    if entry is None and condition in REUSE_FULL:
        entry = hp.get(cell_key(model, dataset, "full"))
    if not entry:
        return True
    return str(entry.get("status", "")) == "complete"
