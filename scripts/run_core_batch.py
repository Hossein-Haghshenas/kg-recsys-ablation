#!/usr/bin/env python3
"""run the next unfinished core/dose jobs, then stop for a quality gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from execution_plan import hp_search_complete, load_plan  # noqa: E402

DATASETS = ("lastfm", "bookcrossing", "ml1m")
MODELS = ("lightgcn", "kgat", "kgin", "kgcl", "lightkg", "cke")
CONDITIONS = {
    "lightgcn": ("full",),
    "kgat": ("full", "none", "shuf"),
    "kgin": ("full", "none", "shuf"),
    "kgcl": ("full", "none", "shuf"),
    "lightkg": ("full", "none", "shuf"),
    "cke": ("full", "none", "shuf"),
}
DOSE_MODELS = ("kgat", "kgin", "lightkg")
DOSE_CONDS = ("dose25", "dose50", "dose75")
SEEDS = (0, 1, 2, 3, 4)
SEEDS_AUX = (0, 1, 2)
# last.fm smoke leftovers sit around 0.01 recall and a few seconds
MIN_RECALL20 = {"lastfm": 0.05, "bookcrossing": 0.01, "amazonbook": 0.01, "ml1m": 0.05}
MIN_TRAIN_S = 30.0


def _core_jobs(dataset: str | None = None) -> list[tuple[str, str, str, int]]:
    datasets = DATASETS if dataset is None else (dataset,)
    jobs = []
    for name in datasets:
        for seed in SEEDS:
            for model in MODELS:
                for condition in CONDITIONS[model]:
                    jobs.append((model, name, condition, seed))
    return jobs


def _dose_jobs(dataset: str | None = None) -> list[tuple[str, str, str, int]]:
    datasets = DATASETS if dataset is None else (dataset,)
    jobs = []
    for name in datasets:
        for seed in SEEDS_AUX:
            for model in DOSE_MODELS:
                for condition in DOSE_CONDS:
                    jobs.append((model, name, condition, seed))
    return jobs


def _jobs_for_queue(queue: str, dataset: str | None) -> list[tuple[str, str, str, int]]:
    if queue == "core":
        return _core_jobs(dataset)
    if queue == "dose":
        return _dose_jobs(dataset)
    return _core_jobs(dataset) + _dose_jobs(dataset)


def _run_id(model: str, dataset: str, condition: str, seed: int) -> str:
    return f"{model}__{dataset}__{condition}__s{seed}"


def _metrics_path(model: str, run_id: str) -> Path:
    return ROOT / "10_models" / model / "runs" / run_id / "metrics.json"


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: object, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quality(payload: dict, dataset: str) -> tuple[bool, str]:
    if payload.get("status") != "ok":
        return False, f"status={payload.get('status')}"
    recall20 = _as_float(payload.get("recall@20"))
    train_s = _as_float(payload.get("train_time_s"))
    floor = MIN_RECALL20.get(dataset, 0.0)
    # short + tiny recall is a leftover smoke; a full train can legitimately score below the floor
    if train_s < MIN_TRAIN_S:
        if recall20 < floor:
            return False, f"recall@20={recall20} < {floor} on a short run"
        return False, f"train_time_s={train_s} looks like a smoke leftover"
    return True, "ok"


def _config_path(model: str, dataset: str, condition: str) -> Path:
    return ROOT / "10_models" / model / "configs" / f"{dataset}__{condition}.yaml"


def _pending(
    queue: str,
    dataset: str | None,
    plan: dict,
) -> list[tuple[str, str, str, int]]:
    pending = []
    for job in _jobs_for_queue(queue, dataset):
        model, name, condition, _seed = job
        if not hp_search_complete(plan, model, name, condition):
            continue
        if not _config_path(model, name, condition).exists():
            continue
        run_id = _run_id(*job)
        payload = _load_metrics(_metrics_path(model, run_id))
        if payload is None:
            pending.append(job)
            continue
        ok, _reason = _quality(payload, name)
        if not ok:
            pending.append(job)
    return pending


def _launch(model: str, dataset: str, condition: str, seed: int) -> int:
    if model == "kgcl":
        script = ROOT / "10_models" / "kgcl" / "run_kgcl.py"
        cmd = [
            sys.executable,
            str(script),
            "--dataset",
            dataset,
            "--condition",
            condition,
            "--seed",
            str(seed),
        ]
    else:
        script = ROOT / "10_models" / "_shared" / "run_recbole.py"
        cmd = [
            sys.executable,
            str(script),
            "--model",
            model,
            "--dataset",
            dataset,
            "--condition",
            condition,
            "--seed",
            str(seed),
        ]
    print(f"\n=== start {model} {dataset} {condition} s{seed} ===", flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    return completed.returncode


def _append_gate(rows: list[dict]) -> None:
    dest = ROOT / "log" / "batch_gate.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with dest.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"ts": stamp, **row}, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=("all",) + DATASETS)
    parser.add_argument("--queue", default="remaining", choices=("core", "dose", "remaining"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--until-empty",
        action="store_true",
        help="keep taking --limit batches until the ready queue is empty",
    )
    args = parser.parse_args()
    dataset = None if args.dataset == "all" else args.dataset

    while True:
        plan = load_plan()
        pending = _pending(args.queue, dataset, plan)
        total = len(_jobs_for_queue(args.queue, dataset))
        print(
            f"pending {len(pending)} / {total} queue={args.queue} dataset={args.dataset}",
            flush=True,
        )
        if not pending:
            print("queue empty")
            return 0

        batch = pending[: args.limit]
        gate_rows: list[dict] = []
        failed = 0
        for model, name, condition, seed in batch:
            run_id = _run_id(model, name, condition, seed)
            started = time.time()
            code = _launch(model, name, condition, seed)
            payload = _load_metrics(_metrics_path(model, run_id)) or {}
            ok, reason = _quality(payload, name) if payload else (False, "missing metrics.json")
            if code != 0:
                ok = False
                reason = f"exit={code}; {reason}"
            row = {
                "run_id": run_id,
                "ok": ok,
                "reason": reason,
                "recall@20": payload.get("recall@20", ""),
                "total_epochs": payload.get("total_epochs", ""),
                "train_time_s": payload.get("train_time_s", ""),
                "elapsed_s": round(time.time() - started, 2),
            }
            gate_rows.append(row)
            mark = "PASS" if ok else "FAIL"
            print(
                f"=== {mark} {run_id} recall@20={row['recall@20']} "
                f"epochs={row['total_epochs']} time={row['train_time_s']} ({reason}) ===",
                flush=True,
            )
            if not ok:
                failed += 1
                break

        collect = subprocess.run([sys.executable, str(ROOT / "20_results" / "collect.py")], cwd=ROOT)
        if collect.returncode != 0:
            print("collect.py failed", file=sys.stderr)
            failed += 1
        _append_gate(gate_rows)

        print("\n--- quality gate ---", flush=True)
        for row in gate_rows:
            print(json.dumps(row, ensure_ascii=False), flush=True)
        leftover = len(_pending(args.queue, dataset, plan))
        print(f"still pending: {leftover}", flush=True)
        if failed:
            return 1
        if not args.until_empty:
            return 0


if __name__ == "__main__":
    sys.exit(main())
