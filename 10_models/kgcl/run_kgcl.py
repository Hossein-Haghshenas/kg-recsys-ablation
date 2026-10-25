#!/usr/bin/env python3
"""run official KGCL and write the shared metrics.json schema."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from adapt_data import convert


ROOT = Path(__file__).resolve().parents[2]
KGCL_DIR = Path(__file__).resolve().parent
KGCL_CODE = KGCL_DIR / "code" / "official" / "code"
EPOCH_RE = re.compile(r"EPOCH\[(\d+)/(\d+)\]")
EARLY_RE = re.compile(r"early stop triggerd at epoch (\d+)")
ARRAY_RE = re.compile(r"'(\w+)': array\(\[([^\]]+)\]\)")
BEST_VALID_RE = re.compile(r"BEST_VALID\s+([0-9.]+)")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT / "10_models" / "kgcl" / "code" / "official",
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _floats(blob: str) -> list[float]:
    return [float(part) for part in blob.split(",") if part.strip()]


def _metrics_from_result_line(line: str, ks: list[int] | None = None) -> dict[str, float]:
    arrays = {name: _floats(values) for name, values in ARRAY_RE.findall(line)}
    if "recall" not in arrays or "ndcg" not in arrays:
        return {}
    ks = ks or [10, 20]
    out: dict[str, float] = {}
    for metric in ("recall", "ndcg", "mrr"):
        values = arrays.get(metric)
        if not values:
            continue
        for index, k in enumerate(ks):
            if index < len(values):
                out[f"{metric}@{k}"] = values[index]
    return out


def parse_kgcl_stdout(text: str, ks: list[int] | None = None) -> dict[str, object]:
    last_epoch = ""
    best_epoch = ""
    last_metrics: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    for line in text.splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            last_epoch = int(epoch_match.group(1))
        parsed = _metrics_from_result_line(line, ks)
        if parsed:
            last_metrics = parsed
        if "find a better model" in line and last_metrics:
            best_metrics = dict(last_metrics)
            best_epoch = last_epoch
        early = EARLY_RE.search(line)
        if early and not last_epoch:
            last_epoch = int(early.group(1)) + 1
    chosen = best_metrics or last_metrics
    best_valid = ""
    match = BEST_VALID_RE.search(text)
    if match:
        best_valid = float(match.group(1))
    return {
        "recall@10": chosen.get("recall@10", ""),
        "recall@20": chosen.get("recall@20", ""),
        "ndcg@10": chosen.get("ndcg@10", ""),
        "ndcg@20": chosen.get("ndcg@20", ""),
        "mrr@10": chosen.get("mrr@10", ""),
        "best_epoch": best_epoch,
        "total_epochs": last_epoch,
        "best_valid_score": best_valid,
    }


def _round_metric(value: object) -> object:
    if value == "" or value is None:
        return ""
    return round(float(value), 4)


def _run_logged(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            chunks.append(line)
    code = proc.wait()
    return code, "".join(chunks)


def _eval_checkpoint(dataset: str, seed: int, model_name: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(KGCL_DIR / "eval_ckpt.py"),
            "--dataset",
            dataset,
            "--seed",
            str(seed),
            "--model",
            model_name,
        ],
        cwd=KGCL_CODE,
        check=False,
        text=True,
        capture_output=True,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    for line in text.splitlines():
        if line.startswith("KGCL_METRICS "):
            return json.loads(line[len("KGCL_METRICS ") :])
    raise RuntimeError(text[-4000:])


def _payload(
    run_id: str,
    args: argparse.Namespace,
    status: str,
    train_time: float,
    parsed: dict[str, object],
    extra: dict[str, object] | None = None,
    notes: str = "",
) -> dict[str, object]:
    extra = extra or {}
    total_epochs = parsed.get("total_epochs") or args.max_epochs
    try:
        epoch_count = max(int(total_epochs), 1)
    except (TypeError, ValueError):
        epoch_count = max(int(args.max_epochs), 1)
    body = {
        "run_id": run_id,
        "model": "kgcl",
        "dataset": args.dataset,
        "condition": args.condition,
        "seed": args.seed,
        "code_source": "official",
        "commit": _git_commit(),
        "config_hash": "",
        "recall@10": _round_metric(extra.get("recall@10", parsed.get("recall@10", ""))),
        "recall@20": _round_metric(extra.get("recall@20", parsed.get("recall@20", ""))),
        "ndcg@10": _round_metric(extra.get("ndcg@10", parsed.get("ndcg@10", ""))),
        "ndcg@20": _round_metric(extra.get("ndcg@20", parsed.get("ndcg@20", ""))),
        "mrr@10": _round_metric(extra.get("mrr@10", parsed.get("mrr@10", ""))),
        "best_epoch": parsed.get("best_epoch", ""),
        "total_epochs": total_epochs,
        "train_time_s": round(train_time, 2),
        "time_per_epoch_s": round(train_time / epoch_count, 2),
        "gpu_mem_mb": extra.get("gpu_mem_mb", ""),
        "n_params": extra.get("n_params", ""),
        "best_valid_score": _round_metric(parsed.get("best_valid_score", extra.get("best_valid_score", ""))),
        "status": status,
        "notes": notes,
    }
    return body


def _write_metrics(dest: Path, payload: dict[str, object]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--neg-k", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    try:
        return _run(args)
    except FileNotFoundError as exc:
        print(f"fatal: {exc}", flush=True)
        return 2


def _run(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"kgcl__{args.dataset}__{args.condition}__s{args.seed}"
    dest = ROOT / "10_models" / "kgcl" / "runs" / run_id / "metrics.json"
    log_path = dest.parent / "stdout.log"
    dest.parent.mkdir(parents=True, exist_ok=True)
    convert(args.dataset, args.condition, args.seed)
    import yaml

    cfg_path = args.config or (ROOT / "10_models" / "kgcl" / "configs" / f"{args.dataset}__{args.condition}.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"missing config {cfg_path}; silent fallback is forbidden"
        )
    packed = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    args.neg_k = int(packed.get("n_neg", packed.get("neg_k", args.neg_k)))
    args.lr = packed.get("learning_rate", packed.get("lr", args.lr))
    args.decay = packed.get("decay", args.decay)
    args.layer = packed.get("layer", args.layer)
    # none drops the knowledge view; official sgl flag is the locked reduce.md mapping
    model_name = "sgl" if args.condition == "none" else "kgcl"
    cmd = [
        sys.executable,
        "main.py",
        f"--dataset={args.dataset}",
        f"--model={model_name}",
        f"--seed={args.seed}",
        f"--epochs={args.max_epochs}",
        "--recdim=64",
        "--topks=[10,20]",
        "--tensorboard=0",
        f"--neg_k={args.neg_k}",
    ]
    if args.lr is not None:
        cmd.append(f"--lr={args.lr}")
    if args.decay is not None:
        cmd.append(f"--decay={args.decay}")
    if args.layer is not None:
        cmd.append(f"--layer={args.layer}")
    start = time.time()
    code, log_text = _run_logged(cmd, KGCL_CODE, log_path)
    train_time = time.time() - start
    parsed = parse_kgcl_stdout(log_text)
    extra: dict[str, object] = {}
    notes = ""
    if code != 0:
        payload = _payload(run_id, args, "failed", train_time, parsed, notes=log_text[-4000:])
        _write_metrics(dest, payload)
        return 1
    try:
        extra = _eval_checkpoint(args.dataset, args.seed, model_name)
    except (RuntimeError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        notes = f"checkpoint eval failed; using parsed log. {exc}"[:1000]
    payload = _payload(run_id, args, "ok", train_time, parsed, extra=extra, notes=notes)
    _write_metrics(dest, payload)
    print(f"wrote {dest}")
    print(json.dumps({key: payload[key] for key in (
        "status", "recall@10", "recall@20", "ndcg@10", "ndcg@20", "mrr@10",
        "best_epoch", "total_epochs", "train_time_s", "best_valid_score",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
