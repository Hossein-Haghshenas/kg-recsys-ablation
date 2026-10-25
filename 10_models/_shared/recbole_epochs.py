#!/usr/bin/env python3
"""parse RecBole training logs for true epoch counts."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


TRAIN_RE = re.compile(r"epoch (\d+) training")
BEST_RE = re.compile(r"Finished training, best eval result in epoch (\d+)")
TIME_RE = re.compile(
    r"(\d{2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"(\d{4}) (\d{2}):(\d{2}):(\d{2})"
)
MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}
DATASET_FROM_LOG = {
    "lastfm": "lastfm",
    "book-crossing": "bookcrossing",
    "Amazon-book": "amazonbook",
    "ml-1m": "ml1m",
}
MODEL_FROM_LOGDIR = {
    "LightGCN": "lightgcn",
    "KGAT": "kgat",
    "KGIN": "kgin",
    "LightKG": "lightkg",
}
LOG_NAME_RE = re.compile(
    r"^(LightGCN|KGAT|KGIN|LightKG|CKE)-(lastfm|book-crossing|Amazon-book|ml-1m)-"
)


def _parse_timestamp(line: str) -> datetime | None:
    match = TIME_RE.search(line)
    if not match:
        return None
    day, mon, year, hour, minute, second = match.groups()
    return datetime(
        int(year),
        MONTHS[mon],
        int(day),
        int(hour),
        int(minute),
        int(second),
    )


def parse_recbole_log(text: str) -> dict[str, object]:
    last_idx = None
    best_epoch = ""
    first_ts = None
    last_ts = None
    for line in text.splitlines():
        ts = _parse_timestamp(line)
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        train = TRAIN_RE.search(line)
        if train:
            last_idx = int(train.group(1))
        best = BEST_RE.search(line)
        if best:
            best_epoch = int(best.group(1))
    finished = best_epoch != ""
    total_epochs = last_idx + 1 if last_idx is not None else ""
    duration_s = ""
    if first_ts is not None and last_ts is not None:
        duration_s = (last_ts - first_ts).total_seconds()
    smoke = (last_idx is not None and last_idx <= 1) or (
        total_epochs != "" and int(total_epochs) <= 2
    )
    return {
        "best_epoch": best_epoch,
        "total_epochs": total_epochs,
        "last_train_idx": last_idx if last_idx is not None else "",
        "duration_s": duration_s,
        "finished": finished,
        "smoke": smoke,
    }


def parse_log_file(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    payload = parse_recbole_log(text)
    name_match = LOG_NAME_RE.match(path.name)
    log_model = path.parent.name
    payload["log_path"] = str(path)
    payload["log_model"] = log_model
    payload["model"] = MODEL_FROM_LOGDIR.get(log_model, "")
    payload["dataset"] = DATASET_FROM_LOG.get(name_match.group(2), "") if name_match else ""
    return payload


def latest_log_after(log_root: Path, model_class_name: str, after_mtime: float) -> Path | None:
    log_dir = log_root / model_class_name
    if not log_dir.exists():
        return None
    candidates = [
        path for path in log_dir.glob("*.log") if path.stat().st_mtime >= after_mtime - 2
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)
