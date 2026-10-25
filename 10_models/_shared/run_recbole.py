#!/usr/bin/env python3
"""train one RecBole-backed model and write metrics.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from recbole_epochs import latest_log_after, parse_log_file, parse_recbole_log

import numpy as np

# recbole 1.1.1 still uses np.float; do not alias np.bool (breaks scipy)
if not hasattr(np, "float"):
    np.float = float

ROOT = Path(__file__).resolve().parents[2]
LIGHTKG_ROOT = ROOT / "10_models" / "lightkg" / "code" / "official"
DATASET_FILES = {
    "lastfm": "lastfm",
    "bookcrossing": "book-crossing",
    "amazonbook": "Amazon-book",
    "ml1m": "ml-1m",
}
RECBOLLE_NAME = {
    "lastfm": "lastfm",
    "bookcrossing": "book-crossing",
    "amazonbook": "Amazon-book",
    "ml1m": "ml-1m",
}
MODEL_CLASS_PATH = {
    "lightgcn": ("model.LightGCN", "LightGCN"),
    "kgat": ("model.KGAT", "KGAT"),
    "kgin": ("model.KGIN", "KGIN"),
    "lightkg": ("LightKG", "LightKG"),
    "cke": ("model.CKE", "CKE"),
}
CODE_SOURCE = {
    "lightgcn": "recbole",
    "kgat": "recbole",
    "kgin": "recbole",
    "lightkg": "lightkg-official-model/recbole-trainer",
    "cke": "lightkg-official-model/recbole-trainer",
}
DEFAULTS = {
    "lightgcn": {
        "embedding_size": 64,
        "n_layers": 3,
        "reg_weight": 1e-4,
        "require_pow": True,
        "learning_rate": 0.001,
    },
    "kgat": {
        "embedding_size": 64,
        "kg_embedding_size": 64,
        "layers": [64, 32, 16],
        "aggregator_type": "bi",
        "mess_dropout": 0.1,
        "reg_weight": 1e-5,
        "learning_rate": 0.0001,
    },
    "kgin": {
        "embedding_size": 64,
        "n_factors": 4,
        "context_hops": 3,
        "node_dropout_rate": 0.5,
        "mess_dropout_rate": 0.1,
        "ind": "cosine",
        "sim_regularity": 1e-4,
        "reg_weight": 1e-5,
        "temperature": 0.2,
        "learning_rate": 0.0001,
    },
    "lightkg": {
        "embedding_size": 64,
        "layer": 2,
        "node_dropout_rate": 0.1,
        "mess_dropout_rate": 0.1,
        "weight_decay": 5e-5,
        "cos_loss": 1,
        "user_loss": 1e-8,
        "item_loss": 1e-7,
        "learning_rate": 0.0005,
    },
    "cke": {
        "embedding_size": 64,
        "kg_embedding_size": 64,
        "reg_weights": [1e-2, 1e-2],
        "learning_rate": 0.0001,
    },
}


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _variant_name(condition: str, seed: int) -> str:
    if condition == "full":
        return "full.txt"
    if condition == "none":
        return "none.txt"
    if condition == "shuf":
        return f"shuf_s{seed % 3}.txt"
    if condition.startswith("dose"):
        return f"{condition}_s{seed % 3}.txt"
    raise ValueError(f"unknown condition: {condition}")


def _write_dummy_kg(dest: Path, processed_kg: Path, link_path: Path) -> str:
    # recbole rejects a header-only kg; one item-linked self-loop keeps the loader alive
    text = processed_kg.read_text(encoding="utf-8").splitlines()
    header = text[0]
    entity = "0"
    if link_path.exists():
        link_lines = link_path.read_text(encoding="utf-8").splitlines()
        if len(link_lines) > 1:
            cols = link_lines[0].split("\t")
            idx = next((i for i, col in enumerate(cols) if col.startswith("entity_id")), 1)
            entity = link_lines[1].split("\t")[idx]
    dest.write_text(
        f"{header}\n{entity}\tnone_rel\t{entity}\n",
        encoding="utf-8",
    )
    return "empty kg replaced by one dummy self-loop"


def _prepare_dataset(dataset: str, condition: str, seed: int) -> tuple[Path, str]:
    stem = DATASET_FILES[dataset]
    rec_name = RECBOLLE_NAME[dataset]
    work = ROOT / "00_data" / "work" / f"{dataset}__{condition}__s{seed}" / rec_name
    if work.exists():
        shutil.rmtree(work)
    src = ROOT / "00_data" / "processed" / dataset
    shutil.copytree(src, work)
    note = ""
    if dataset != "lightgcn_skip":
        variant = ROOT / "00_data" / "kg_variants" / dataset / _variant_name(condition, seed)
        dest_kg = work / f"{stem}.kg"
        if condition == "none":
            note = _write_dummy_kg(dest_kg, src / f"{stem}.kg", src / f"{stem}.link")
        else:
            shutil.copy2(variant, dest_kg)
    return work.parent, note


def resolve_config_path(
    model: str,
    dataset: str,
    condition: str,
    config_arg: Path | str | None = None,
) -> Path:
    if config_arg:
        path = Path(config_arg)
        if not path.exists():
            raise FileNotFoundError(f"missing --config {path}")
        return path
    path = ROOT / "10_models" / model / "configs" / f"{dataset}__{condition}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"missing config {path}; silent fallback to another dataset or "
            f"condition is forbidden"
        )
    return path


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def assemble_config_dict(
    model: str, user_cfg: dict, args: argparse.Namespace, data_path: Path | str
) -> dict:
    # yaml-overridable keys sit above **user_cfg so published configs can win
    yaml_overridable = {
        "stopping_step": min(20, args.max_epochs),
        "eval_step": 1,
        "metrics": ["Recall", "NDCG", "MRR"],
        "topk": [10, 20],
        "valid_metric": "Recall@20",
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
        "eval_args": {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "group_by": "user",
            "order": "RO",
            "mode": "full",
        },
    }
    return {
        **DEFAULTS[model],
        **yaml_overridable,
        **user_cfg,
        "seed": args.seed,
        "reproducibility": True,
        "data_path": str(data_path),
        "benchmark_filename": ["train", "valid", "test"],
        "epochs": args.max_epochs,
        "device": args.device,
        "gpu_id": 0,
        "use_gpu": str(args.device).startswith("cuda"),
        "show_progress": False,
        "enable_amp": False,
    }


def _jsonable(value):
    if hasattr(value, "items"):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _empty_metrics(run_id: str, args: argparse.Namespace, note: str, status: str) -> dict:
    return {
        "run_id": run_id,
        "model": args.model,
        "dataset": args.dataset,
        "condition": args.condition,
        "seed": args.seed,
        "code_source": CODE_SOURCE[args.model],
        "commit": _git_commit(LIGHTKG_ROOT),
        "config_hash": "",
        "recall@10": "",
        "recall@20": "",
        "ndcg@10": "",
        "ndcg@20": "",
        "mrr@10": "",
        "best_epoch": "",
        "total_epochs": "",
        "train_time_s": "",
        "time_per_epoch_s": "",
        "gpu_mem_mb": "",
        "n_params": "",
        "best_valid_score": "",
        "status": status,
        "notes": note,
    }


def _write_metrics(model: str, run_id: str, payload: dict) -> Path:
    dest = ROOT / "10_models" / model / "runs" / run_id / "metrics.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def _pick(result: dict, *keys: str):
    for key in keys:
        if key in result:
            return result[key]
        lower = {k.lower(): v for k, v in result.items()}
        if key.lower() in lower:
            return lower[key.lower()]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_CLASS_PATH))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FILES))
    parser.add_argument("--condition", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"{args.model}__{args.dataset}__{args.condition}__s{args.seed}"
    if args.model == "lightgcn" and args.condition != "full":
        payload = _empty_metrics(run_id, args, "lightgcn has no kg condition", "ok")
        _write_metrics(args.model, run_id, payload)
        print(f"skipped kg condition for lightgcn: {run_id}")
        return 0

    sys.path.insert(0, str(LIGHTKG_ROOT))
    try:
        config_path = resolve_config_path(
            args.model, args.dataset, args.condition, args.config
        )
        data_path, note = _prepare_dataset(args.dataset, args.condition, args.seed)
        module_name, class_name = MODEL_CLASS_PATH[args.model]
        module = __import__(module_name, fromlist=[class_name])
        model_cls = getattr(module, class_name)

        import torch

        _orig_load = torch.load

        def _load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _load
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.trainer import KGTrainer
        from recbole.utils import init_logger, init_seed

        user_cfg = _load_yaml(config_path)
        config_dict = assemble_config_dict(args.model, user_cfg, args, data_path)
        config_hash = hashlib.sha256(json.dumps(config_dict, sort_keys=True, default=str).encode()).hexdigest()[:12]
        config = Config(
            model=model_cls,
            dataset=RECBOLLE_NAME[args.dataset],
            config_dict=config_dict,
        )
        resolved = {
            "config_path": str(config_path),
            "train_neg_sample_args": _jsonable(config["train_neg_sample_args"]),
            "stopping_step": _jsonable(config["stopping_step"]),
            "metrics": _jsonable(config["metrics"]),
            "topk": _jsonable(config["topk"]),
            "valid_metric": _jsonable(config["valid_metric"]),
            "eval_args": _jsonable(config["eval_args"]),
            "seed": _jsonable(config["seed"]),
            "epochs": _jsonable(config["epochs"]),
        }
        print(f"resolved_config {json.dumps(resolved, default=str)}", flush=True)
        run_dir = ROOT / "10_models" / args.model / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
        )
        init_seed(config["seed"], config["reproducibility"])
        log_t0 = time.time()
        init_logger(config)
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = model_cls(config, train_data._dataset).to(config["device"])
        n_params = sum(param.numel() for param in model.parameters())
        trainer = KGTrainer(config, model)
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=False
        )
        train_time = time.time() - start
        test_result = trainer.evaluate(test_data) or {}
        gpu_mem = ""
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            gpu_mem = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        log_path = latest_log_after(ROOT / "log", class_name, log_t0)
        epoch_info = parse_log_file(log_path) if log_path else parse_recbole_log("")
        best_epoch = epoch_info["best_epoch"]
        total_epochs = epoch_info["total_epochs"]
        if total_epochs == "":
            time_per_epoch_s = ""
        else:
            time_per_epoch_s = round(train_time / max(int(total_epochs), 1), 2)
        payload = {
            "run_id": run_id,
            "model": args.model,
            "dataset": args.dataset,
            "condition": args.condition,
            "seed": args.seed,
            "code_source": CODE_SOURCE[args.model],
            "commit": _git_commit(LIGHTKG_ROOT),
            "config_hash": config_hash,
            "recall@10": _pick(test_result, "recall@10", "Recall@10"),
            "recall@20": _pick(test_result, "recall@20", "Recall@20"),
            "ndcg@10": _pick(test_result, "ndcg@10", "NDCG@10"),
            "ndcg@20": _pick(test_result, "ndcg@20", "NDCG@20"),
            "mrr@10": _pick(test_result, "mrr@10", "MRR@10"),
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "train_time_s": round(train_time, 2),
            "time_per_epoch_s": time_per_epoch_s,
            "gpu_mem_mb": gpu_mem,
            "n_params": n_params,
            "best_valid_score": best_valid_score,
            "status": "ok",
            "notes": note,
        }
        dest = _write_metrics(args.model, run_id, payload)
        print(f"wrote {dest}")
        print(json.dumps(test_result, default=str))
        return 0
    except KeyboardInterrupt:
        payload = _empty_metrics(run_id, args, "interrupted by user", "interrupted")
        _write_metrics(args.model, run_id, payload)
        print("interrupted")
        return 130
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "failed"
        payload = _empty_metrics(run_id, args, f"{status}: {exc}", status)
        _write_metrics(args.model, run_id, payload)
        traceback.print_exc()
        return 1
    except FileNotFoundError as exc:
        print(f"fatal: {exc}", flush=True)
        traceback.print_exc()
        return 2
    except Exception as exc:
        payload = _empty_metrics(run_id, args, f"failed: {type(exc).__name__}: {exc}", "failed")
        _write_metrics(args.model, run_id, payload)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
