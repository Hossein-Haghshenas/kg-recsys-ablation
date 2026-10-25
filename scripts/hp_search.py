#!/usr/bin/env python3
"""equal-budget TPE search. n_neg is a searched hyper-parameter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "10_models" / "_shared"))
from run_recbole import DEFAULTS  # noqa: E402

SPACE_PATH = ROOT / "20_results" / "hp_space.yaml"
N_NEG = [1, 2, 5, 10]
REUSE_FULL = {"shuf", "dose25", "dose50", "dose75"}


def _load_space() -> dict:
    return yaml.safe_load(SPACE_PATH.read_text(encoding="utf-8"))


def _suggest(trial, model: str) -> dict:
    n_neg = trial.suggest_categorical("n_neg", N_NEG)
    if model == "kgcl":
        return {
            "n_neg": int(n_neg),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True),
            "decay": trial.suggest_float("decay", 1e-5, 1e-3, log=True),
            "layer": trial.suggest_categorical("layer", [2, 3]),
        }
    cfg = {
        **DEFAULTS[model],
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": int(n_neg)},
        "stopping_step": 20,
        "metrics": ["Recall", "NDCG", "MRR"],
        "topk": [10, 20],
        "valid_metric": "Recall@20",
        "benchmark_filename": ["train", "valid", "test"],
    }
    if model == "lightkg":
        cfg["layer"] = trial.suggest_categorical("layer", [2, 3])
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        cfg["node_dropout_rate"] = trial.suggest_categorical("node_dropout_rate", [0.0, 0.1])
        cfg["mess_dropout_rate"] = trial.suggest_categorical("mess_dropout_rate", [0.0, 0.1])
        cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
    elif model == "lightgcn":
        cfg["n_layers"] = trial.suggest_categorical("n_layers", [2, 3, 4])
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        cfg["reg_weight"] = trial.suggest_float("reg_weight", 1e-5, 1e-3, log=True)
    elif model == "kgat":
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        cfg["mess_dropout"] = trial.suggest_categorical("mess_dropout", [0.0, 0.1])
        cfg["reg_weight"] = trial.suggest_float("reg_weight", 1e-6, 1e-4, log=True)
    elif model == "kgin":
        cfg["n_factors"] = trial.suggest_categorical("n_factors", [4, 8])
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        cfg["node_dropout_rate"] = trial.suggest_categorical("node_dropout_rate", [0.0, 0.5])
        cfg["mess_dropout_rate"] = trial.suggest_categorical("mess_dropout_rate", [0.0, 0.1])
    elif model == "cke":
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        rec_w = trial.suggest_float("reg_weight_rec", 1e-4, 1e-1, log=True)
        kg_w = trial.suggest_float("reg_weight_kg", 1e-4, 1e-1, log=True)
        cfg["reg_weights"] = [rec_w, kg_w]
    else:
        raise ValueError(model)
    return cfg


def _cfg_from_params(model: str, params: dict) -> dict:
    if model == "kgcl":
        return {
            "n_neg": int(params["n_neg"]),
            "learning_rate": params["learning_rate"].item() if hasattr(params["learning_rate"], "item") else params["learning_rate"],
            "decay": params["decay"].item() if hasattr(params["decay"], "item") else params["decay"],
            "layer": int(params["layer"]),
        }
    n_neg = int(params["n_neg"])
    cfg = {
        **DEFAULTS[model],
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": n_neg},
        "stopping_step": 20,
        "metrics": ["Recall", "NDCG", "MRR"],
        "topk": [10, 20],
        "valid_metric": "Recall@20",
        "benchmark_filename": ["train", "valid", "test"],
    }
    skip = {"n_neg", "reg_weight_rec", "reg_weight_kg"}
    for key, value in params.items():
        if key in skip:
            continue
        cfg[key] = value.item() if hasattr(value, "item") else value
    if model == "cke":
        rec_w = params.get("reg_weight_rec")
        kg_w = params.get("reg_weight_kg")
        if rec_w is not None and kg_w is not None:
            cfg["reg_weights"] = [
                rec_w.item() if hasattr(rec_w, "item") else rec_w,
                kg_w.item() if hasattr(kg_w, "item") else kg_w,
            ]
    return cfg


def _baseline_params(model: str) -> dict:
    if model == "lightkg":
        return {
            "n_neg": 10,
            "layer": 2,
            "learning_rate": 0.0005,
            "node_dropout_rate": 0.1,
            "mess_dropout_rate": 0.1,
            "weight_decay": 5e-5,
        }
    if model == "lightgcn":
        return {"n_neg": 1, "n_layers": 3, "learning_rate": 0.001, "reg_weight": 1e-4}
    if model == "kgat":
        return {"n_neg": 1, "learning_rate": 0.0001, "mess_dropout": 0.1, "reg_weight": 1e-5}
    if model == "kgin":
        return {
            "n_neg": 1,
            "n_factors": 4,
            "learning_rate": 0.0001,
            "node_dropout_rate": 0.5,
            "mess_dropout_rate": 0.1,
        }
    if model == "kgcl":
        return {"n_neg": 1, "learning_rate": 0.001, "decay": 1e-4, "layer": 3}
    if model == "cke":
        return {
            "n_neg": 1,
            "learning_rate": 0.0001,
            "reg_weight_rec": 1e-2,
            "reg_weight_kg": 1e-2,
        }
    raise ValueError(model)


def _write_yaml(path: Path, model: str, dataset: str, condition: str, cfg: dict) -> None:
    payload = {"model": model, "dataset": dataset, "kg_file": condition, **cfg}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _run_trial(
    model: str,
    dataset: str,
    condition: str,
    trial_n: int,
    cfg: dict,
    work: Path,
) -> dict:
    yaml_path = work / f"trial_{trial_n:02d}.yaml"
    run_id = f"{model}__{dataset}__{condition}__hp{trial_n:02d}__s0"
    _write_yaml(yaml_path, model, dataset, condition, cfg)
    if model == "kgcl":
        cmd = [
            sys.executable,
            str(ROOT / "10_models" / "kgcl" / "run_kgcl.py"),
            "--dataset",
            dataset,
            "--condition",
            condition,
            "--seed",
            "0",
            "--config",
            str(yaml_path),
            "--run-id",
            run_id,
        ]
        n_neg = int(cfg["n_neg"])
    else:
        cmd = [
            sys.executable,
            str(ROOT / "10_models" / "_shared" / "run_recbole.py"),
            "--model",
            model,
            "--dataset",
            dataset,
            "--condition",
            condition,
            "--seed",
            "0",
            "--config",
            str(yaml_path),
            "--run-id",
            run_id,
        ]
        n_neg = cfg["train_neg_sample_args"]["sample_num"]
    completed = subprocess.run(cmd, cwd=ROOT)
    metrics_path = ROOT / "10_models" / model / "runs" / run_id / "metrics.json"
    if completed.returncode != 0 or not metrics_path.exists():
        return {"ok": False, "valid_recall@20": 0.0, "metrics_path": str(metrics_path)}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    valid = payload.get("best_valid_score")
    try:
        valid_f = float(valid)
    except (TypeError, ValueError):
        valid_f = 0.0
    return {
        "ok": payload.get("status") == "ok",
        "valid_recall@20": valid_f,
        "recall@10": payload.get("recall@10", ""),
        "recall@20": payload.get("recall@20", ""),
        "n_neg": n_neg,
        "train_time_s": payload.get("train_time_s", ""),
        "metrics_path": str(metrics_path),
        "config_hash": payload.get("config_hash", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=("lightgcn", "kgat", "kgin", "lightkg", "kgcl", "cke"),
    )
    parser.add_argument("--dataset", required=True, choices=("lastfm", "bookcrossing", "ml1m"))
    parser.add_argument("--condition", default="full")
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="number of complete trials; defaults to n_trials in 20_results/hp_space.yaml",
    )
    args = parser.parse_args()
    if args.condition in REUSE_FULL:
        raise SystemExit(
            f"{args.condition} reuses the FULL winner; do not search it separately"
        )

    try:
        import optuna
    except ImportError:
        raise SystemExit("optuna is required: pip install optuna")

    space = _load_space()
    n_trials = args.trials if args.trials is not None else int(space.get("n_trials", 15))
    work = ROOT / "20_results" / "hp_search" / f"{args.model}__{args.dataset}__{args.condition}"
    work.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{work / 'study.db'}".replace("\\", "/")
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.model}__{args.dataset}__{args.condition}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(space.get("trial_seed", 0))),
    )
    from optuna.trial import TrialState

    # an interrupted search leaves RUNNING trials; mark them failed so a resumed
    # study still ends with the full number of complete trials
    for trial in study.get_trials(deepcopy=False):
        if trial.state == TrialState.RUNNING:
            study.tell(trial.number, state=TrialState.FAIL)
    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    waiting = [t for t in study.trials if t.state == TrialState.WAITING]
    if not complete and not waiting:
        study.enqueue_trial(_baseline_params(args.model))

    def objective(trial):
        cfg = _suggest(trial, args.model)
        result = _run_trial(args.model, args.dataset, args.condition, trial.number, cfg, work)
        row = {"trial": trial.number, **{k: cfg[k] for k in cfg if k != "train_neg_sample_args"}, **result}
        with (work / "trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")
        if not result["ok"]:
            return 0.0
        return float(result["valid_recall@20"])

    remaining = n_trials - len([t for t in study.trials if t.state == TrialState.COMPLETE])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best_cfg = _cfg_from_params(args.model, study.best_params)
    dest_yaml = ROOT / "10_models" / args.model / "configs" / f"{args.dataset}__{args.condition}.yaml"
    _write_yaml(dest_yaml, args.model, args.dataset, args.condition, best_cfg)
    n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "condition": args.condition,
        "n_trials": n_complete,
        "best_trial": study.best_trial.number,
        "best_value_valid_recall@20": study.best_value,
        "best_params": study.best_params,
        "best_yaml": str(dest_yaml),
    }
    (work / "best.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    if args.model == "lightkg" and args.dataset == "lastfm" and args.condition == "full":
        best_n = study.best_trial.number
        gate_metrics = (
            ROOT / "10_models" / args.model / "runs"
            / f"{args.model}__{args.dataset}__{args.condition}__hp{best_n:02d}__s0"
            / "metrics.json"
        )
        payload = json.loads(gate_metrics.read_text(encoding="utf-8"))
        r10f = float(payload.get("recall@10") or 0.0)
        summary["gate_recall@10"] = r10f
        summary["gate_trial"] = best_n
        (work / "best.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        print("reproduction_gate_recall@10", r10f)
        if not 0.26 <= r10f <= 0.33:
            print(
                f"GATE FAIL: test recall@10={r10f} is outside 0.26-0.33 around LightKG Table 6 (0.2929)",
                file=sys.stderr,
            )
            return 1
        print("GATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
