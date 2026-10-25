#!/usr/bin/env python3
"""load official KGCL weights and print JSON metrics for the runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code" / "official" / "code"


def _at_k(values, ks: list[int], k: int):
    if k not in ks:
        return ""
    return float(values[ks.index(k)])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="kgcl")
    args = parser.parse_args()
    sys.path.insert(0, str(CODE))
    sys.argv = [
        "main.py",
        f"--dataset={args.dataset}",
        f"--model={args.model}",
        f"--seed={args.seed}",
        "--tensorboard=0",
        "--topks=[10,20]",
        "--recdim=64",
        "--load=1",
        "--epochs=0",
    ]
    import torch
    import utils
    import Procedure
    import register
    from register import dataset, kg_dataset
    import world

    recmodel = register.MODELS[world.model_name](world.config, dataset, kg_dataset)
    recmodel = recmodel.to(world.device)
    weight_file = utils.getFileName()
    recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
    result = Procedure.Test(dataset, recmodel, 0, None, world.config["multicore"])
    ks = list(world.topks)
    payload = {
        "recall@10": _at_k(result["recall"], ks, 10),
        "recall@20": _at_k(result["recall"], ks, 20),
        "ndcg@10": _at_k(result["ndcg"], ks, 10),
        "ndcg@20": _at_k(result["ndcg"], ks, 20),
        "mrr@10": _at_k(result["mrr"], ks, 10),
        "n_params": int(sum(param.numel() for param in recmodel.parameters())),
    }
    print("KGCL_METRICS " + json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
