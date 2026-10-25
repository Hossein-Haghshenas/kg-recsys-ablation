#!/usr/bin/env python3
"""Reproduce every number in the paper's measurement tables from the run log.

Tables reproduced (numbering of the manuscript):
  ourcost      capacity and cost          -> tables/paper_cost.csv
  ourablation  cell means and SDs         -> tables/paper_cells.csv
  ourstats     all contrasts              -> tables/paper_contrasts.csv
  ourdose      dose-response trends       -> tables/paper_dose.csv
  ourneg       negative-sampling spread   -> tables/paper_nneg.csv

Conventions (Section 8.1 of the paper):
  * one reproducible stall (LightKG / Book-Crossing / SHUF / seed 2: early
    stopping ended the run on its starting loss plateau) is excluded from the
    primary analysis and reported as a pooled sensitivity;
  * contrasts FULL-NONE, SHUF-NONE, FULL-SHUF per knowledge-consuming model and
    dataset; Welch t-test, Welch-Satterthwaite 95% CI, Hedges's g;
  * Holm correction over every contrast in runs.csv taken as one family;
  * dose trends: Jonckheere-Terpstra with an exact two-sided p-value taken from
    the complete permutation distribution of run labels over the five levels,
    conditional on the observed ties (no sampling); Holm over rows;
  * a cell is flagged when its seed CV in Recall@20 exceeds 10%.

Robustness checks (Section 8.1 and the limits of Section 8.8):
  * paired t-test over common seeds, same Holm family     -> printed
  * exact Mann-Whitney and complete separation per contrast -> tables/paper_rank.csv
  * SHUF contrasts with the graph draw (seed % 3) as unit  -> tables/paper_shuf_draws.csv

Usage:  python 30_analysis/paper_statistics.py
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "20_results" / "runs.csv"
HP = ROOT / "20_results" / "hp_search"
OUT = ROOT / "30_analysis" / "tables"

METRIC = "recall@20"
KG_MODELS = ["kgat", "kgin", "kgcl", "lightkg", "cke"]
CONTRASTS = [("presence", "full", "none"), ("structure", "shuf", "none"), ("content", "full", "shuf")]
DOSE_LEVELS = [("none", 0), ("dose25", 25), ("dose50", 50), ("dose75", 75), ("full", 100)]
DOSE_MODELS = ["kgat", "kgin", "lightkg"]
STALLED = {("lightkg", "bookcrossing", "shuf", 2)}
SHUF_DRAWS = 3                      # seeds 0..4 use permuted graph seed % 3
CV_FLAG = 0.10


def load_runs() -> pd.DataFrame:
    d = pd.read_csv(RUNS)
    d = d[d["status"] == "ok"].copy()
    d["stalled"] = [
        (m, ds, c, int(s)) in STALLED
        for m, ds, c, s in zip(d["model"], d["dataset"], d["condition"], d["seed"])
    ]
    return d


def cell(d: pd.DataFrame, model: str, dataset: str, cond: str,
         metric: str = METRIC, pooled: bool = False) -> np.ndarray:
    x = d[(d.model == model) & (d.dataset == dataset) & (d.condition == cond)]
    if not pooled:
        x = x[~x.stalled]
    return x.sort_values("seed")[metric].to_numpy(dtype=float)


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t = (m1 - m2) / se
    q = stats.t.ppf(0.975, df)
    sp = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    g = (1 - 3 / (4 * (n1 + n2) - 9)) * (m1 - m2) / sp
    return {"delta": m1 - m2, "pct": 100 * (m1 - m2) / m2, "ci_lo": m1 - m2 - q * se,
            "ci_hi": m1 - m2 + q * se, "g": g, "p": 2 * stats.t.sf(abs(t), df), "df": df,
            "n1": n1, "n2": n2}


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out


def cells_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in sorted(d.dataset.unique()):
        for m in ["lightgcn"] + KG_MODELS:
            for c in ["full", "none", "shuf"]:
                for pooled in (False, True):
                    r = cell(d, m, ds, c, pooled=pooled)
                    if len(r) == 0 or (pooled and len(r) == len(cell(d, m, ds, c))):
                        continue
                    n20 = cell(d, m, ds, c, "ndcg@20", pooled=pooled)
                    cv = r.std(ddof=1) / r.mean()
                    rows.append({"dataset": ds, "model": m, "condition": c, "pooled": pooled,
                                 "n": len(r), "recall20_mean": r.mean(), "recall20_sd": r.std(ddof=1),
                                 "ndcg20_mean": n20.mean(), "ndcg20_sd": n20.std(ddof=1),
                                 "cv": cv, "flag": cv > CV_FLAG})
    return pd.DataFrame(rows)


def contrasts_table(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fam, a, b in CONTRASTS:
        for ds in sorted(d.dataset.unique()):
            for m in KG_MODELS:
                x, y = cell(d, m, ds, a), cell(d, m, ds, b)
                if len(x) < 2 or len(y) < 2:
                    continue
                r = welch(x, y)
                r.update(family=fam, dataset=ds, model=m)
                rows.append(r)
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"])
    # pooled sensitivity for the stalled cell
    s = t.copy()
    for (m, ds, c, _seed) in STALLED:
        for fam, a, b in CONTRASTS:
            if c not in (a, b):
                continue
            x = cell(d, m, ds, a, pooled=True)
            y = cell(d, m, ds, b, pooled=True)
            idx = s[(s.family == fam) & (s.dataset == ds) & (s.model == m)].index
            if len(idx):
                r = welch(x, y)
                for k, v in r.items():
                    s.loc[idx[0], k] = v
    s["p_holm"] = holm(s["p"])
    # paired robustness check (the analysis first planned)
    paired = []
    for fam, a, b in CONTRASTS:
        for ds in sorted(d.dataset.unique()):
            for m in KG_MODELS:
                xa = d[(d.model == m) & (d.dataset == ds) & (d.condition == a) & ~d.stalled].set_index("seed")[METRIC]
                xb = d[(d.model == m) & (d.dataset == ds) & (d.condition == b) & ~d.stalled].set_index("seed")[METRIC]
                common = xa.index.intersection(xb.index)
                if len(common) < 2:
                    continue
                paired.append(stats.ttest_rel(xa[common], xb[common]).pvalue)
    t["p_holm_paired"] = holm(np.array(paired))
    return t, s


def _compositions(total: int, room: list[int]):
    """Every way of splitting `total` tied runs across the levels, within capacity."""
    if len(room) == 1:
        if total <= room[0]:
            yield (total,)
        return
    for x in range(min(total, room[0]) + 1):
        for rest in _compositions(total - x, room[1:]):
            yield (x,) + rest


def jonckheere_exact(groups: list[np.ndarray]) -> tuple[float, float, int]:
    """Jonckheere-Terpstra statistic and its exact two-sided permutation p-value.

    The statistic counts, over every ordered pair of levels, the pairs of runs in
    which the run at the higher level scores more (ties count 1/2). Its null
    distribution is the distribution over all N!/(n_1!...n_k!) assignments of the
    run labels to the levels with the observed values held fixed, so ties are
    conditioned on. It is built by dynamic programming over the distinct values
    in ascending order, which counts every assignment without listing them.
    """
    k = len(groups)
    sizes = [len(g) for g in groups]
    obs2 = 0                                    # twice the statistic, an integer
    for i in range(k):
        for j in range(i + 1, k):
            obs2 += 2 * int((groups[j][None, :] > groups[i][:, None]).sum()) \
                + int((groups[j][None, :] == groups[i][:, None]).sum())
    _, tie_sizes = np.unique(np.concatenate(groups), return_counts=True)
    dist: dict[tuple, dict[int, int]] = {tuple([0] * k): {0: 1}}
    for t in map(int, tie_sizes):
        new: dict[tuple, dict[int, int]] = {}
        for state, sub in dist.items():
            room = [sizes[i] - state[i] for i in range(k)]
            for a in _compositions(t, room):
                ways = math.factorial(t)
                for x in a:
                    ways //= math.factorial(x)
                add = sum(2 * a[j] * state[i] + a[i] * a[j]
                          for i in range(k) for j in range(i + 1, k))
                target = new.setdefault(tuple(s + x for s, x in zip(state, a)), {})
                for j2, c in sub.items():
                    target[j2 + add] = target.get(j2 + add, 0) + c * ways
        dist = new
    final = dist[tuple(sizes)]
    total = sum(final.values())
    assert total == math.factorial(sum(sizes)) // math.prod(math.factorial(n) for n in sizes)
    upper = sum(c for j2, c in final.items() if j2 >= obs2)
    lower = sum(c for j2, c in final.items() if j2 <= obs2)
    return obs2 / 2, min(1.0, 2 * min(upper, lower) / total), total


def dose_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in sorted(d.dataset.unique()):
        for m in DOSE_MODELS:
            groups = [cell(d, m, ds, c) for c, _ in DOSE_LEVELS]
            if any(len(g) == 0 for g in groups):
                continue
            jt, p, arrangements = jonckheere_exact(groups)
            sizes = [len(g) for g in groups]
            jmax = sum(sizes[i] * sizes[j] for i in range(5) for j in range(i + 1, 5))
            means = [g.mean() for g in groups]
            rows.append({"dataset": ds, "model": m,
                         **{f"mean_{f}": mu for (_, f), mu in zip(DOSE_LEVELS, means)},
                         **{f"sd_{f}": g.std(ddof=1) for (_, f), g in zip(DOSE_LEVELS, groups)},
                         "spearman_rho": stats.spearmanr([f for _, f in DOSE_LEVELS], means).correlation,
                         "jt": jt, "jt_max": jmax, "arrangements": arrangements, "p": p,
                         "max_dev_from_own_none_pct": 100 * max(abs(x - means[0]) / means[0] for x in means)})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"])
    return t


def cost_table(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    core = d[d.condition.isin(["full", "none", "shuf"])]
    dose = d[d.condition.str.startswith("dose")]
    rows = []
    for m in ["lightgcn"] + KG_MODELS:
        x = core[core.model == m]
        if x.empty:
            continue
        row = {"model": m, "s_per_epoch_median": x.time_per_epoch_s.median(),
               "gpu_hours_core": x.train_time_s.sum() / 3600, "peak_mb": x.gpu_mem_mb.max()}
        for ds in sorted(d.dataset.unique()):
            for c in ["full", "none", "shuf"]:
                p = sorted(set(x[(x.dataset == ds) & (x.condition == c)].n_params.dropna().astype(int)))
                row[f"params_{ds}_{c}"] = ";".join(map(str, p))
        rows.append(row)
    trials = load_trials()
    totals = {"core_runs": len(core), "core_gpu_h": core.train_time_s.sum() / 3600,
              "dose_runs": len(dose), "dose_gpu_h": dose.train_time_s.sum() / 3600,
              "trials": len(trials), "search_gpu_h": trials.train_time_s.sum() / 3600}
    totals["total_gpu_h"] = totals["core_gpu_h"] + totals["dose_gpu_h"] + totals["search_gpu_h"]
    return pd.DataFrame(rows), totals


def load_trials() -> pd.DataFrame:
    rows = []
    datasets = set(pd.read_csv(RUNS).dataset.unique())
    for f in sorted(glob.glob(str(HP / "*" / "trials.jsonl"))):
        m, ds, c = Path(f).parent.name.split("__")
        if ds not in datasets:          # a search whose seed runs are not yet in runs.csv
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                j = json.loads(line)
                if j.get("ok", True) and j.get("n_neg") is not None:
                    rows.append({"model": m, "dataset": ds, "condition": c, "n_neg": int(j["n_neg"]),
                                 "valid": j["valid_recall@20"], "train_time_s": j["train_time_s"]})
    return pd.DataFrame(rows)


def nneg_table() -> pd.DataFrame:
    t = load_trials()
    rows = []
    for (m, ds, c), g in t.groupby(["model", "dataset", "condition"]):
        best = g.groupby("n_neg")["valid"].max()
        rows.append({"model": m, "dataset": ds, "condition": c, "values_sampled": len(best),
                     "spread_pct": 100 * (best.max() - best.min()) / best.min(),
                     "lo": best.min(), "hi": best.max(),
                     "winning_n_neg": int(g.loc[g["valid"].idxmax(), "n_neg"])})
    return pd.DataFrame(rows)


def rank_table(d: pd.DataFrame, con: pd.DataFrame) -> pd.DataFrame:
    """Exact two-sided Mann-Whitney p and complete separation for every contrast.

    With five runs per condition the smallest attainable p is 2/252, so after a
    Holm correction over thirty contrasts no rank test can fall below 0.238; the
    informative output is whether the two sets of runs separate completely.
    """
    pairs = {fam: (a, b) for fam, a, b in CONTRASTS}
    rows = []
    for _, r in con.iterrows():
        a_c, b_c = pairs[r.family]
        x, y = cell(d, r.model, r.dataset, a_c), cell(d, r.model, r.dataset, b_c)
        sep = "first>second" if x.min() > y.max() else ("second>first" if y.min() > x.max() else "")
        rows.append({"family": r.family, "dataset": r.dataset, "model": r.model,
                     "n1": len(x), "n2": len(y),
                     "p_mannwhitney": stats.mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue,
                     "smallest_attainable_p": 2 / math.comb(len(x) + len(y), len(x)),
                     "complete_separation": sep, "p_welch": r.p, "p_welch_holm": r.p_holm})
    t = pd.DataFrame(rows)
    t["p_mannwhitney_holm"] = holm(t["p_mannwhitney"])
    return t


def shuf_draw_table(d: pd.DataFrame, con: pd.DataFrame) -> pd.DataFrame:
    """Re-test the SHUF contrasts with the permuted-graph draw as the unit.

    Runs that share a draw (seed % 3) are averaged; the draw means are compared
    with the FULL or NONE seeds by Welch's test, and the draw-level p replaces the
    seed-level p of that contrast inside the same thirty-contrast Holm family.
    """
    alt = con.copy()
    rows = []
    for i, r in con.iterrows():
        if r.family not in ("content", "structure"):
            continue
        x = d[(d.model == r.model) & (d.dataset == r.dataset) & (d.condition == "shuf") & ~d.stalled]
        draws = x.assign(draw=x.seed.astype(int) % SHUF_DRAWS).groupby("draw")["recall@20"]
        dm = draws.mean().to_numpy()
        if r.family == "content":
            w = welch(cell(d, r.model, r.dataset, "full"), dm)
        else:
            w = welch(dm, cell(d, r.model, r.dataset, "none"))
        alt.loc[i, "p"] = w["p"]
        rows.append({"family": r.family, "dataset": r.dataset, "model": r.model,
                     "draws": len(dm), "runs_per_draw": "/".join(map(str, draws.size().to_numpy())),
                     "delta_draw": w["delta"], "ci_lo": w["ci_lo"], "ci_hi": w["ci_hi"], "df": w["df"],
                     "p_draw": w["p"], "p_seed": r.p, "p_holm_seed": r.p_holm})
    alt["p_holm"] = holm(alt["p"])
    t = pd.DataFrame(rows)
    t["p_holm_draw"] = [alt[(alt.family == f) & (alt.dataset == s_) & (alt.model == m)].p_holm.iloc[0]
                        for f, s_, m in zip(t.family, t.dataset, t.model)]
    t.attrs["verdicts_changed"] = int(((alt.p_holm < 0.05) != (con.p_holm < 0.05)).sum())
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_runs()
    cells = cells_table(d)
    cells.to_csv(OUT / "paper_cells.csv", index=False)
    con, pooled = contrasts_table(d)
    con.to_csv(OUT / "paper_contrasts.csv", index=False)
    pooled.to_csv(OUT / "paper_contrasts_pooled_sensitivity.csv", index=False)
    dose = dose_table(d)
    dose.to_csv(OUT / "paper_dose.csv", index=False)
    cost, totals = cost_table(d)
    cost.to_csv(OUT / "paper_cost.csv", index=False)
    nneg = nneg_table()
    nneg.to_csv(OUT / "paper_nneg.csv", index=False)
    rank = rank_table(d, con)
    rank.to_csv(OUT / "paper_rank.csv", index=False)
    draws = shuf_draw_table(d, con)
    draws.to_csv(OUT / "paper_shuf_draws.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"contrasts in the Holm family: {len(con)}")
    print(con[["family", "dataset", "model", "delta", "pct", "ci_lo", "ci_hi", "g", "p", "p_holm"]]
          .round(4).to_string(index=False))
    changed = int(((pooled.p_holm < 0.05) != (con.p_holm < 0.05)).sum())
    print(f"verdicts changed by pooling the stalled run: {changed}")
    print(f"verdicts differing under the paired analysis: {int(((con.p_holm < 0.05) != (con.p_holm_paired < 0.05)).sum())}")
    print(dose[["dataset", "model", "spearman_rho", "jt", "jt_max", "p", "p_holm"]].to_string(index=False))
    print({k: round(v, 2) if isinstance(v, float) else v for k, v in totals.items()})
    sig = rank.p_welch_holm < 0.05
    print(f"exact rank test: smallest Holm-corrected p {rank.p_mannwhitney_holm.min():.3f}; "
          f"complete separation in {int((sig & (rank.complete_separation != '')).sum())} of {int(sig.sum())} "
          f"significant contrasts and in {int((~sig & (rank.complete_separation != '')).sum())} others")
    print(f"SHUF contrasts with the draw as unit: verdicts changed {draws.attrs['verdicts_changed']}")
    print(draws[draws.p_holm_draw < 0.05][["family", "dataset", "model", "draws", "p_draw", "p_holm_draw"]]
          .to_string(index=False))
    print(f"flagged cells (CV > {CV_FLAG:.0%}):")
    print(cells[cells.flag][["dataset", "model", "condition", "pooled", "cv"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
