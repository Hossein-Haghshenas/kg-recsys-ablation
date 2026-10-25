#!/usr/bin/env python3
"""build E1–E6 tables and figures from 20_results/runs.csv only."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from stats import binomial_k_of_n, fmt, fmt_pm, mean_std, paired_delta


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "20_results" / "runs.csv"
TABLES = ROOT / "30_analysis" / "tables"
FIGURES = ROOT / "30_analysis" / "figures"
FAILURES = ROOT / "log" / "failures.md"

# knowledge density from LightKG Table 7; used as labels only, never correlated (n=3)
KAPPA = {"lastfm": 0.73, "bookcrossing": 6.15, "amazonbook": 0.20, "ml1m": 0.66}
METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20", "mrr@10")
PRIMARY = "recall@20"
MIN_SEEDS = 3


def _float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_ok_rows() -> list[dict[str, str]]:
    with RUNS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("status") == "ok"]


def _group(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["model"], row["condition"])].append(row)
    return grouped


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _tex_escape(value: object) -> str:
    return str(value).replace("_", r"\_")


def _write_tex(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]]) -> None:
    cols = "l" * len(headers)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_tex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}",
        r"\hline",
        " & ".join(_tex_escape(h) for h in headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_tex_escape(cell) for cell in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_failure(text: str) -> None:
    FAILURES.parent.mkdir(parents=True, exist_ok=True)
    prev = FAILURES.read_text(encoding="utf-8") if FAILURES.exists() else "# failures\n"
    if text in prev:
        return
    with FAILURES.open("a", encoding="utf-8") as handle:
        handle.write("\n" + text + "\n")


def _cell_values(rows: list[dict[str, str]], metric: str) -> list[float]:
    values = []
    for row in rows:
        value = _float(row.get(metric, ""))
        if value is not None:
            values.append(value)
    return values


def _by_seed(rows: list[dict[str, str]], metric: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        value = _float(row.get(metric, ""))
        if value is None:
            continue
        out[int(row["seed"])] = value
    return out


def e1(grouped) -> list[dict[str, object]]:
    rows = []
    for key in sorted(grouped):
        dataset, model, condition = key
        cell = grouped[key]
        record: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "condition": condition,
            "n_seeds": len(cell),
            "seeds_below_protocol": int(len(cell) < MIN_SEEDS),
            "code_sources": ",".join(sorted({row["code_source"] for row in cell})),
        }
        for metric in METRICS:
            values = _cell_values(cell, metric)
            mean, std, n = mean_std(values)
            record[f"{metric}_mean"] = fmt(mean)
            record[f"{metric}_std"] = fmt(std)
            record[f"{metric}_n"] = n
            record[f"{metric}_pm"] = fmt_pm(mean, std)
        rows.append(record)
    fields = (
        ["dataset", "model", "condition", "n_seeds", "seeds_below_protocol", "code_sources"]
        + [f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std", "n", "pm")]
    )
    _write_csv(TABLES / "e1_accuracy.csv", fields, rows)
    tex_rows = [
        [
            row["dataset"],
            row["model"],
            row["condition"],
            str(row["n_seeds"]),
            row["recall@10_pm"],
            row["recall@20_pm"],
            row["ndcg@10_pm"],
            row["ndcg@20_pm"],
            row["mrr@10_pm"],
        ]
        for row in rows
    ]
    _write_tex(
        TABLES / "e1_accuracy.tex",
        "E1 accuracy mean ± std over seeds. Cells with n<3 violate the frozen protocol.",
        "tab:e1",
        ["dataset", "model", "condition", "n", "R@10", "R@20", "N@10", "N@20", "MRR@10"],
        tex_rows,
    )
    return rows


def e2(grouped) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    models = sorted({model for _, model, _ in grouped if model != "lightgcn"})
    datasets = sorted({dataset for dataset, _, _ in grouped})
    rows = []
    for dataset in datasets:
        for model in models:
            full_rows = grouped.get((dataset, model, "full"), [])
            none_rows = grouped.get((dataset, model, "none"), [])
            full_seeds = _by_seed(full_rows, PRIMARY)
            none_seeds = _by_seed(none_rows, PRIMARY)
            result = paired_delta(full_seeds, none_seeds)
            n_full = len(full_rows)
            n_none = len(none_rows)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "metric": PRIMARY,
                    "kappa_table7": KAPPA.get(dataset, ""),
                    "n_full": n_full,
                    "n_none": n_none,
                    "n_paired": result.n,
                    "seeds_below_protocol": int(result.n < MIN_SEEDS),
                    "mean_full": fmt(None if not full_seeds else float(sum(full_seeds.values()) / len(full_seeds))),
                    "mean_none": fmt(None if not none_seeds else float(sum(none_seeds.values()) / len(none_seeds))),
                    "delta_full_minus_none": fmt(result.mean),
                    "ci95_low": fmt(result.ci_low),
                    "ci95_high": fmt(result.ci_high),
                    "paired_t_pvalue": fmt(result.t_pvalue, 4),
                    "paired_seeds": ",".join(str(seed) for seed in result.seeds),
                    "ci_method": "bootstrap_percentile" if result.ci_low is not None else "",
                }
            )
    fields = [
        "dataset",
        "model",
        "metric",
        "kappa_table7",
        "n_full",
        "n_none",
        "n_paired",
        "seeds_below_protocol",
        "mean_full",
        "mean_none",
        "delta_full_minus_none",
        "ci95_low",
        "ci95_high",
        "paired_t_pvalue",
        "paired_seeds",
        "ci_method",
    ]
    _write_csv(TABLES / "e2_kg_effect.csv", fields, rows)
    countable = [row for row in rows if row["n_paired"] >= 1 and row["delta_full_minus_none"] != ""]
    k_all = sum(1 for row in countable if float(row["delta_full_minus_none"]) > 0)
    n_all = len(countable)
    countable3 = [row for row in countable if int(row["n_paired"]) >= MIN_SEEDS]
    k3 = sum(1 for row in countable3 if float(row["delta_full_minus_none"]) > 0)
    n3 = len(countable3)
    bin_all = binomial_k_of_n(k_all, n_all)
    bin3 = binomial_k_of_n(k3, n3)
    counts = [
        {
            "definition": "FULL_mean_gt_NONE on paired cells with n_paired>=1",
            "k": bin_all.k,
            "n": bin_all.n,
            "proportion": fmt(None if bin_all.n == 0 else bin_all.k / bin_all.n),
            "ci95_low": fmt(bin_all.ci_low),
            "ci95_high": fmt(bin_all.ci_high),
            "ci_method": bin_all.method,
        },
        {
            "definition": "FULL_mean_gt_NONE on paired cells with n_paired>=3",
            "k": bin3.k,
            "n": bin3.n,
            "proportion": fmt(None if bin3.n == 0 else bin3.k / bin3.n),
            "ci95_low": fmt(bin3.ci_low),
            "ci95_high": fmt(bin3.ci_high),
            "ci_method": bin3.method,
        },
    ]
    _write_csv(
        TABLES / "e2_binomial_cells.csv",
        ["definition", "k", "n", "proportion", "ci95_low", "ci95_high", "ci_method"],
        counts,
    )
    _write_tex(
        TABLES / "e2_kg_effect.tex",
        "E2 paired FULL-NONE delta on Recall@20. CI is a seed-level bootstrap; p from a paired t-test. kappa from LightKG Table 7 is a label only.",
        "tab:e2",
        ["dataset", "model", "n", "mean FULL", "mean NONE", "delta", "CI95", "p"],
        [
            [
                row["dataset"],
                row["model"],
                str(row["n_paired"]),
                row["mean_full"],
                row["mean_none"],
                row["delta_full_minus_none"],
                f"[{row['ci95_low']}, {row['ci95_high']}]" if row["ci95_low"] else "",
                row["paired_t_pvalue"],
            ]
            for row in rows
        ],
    )
    return rows, counts


def e3(grouped) -> list[dict[str, object]]:
    rows = []
    models = sorted({model for _, model, _ in grouped if model != "lightgcn"})
    datasets = sorted({dataset for dataset, _, _ in grouped})
    for dataset in datasets:
        for model in models:
            record: dict[str, object] = {"dataset": dataset, "model": model, "metric": PRIMARY}
            by_cond = {}
            for condition in ("full", "shuf", "none"):
                cell = grouped.get((dataset, model, condition), [])
                values = _cell_values(cell, PRIMARY)
                mean, std, n = mean_std(values)
                record[f"{condition}_n"] = n
                record[f"{condition}_mean"] = fmt(mean)
                record[f"{condition}_std"] = fmt(std)
                record[f"{condition}_pm"] = fmt_pm(mean, std)
                by_cond[condition] = mean
            full_m, shuf_m, none_m = by_cond["full"], by_cond["shuf"], by_cond["none"]
            record["full_minus_none"] = fmt(None if full_m is None or none_m is None else full_m - none_m)
            record["full_minus_shuf"] = fmt(None if full_m is None or shuf_m is None else full_m - shuf_m)
            record["shuf_minus_none"] = fmt(None if shuf_m is None or none_m is None else shuf_m - none_m)
            pattern = ""
            if full_m is not None and shuf_m is not None and none_m is not None:
                if abs(full_m - shuf_m) < abs(shuf_m - none_m) and shuf_m > none_m:
                    pattern = "structure_not_content"
                elif full_m > shuf_m and shuf_m > none_m:
                    pattern = "content_sensitive"
                else:
                    pattern = "other"
            record["descriptive_pattern"] = pattern
            rows.append(record)
    fields = [
        "dataset",
        "model",
        "metric",
        "full_n",
        "full_mean",
        "full_std",
        "full_pm",
        "shuf_n",
        "shuf_mean",
        "shuf_std",
        "shuf_pm",
        "none_n",
        "none_mean",
        "none_std",
        "none_pm",
        "full_minus_none",
        "full_minus_shuf",
        "shuf_minus_none",
        "descriptive_pattern",
    ]
    _write_csv(TABLES / "e3_full_shuf_none.csv", fields, rows)
    _write_tex(
        TABLES / "e3_full_shuf_none.tex",
        "E3 FULL / SHUF / NONE on Recall@20. Pattern labels are descriptive, not tests.",
        "tab:e3",
        ["dataset", "model", "FULL", "SHUF", "NONE", "FULL-SHUF", "SHUF-NONE"],
        [
            [
                row["dataset"],
                row["model"],
                row["full_pm"],
                row["shuf_pm"],
                row["none_pm"],
                row["full_minus_shuf"],
                row["shuf_minus_none"],
            ]
            for row in rows
        ],
    )
    return rows


# kg fraction axis for E4; SHUF is a permutation, not a dose, so it stays off this plot
E4_AXIS = (
    ("none", 0.00),
    ("dose25", 0.25),
    ("dose50", 0.50),
    ("dose75", 0.75),
    ("full", 1.00),
)
E4_MODELS = ("kgat", "kgin", "lightkg")
E4_DATASETS = ("lastfm", "bookcrossing", "amazonbook")


def e4(grouped) -> list[dict[str, object]]:
    rows = []
    for dataset in E4_DATASETS:
        for model in E4_MODELS:
            for condition, frac in E4_AXIS:
                cell = grouped.get((dataset, model, condition), [])
                values = _cell_values(cell, PRIMARY)
                mean, std, n = mean_std(values)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "condition": condition,
                        "kg_frac": f"{frac:.2f}",
                        "kappa_table7": KAPPA.get(dataset, ""),
                        "n_seeds": n,
                        "seeds_below_protocol": int(n > 0 and n < MIN_SEEDS),
                        "recall@20_mean": fmt(mean),
                        "recall@20_std": fmt(std),
                        "recall@20_pm": fmt_pm(mean, std),
                    }
                )
    fields = [
        "dataset",
        "model",
        "condition",
        "kg_frac",
        "kappa_table7",
        "n_seeds",
        "seeds_below_protocol",
        "recall@20_mean",
        "recall@20_std",
        "recall@20_pm",
    ]
    _write_csv(TABLES / "e4_dose.csv", fields, rows)
    tex_rows = [
        [
            row["dataset"],
            row["model"],
            row["condition"],
            row["kg_frac"],
            str(row["n_seeds"]),
            row["recall@20_pm"],
        ]
        for row in rows
        if row["n_seeds"]
    ]
    _write_tex(
        TABLES / "e4_dose.tex",
        "E4 dose-response on Recall@20. NONE=0, dose25/50/75, FULL=1. "
        "Amazon-Book is deferred. SHUF is not a dose and is omitted. "
        "kappa from LightKG Table 7 is a panel label only.",
        "tab:e4",
        ["dataset", "model", "condition", "kg frac", "n", "R@20"],
        tex_rows,
    )
    (TABLES / "e4_dose.note.txt").write_text(
        "E4 is a descriptive dose curve on the locked KG-fraction axis "
        "(NONE, dose25, dose50, dose75, FULL). SHUF is excluded. "
        "Amazon-Book was deferred in protocol.yaml. n=3 auxiliary seeds. "
        "Do not correlate slope against kappa (n_datasets<10).\n",
        encoding="utf-8",
    )
    missing_path = TABLES / "e4_dose.missing.txt"
    if missing_path.exists():
        missing_path.unlink()
    measured = sum(1 for row in rows if row["n_seeds"])
    print(f"E4 rows: {len(rows)}; measured cells: {measured}")
    return rows


def e5(grouped) -> list[dict[str, object]]:
    rows = []
    for key in sorted(grouped):
        dataset, model, condition = key
        cell = grouped[key]
        params = _cell_values(cell, "n_params")
        tpe = _cell_values(cell, "time_per_epoch_s")
        mem = _cell_values(cell, "gpu_mem_mb")
        p_mean, p_std, p_n = mean_std(params)
        t_mean, t_std, t_n = mean_std(tpe)
        m_mean, m_std, m_n = mean_std(mem)
        unique_params = sorted(set(int(value) for value in params))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "n_seeds": len(cell),
                "n_params": unique_params[0] if len(unique_params) == 1 else "",
                "n_params_note": "" if len(unique_params) <= 1 else f"nonunique:{unique_params}",
                "time_per_epoch_s_mean": fmt(t_mean, 2),
                "time_per_epoch_s_std": fmt(t_std, 2),
                "time_per_epoch_s_n": t_n,
                "time_per_epoch_s_pm": fmt_pm(t_mean, t_std, 2),
                "gpu_mem_mb_mean": fmt(m_mean, 2),
                "gpu_mem_mb_std": fmt(m_std, 2),
                "gpu_mem_mb_n": m_n,
                "gpu_mem_mb_pm": fmt_pm(m_mean, m_std, 2),
            }
        )
    fields = [
        "dataset",
        "model",
        "condition",
        "n_seeds",
        "n_params",
        "n_params_note",
        "time_per_epoch_s_mean",
        "time_per_epoch_s_std",
        "time_per_epoch_s_n",
        "time_per_epoch_s_pm",
        "gpu_mem_mb_mean",
        "gpu_mem_mb_std",
        "gpu_mem_mb_n",
        "gpu_mem_mb_pm",
    ]
    _write_csv(TABLES / "e5_cost.csv", fields, rows)
    _write_tex(
        TABLES / "e5_cost.tex",
        "E5 cost under an equal 15-trial search budget, not under equal n_neg. "
        "time/epoch uses rebuilt epoch counts where a log mapping exists; empty if unmapped.",
        "tab:e5",
        ["dataset", "model", "condition", "n_params", "s/epoch", "gpu_mem_mb"],
        [
            [
                row["dataset"],
                row["model"],
                row["condition"],
                str(row["n_params"]),
                row["time_per_epoch_s_pm"],
                row["gpu_mem_mb_pm"],
            ]
            for row in rows
        ],
    )
    (TABLES / "e5_cost.note.txt").write_text(
        "E5 reports cost under an equal 15-trial hyper-parameter search budget. "
        "n_neg is searched, so wall-clock and time/epoch are not comparable under equal n_neg.\n",
        encoding="utf-8",
    )
    return rows


def e6(grouped, e2_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    delta_lookup = {
        (row["dataset"], row["model"]): _float(row["delta_full_minus_none"]) for row in e2_rows
    }
    rows = []
    for key in sorted(grouped):
        dataset, model, condition = key
        values = _cell_values(grouped[key], PRIMARY)
        mean, std, n = mean_std(values)
        delta = delta_lookup.get((dataset, model))
        seed_std_gt_abs_delta = ""
        if std is not None and delta is not None:
            seed_std_gt_abs_delta = int(std > abs(delta))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "condition": condition,
                "metric": PRIMARY,
                "n_seeds": n,
                "mean": fmt(mean),
                "seed_std": fmt(std),
                "abs_delta_full_none": fmt(None if delta is None else abs(delta)),
                "seed_std_gt_abs_delta": seed_std_gt_abs_delta,
            }
        )
    fields = [
        "dataset",
        "model",
        "condition",
        "metric",
        "n_seeds",
        "mean",
        "seed_std",
        "abs_delta_full_none",
        "seed_std_gt_abs_delta",
    ]
    _write_csv(TABLES / "e6_seed_noise.csv", fields, rows)
    _write_tex(
        TABLES / "e6_seed_noise.tex",
        "E6 seed std versus |FULL-NONE| on Recall@20. A 1 in the last column means seed noise exceeds the claimed delta.",
        "tab:e6",
        ["dataset", "model", "condition", "n", "mean", "seed std", "|delta|", "std>|delta|"],
        [
            [
                row["dataset"],
                row["model"],
                row["condition"],
                str(row["n_seeds"]),
                row["mean"],
                row["seed_std"],
                row["abs_delta_full_none"],
                str(row["seed_std_gt_abs_delta"]),
            ]
            for row in rows
        ],
    )
    return rows


def _try_figures(e1_rows, e2_rows, e3_rows, e4_rows, e6_rows) -> list[str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        msg = f"E1-E6 figures skipped: matplotlib missing ({exc})"
        _append_failure(msg)
        print(msg)
        return []

    written = []

    def _bar_e1():
        datasets = sorted({row["dataset"] for row in e1_rows})
        fig, axes = plt.subplots(1, max(len(datasets), 1), figsize=(12, 4), squeeze=False)
        for ax, dataset in zip(axes[0], datasets):
            subset = [row for row in e1_rows if row["dataset"] == dataset]
            labels = [f"{row['model']}/{row['condition']}" for row in subset]
            heights = [_float(row["recall@20_mean"]) or 0.0 for row in subset]
            ax.bar(range(len(labels)), heights, color="#4C78A8")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.set_title(dataset)
            ax.set_ylabel("Recall@20")
        fig.tight_layout()
        path = FIGURES / "e1_accuracy.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    def _forest_e2():
        usable = [row for row in e2_rows if row["delta_full_minus_none"] != ""]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(usable) + 1)))
        ys = list(range(len(usable)))
        xs = [_float(row["delta_full_minus_none"]) or 0.0 for row in usable]
        xerr_low = []
        xerr_high = []
        for row, x in zip(usable, xs):
            lo = _float(row["ci95_low"])
            hi = _float(row["ci95_high"])
            if lo is None or hi is None:
                xerr_low.append(0.0)
                xerr_high.append(0.0)
            else:
                xerr_low.append(x - lo)
                xerr_high.append(hi - x)
        ax.errorbar(xs, ys, xerr=[xerr_low, xerr_high], fmt="o", color="#4C78A8")
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{row['dataset']}/{row['model']}" for row in usable], fontsize=8)
        ax.set_xlabel("FULL - NONE Recall@20 (bootstrap 95% CI)")
        fig.tight_layout()
        path = FIGURES / "e2_kg_effect.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    def _grouped_e3():
        usable = [row for row in e3_rows if row["full_mean"] or row["shuf_mean"] or row["none_mean"]]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = list(range(len(usable)))
        width = 0.25
        for offset, key, color in (
            (-width, "full_mean", "#4C78A8"),
            (0.0, "shuf_mean", "#F58518"),
            (width, "none_mean", "#54A24B"),
        ):
            heights = [_float(row[key]) or 0.0 for row in usable]
            ax.bar([i + offset for i in x], heights, width=width, color=color, label=key.replace("_mean", "").upper())
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['dataset']}/{row['model']}" for row in usable], rotation=90, fontsize=7)
        ax.set_ylabel("Recall@20")
        ax.legend()
        fig.tight_layout()
        path = FIGURES / "e3_full_shuf_none.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    def _scatter_e6():
        points = [
            row
            for row in e6_rows
            if row["condition"] == "full" and row["seed_std"] != "" and row["abs_delta_full_none"] != ""
        ]
        fig, ax = plt.subplots(figsize=(5, 5))
        xs = [_float(row["abs_delta_full_none"]) or 0.0 for row in points]
        ys = [_float(row["seed_std"]) or 0.0 for row in points]
        ax.scatter(xs, ys, color="#4C78A8")
        limit = max(xs + ys + [0.01])
        ax.plot([0, limit], [0, limit], color="#333333", linewidth=0.8)
        ax.set_xlabel("|FULL-NONE| Recall@20")
        ax.set_ylabel("seed std (FULL)")
        fig.tight_layout()
        path = FIGURES / "e6_seed_noise.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    def _curves_e4():
        missing = FIGURES / "e4_dose.missing.txt"
        measured = [row for row in e4_rows if row["n_seeds"]]
        if not measured:
            missing.write_text(
                "E4 figure not drawn: 0 dose/NONE/FULL cells in runs.csv.\n",
                encoding="utf-8",
            )
            written.append(str(missing))
            return
        if missing.exists():
            missing.unlink()
        colors = {"kgat": "#4C78A8", "kgin": "#F58518", "lightkg": "#54A24B"}
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=False)
        for ax, dataset in zip(axes, E4_DATASETS):
            kappa = KAPPA.get(dataset, "")
            ax.set_title(f"{dataset}" + (f" (κ={kappa})" if kappa != "" else ""))
            ax.set_xlabel("KG fraction")
            ax.set_xticks([frac for _, frac in E4_AXIS])
            plotted = False
            for model in E4_MODELS:
                xs = []
                ys = []
                yerr = []
                for condition, frac in E4_AXIS:
                    match = next(
                        (
                            row
                            for row in e4_rows
                            if row["dataset"] == dataset
                            and row["model"] == model
                            and row["condition"] == condition
                        ),
                        None,
                    )
                    mean = _float(match["recall@20_mean"]) if match else None
                    if mean is None:
                        continue
                    xs.append(frac)
                    ys.append(mean)
                    std = _float(match["recall@20_std"]) if match else None
                    yerr.append(0.0 if std is None else std)
                    plotted = True
                if xs:
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=yerr,
                        fmt="-o",
                        color=colors[model],
                        label=model,
                    )
            if not plotted:
                ax.text(0.5, 0.5, "deferred", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(-0.05, 1.05)
        axes[0].set_ylabel("Recall@20")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        path = FIGURES / "e4_dose.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    _bar_e1()
    _forest_e2()
    _grouped_e3()
    _curves_e4()
    _scatter_e6()
    return written


def main() -> int:
    if not RUNS.exists():
        raise SystemExit("runs.csv missing — collect results first")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = _load_ok_rows()
    grouped = _group(rows)
    e1_rows = e1(grouped)
    e2_rows, e2_counts = e2(grouped)
    e3_rows = e3(grouped)
    e4_rows = e4(grouped)
    e5_rows = e5(grouped)
    e6_rows = e6(grouped, e2_rows)
    figures = _try_figures(e1_rows, e2_rows, e3_rows, e4_rows, e6_rows)
    print(f"ok runs: {len(rows)}")
    print(f"E1 rows: {len(e1_rows)}")
    print(f"E2 cells: {len(e2_rows)}; binomial: {e2_counts}")
    print(f"E3 rows: {len(e3_rows)}")
    print(f"E5 rows: {len(e5_rows)}")
    print(f"E6 rows: {len(e6_rows)}")
    print("tables", TABLES)
    print("figures", figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
