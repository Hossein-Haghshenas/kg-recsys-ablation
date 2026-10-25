#!/usr/bin/env python3
"""paired deltas, bootstrap CIs, and binomial CIs. no correlation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest, bootstrap, ttest_rel


BOOTSTRAP_N = 10_000
RNG = 0


@dataclass(frozen=True)
class PairedDelta:
    n: int
    mean: float | None
    ci_low: float | None
    ci_high: float | None
    t_pvalue: float | None
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class BinomialCount:
    k: int
    n: int
    ci_low: float | None
    ci_high: float | None
    method: str


def _as_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def paired_delta(full_by_seed: dict[int, float], none_by_seed: dict[int, float]) -> PairedDelta:
    seeds = tuple(sorted(set(full_by_seed) & set(none_by_seed)))
    if not seeds:
        return PairedDelta(0, None, None, None, None, ())
    deltas = np.array([full_by_seed[seed] - none_by_seed[seed] for seed in seeds], dtype=float)
    mean = float(np.mean(deltas))
    pvalue = None
    ci_low = None
    ci_high = None
    if len(deltas) >= 2:
        t_result = ttest_rel(
            [full_by_seed[seed] for seed in seeds],
            [none_by_seed[seed] for seed in seeds],
        )
        pvalue = float(t_result.pvalue)
        boot = bootstrap(
            (deltas,),
            np.mean,
            n_resamples=BOOTSTRAP_N,
            confidence_level=0.95,
            method="percentile",
            random_state=RNG,
        )
        ci_low = float(boot.confidence_interval.low)
        ci_high = float(boot.confidence_interval.high)
    return PairedDelta(len(seeds), mean, ci_low, ci_high, pvalue, seeds)


def binomial_k_of_n(k: int, n: int) -> BinomialCount:
    if n <= 0:
        return BinomialCount(k, n, None, None, "none")
    interval = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return BinomialCount(k, n, float(interval.low), float(interval.high), "wilson")


def mean_std(values: list[float]) -> tuple[float | None, float | None, int]:
    clean = [value for value in (_as_float(item) for item in values) if value is not None]
    if not clean:
        return None, None, 0
    arr = np.array(clean, dtype=float)
    std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else None
    return float(np.mean(arr)), std, len(arr)


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def fmt_pm(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return ""
    if std is None:
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} ± {fmt(std, digits)}"
