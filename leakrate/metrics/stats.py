"""Uncertainty quantification (METHODS.md §5.1).

Mentions cluster inside documents, so the primary interval is a cluster bootstrap over documents.
The Wilson interval is computed only as the (anti-conservative) reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class RateCI:
    rate: float
    lo: float
    hi: float
    n: int               # denominator (mentions / entities / documents)
    k: int               # numerator
    clusters: int        # documents
    method: str
    wilson_lo: float | None = None
    wilson_hi: float | None = None
    rule_of_three: float | None = None

    def to_json(self) -> dict:
        return asdict(self)

    def fmt(self, pct: bool = True) -> str:
        if self.n == 0:
            return "n/a"
        f = 100 if pct else 1
        return f"{self.rate * f:.1f} [{self.lo * f:.1f}, {self.hi * f:.1f}]" + ("%" if pct else "")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cluster_bootstrap_rate(num: np.ndarray, den: np.ndarray, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> RateCI:
    """num[i], den[i] = leaked / total mentions in document i. Resamples documents with replacement."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    D = len(num)
    k, n = int(num.sum()), int(den.sum())
    if n == 0 or D == 0:
        return RateCI(float("nan"), float("nan"), float("nan"), n, k, D, "cluster_bootstrap")
    rate = k / n
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=(B, D))
    bn = num[idx].sum(axis=1)
    bd = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rates = np.where(bd > 0, bn / bd, np.nan)
    rates = rates[~np.isnan(rates)]
    lo, hi = (np.quantile(rates, [alpha / 2, 1 - alpha / 2]) if len(rates) else (float("nan"), float("nan")))
    wl, wh = wilson(k, n)
    r3 = 3.0 / n if k == 0 else None
    return RateCI(rate, float(lo), float(hi), n, k, D, "cluster_bootstrap", wl, wh, r3)


def paired_cluster_bootstrap_diff(num_a, den_a, num_b, den_b, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for rate(A) - rate(B) resampling the same documents for both systems."""
    num_a, den_a, num_b, den_b = (np.asarray(x, dtype=float) for x in (num_a, den_a, num_b, den_b))
    D = len(num_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, D, size=(B, D))
    with np.errstate(invalid="ignore", divide="ignore"):
        ra = num_a[idx].sum(1) / den_a[idx].sum(1)
        rb = num_b[idx].sum(1) / den_b[idx].sum(1)
    diff = ra - rb
    diff = diff[~np.isnan(diff)]
    point = (num_a.sum() / den_a.sum() if den_a.sum() else float("nan")) - (num_b.sum() / den_b.sum() if den_b.sum() else float("nan"))
    lo, hi = np.quantile(diff, [alpha / 2, 1 - alpha / 2]) if len(diff) else (float("nan"), float("nan"))
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "B": B, "clusters": D}


def paired_bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """CI for mean(a) - mean(b) over paired observations (downstream quality Δ)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)), "B": B}
