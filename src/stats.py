"""Статистик шалгуурууд: Wilcoxon signed-rank, Holm засвар, Cliff's delta,
percentile bootstrap confidence interval.
"""
import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_paired(a, b):
    """Хосолсон хоёр хэмжилтийн Wilcoxon signed-rank шалгуур (two-sided).

    Бүх ялгаа тэг бол шалгуур тооцох боломжгүй тул (nan, 1.0) буцаана.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return float("nan"), 1.0
    res = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def holm(pvalues):
    """Holm–Bonferroni засвар хийсэн p-утгууд (оролтын дарааллаар)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def cliffs_delta(a, b):
    """Cliff's delta: P(a > b) − P(a < b). Утга нь [−1, 1]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def cliffs_magnitude(d):
    """Romano et al. (2006)-ын босгоор effect size-ийн түвшин."""
    d = abs(d)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def percentile_ci(samples, level=0.95):
    """Bootstrap дээжүүдээс percentile confidence interval."""
    s = np.asarray(samples, float)
    s = s[~np.isnan(s)]
    lo, hi = np.percentile(s, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)
