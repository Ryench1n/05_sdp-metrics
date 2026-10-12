"""run_cv.py-ийн бүртгэлээс статистик шинжилгээ хийнэ.

1. Хураангуй: model × ranking × metric бүрийн дундаж ба стандарт хазайлт (100 fold).
2. Wilcoxon signed-rank (two-sided), repeat түвшний дундаж утгууд дээр (n = repeats):
   - загвар хоорондын бүх хос (probability ranking), metric бүрт Holm засвар;
   - probability ба density ranking (ranking-based metric), загвар бүрт; Holm засвар.
   Нэг repeat доторх fold-ууд хоорондоо хамааралтай тул repeat-ийн дунджийг ашиглана.
   Effect size: fold түвшний утгууд дээрх Cliff's delta.
3. Paired bootstrap 95% CI: дээж бүрт repeat-ийг санамсаргүй сонгож, модулиудыг
   буцаан тавих замаар дахин түүвэрлэн, бүх загварын хэмжүүрийг ижил дээж дээр
   тооцно. Хэмжүүр бүрийн болон загвар/ranking хоорондын ялгааны CI гаргана.
4. Загваруудын эрэмбэ: хэмжүүр бүрээр аль загвар хамгийн сайн гарч байгааг хоёр
   ranking-ийн аргаар харьцуулна; repeat бүрийн ялагчийг тоолж тогтвортой байдлыг шалгана.

Ажиллуулах:  python experiments/analyze.py [--boot 2000]
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import DECISION_METRICS, RANKING_METRICS, evaluate   # noqa: E402
from stats import (cliffs_delta, cliffs_magnitude, holm,          # noqa: E402
                   percentile_ci, wilcoxon_paired)

METRICS = DECISION_METRICS + RANKING_METRICS
LOWER_IS_BETTER = {"IFA"}
THRESHOLD = 0.5


def model_tests(folds, models):
    """Загвар хоорондын Wilcoxon (probability ranking)."""
    d = folds[folds.ranking == "probability"]
    rep = d.groupby(["model", "repeat"])[METRICS].mean()
    rows = []
    for metric in METRICS:
        block = []
        for a, b in itertools.combinations(models, 2):
            xa, xb = rep.loc[a, metric].values, rep.loc[b, metric].values
            stat, p = wilcoxon_paired(xa, xb)
            fa = d.loc[d.model == a, metric].values
            fb = d.loc[d.model == b, metric].values
            delta = cliffs_delta(fa, fb)
            block.append({"metric": metric, "model_a": a, "model_b": b,
                          "mean_a": fa.mean(), "mean_b": fb.mean(),
                          "W": stat, "p": p, "cliffs_delta": delta,
                          "magnitude": cliffs_magnitude(delta)})
        adj = holm([r["p"] for r in block])
        for r, pa in zip(block, adj):
            r["p_holm"] = pa
            r["significant"] = bool(pa < 0.05)
        rows += block
    return pd.DataFrame(rows)


def ranking_tests(folds, models):
    """Probability ба density ranking-ийн Wilcoxon (загвар бүрт)."""
    rep = folds.groupby(["model", "ranking", "repeat"])[RANKING_METRICS].mean()
    rows = []
    for metric in RANKING_METRICS:
        block = []
        for m in models:
            xp = rep.loc[(m, "probability"), metric].values
            xd = rep.loc[(m, "density"), metric].values
            stat, p = wilcoxon_paired(xd, xp)
            fp = folds.loc[(folds.model == m) & (folds.ranking == "probability"), metric].values
            fd = folds.loc[(folds.model == m) & (folds.ranking == "density"), metric].values
            delta = cliffs_delta(fd, fp)
            block.append({"metric": metric, "model": m,
                          "mean_probability": fp.mean(), "mean_density": fd.mean(),
                          "W": stat, "p": p, "cliffs_delta": delta,
                          "magnitude": cliffs_magnitude(delta)})
        adj = holm([r["p"] for r in block])
        for r, pa in zip(block, adj):
            r["p_holm"] = pa
            r["significant"] = bool(pa < 0.05)
        rows += block
    return pd.DataFrame(rows)


def bootstrap(oof, models, n_boot, seed):
    """Paired bootstrap: бүх загвар, ranking ижил дээж дээр үнэлэгдэнэ."""
    repeats = sorted(oof.repeat.unique())
    base = oof[oof.model == models[0]].sort_values(["repeat", "module"])
    n = base.module.nunique()
    y = base[base.repeat == repeats[0]].y.to_numpy()
    effort = base[base.repeat == repeats[0]]["effort"].to_numpy()
    prob = {m: oof[oof.model == m].sort_values(["repeat", "module"])
            .prob.to_numpy().reshape(len(repeats), n) for m in models}

    rng = np.random.default_rng(seed)
    draws = []
    for b in range(n_boot):
        r = rng.integers(len(repeats))
        idx = rng.integers(0, n, n)
        yb, eb = y[idx], effort[idx]
        if yb.sum() == 0:
            continue
        for m in models:
            p = prob[m][r, idx]
            pred = (p >= THRESHOLD).astype(int)
            for ranking, score in (("probability", p), ("density", p / np.maximum(eb, 1))):
                draws.append({"b": b, "model": m, "ranking": ranking,
                              **evaluate(yb, pred, score, eb)})
    draws = pd.DataFrame(draws)

    ci = []
    for (m, ranking), g in draws.groupby(["model", "ranking"]):
        for metric in METRICS:
            lo, hi = percentile_ci(g[metric])
            ci.append({"model": m, "ranking": ranking, "metric": metric,
                       "boot_mean": g[metric].mean(), "ci_low": lo, "ci_high": hi})

    wide = draws.set_index(["b", "model", "ranking"])
    diff = []
    prob_only = wide.xs("probability", level="ranking")
    for a, c in itertools.combinations(models, 2):
        d = prob_only.xs(a, level="model")[METRICS] - prob_only.xs(c, level="model")[METRICS]
        for metric in METRICS:
            lo, hi = percentile_ci(d[metric])
            diff.append({"comparison": f"{a} − {c}", "type": "model", "metric": metric,
                         "diff_mean": d[metric].mean(), "ci_low": lo, "ci_high": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    for m in models:
        d = (wide.xs((m, "density"), level=["model", "ranking"])[RANKING_METRICS]
             - wide.xs((m, "probability"), level=["model", "ranking"])[RANKING_METRICS])
        for metric in RANKING_METRICS:
            lo, hi = percentile_ci(d[metric])
            diff.append({"comparison": f"{m}: density − probability", "type": "ranking",
                         "metric": metric, "diff_mean": d[metric].mean(),
                         "ci_low": lo, "ci_high": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(ci), pd.DataFrame(diff), draws.b.nunique()


def model_ranks(summary, models, ranking="probability"):
    """Хэмжүүр бүрээр загваруудыг эрэмбэлнэ (1 = хамгийн сайн)."""
    s = summary.xs(ranking, level="ranking")
    metrics = METRICS if ranking == "probability" else RANKING_METRICS
    rows = {}
    for metric in metrics:
        means = s[(metric, "mean")]
        asc = metric in LOWER_IS_BETTER
        rows[metric] = means.rank(ascending=asc, method="min").astype(int)
    return pd.DataFrame(rows).T[models]


def repeat_winners(folds, models):
    """Repeat бүрт хэмжүүр тус бүрээр хамгийн сайн загварыг тоолно (probability ranking).

    Тэнцсэн тохиолдолд тэнцсэн загвар бүрт 1/k оноо өгнө.
    """
    d = folds[folds.ranking == "probability"]
    rep = d.groupby(["repeat", "model"])[METRICS].mean()
    counts = pd.DataFrame(0.0, index=METRICS, columns=models)
    for r in rep.index.get_level_values("repeat").unique():
        block = rep.xs(r, level="repeat")
        for metric in METRICS:
            v = block[metric]
            best = v.min() if metric in LOWER_IS_BETTER else v.max()
            winners = v.index[np.isclose(v, best)]
            counts.loc[metric, winners] += 1 / len(winners)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = ROOT / "results" / "cv"
    folds = pd.read_csv(out / "folds.csv")
    oof = pd.read_csv(out / "oof.csv.gz")
    models = list(dict.fromkeys(folds.model))

    summary = folds.groupby(["model", "ranking"])[METRICS].agg(["mean", "std"])
    summary.to_csv(out / "summary.csv")
    mt = model_tests(folds, models)
    mt.to_csv(out / "wilcoxon_models.csv", index=False)
    rt = ranking_tests(folds, models)
    rt.to_csv(out / "wilcoxon_ranking.csv", index=False)
    ranks = model_ranks(summary, models)
    ranks.to_csv(out / "model_ranks.csv")
    ranks_d = model_ranks(summary, models, "density")
    ranks_d.to_csv(out / "model_ranks_density.csv")
    winners = repeat_winners(folds, models)
    winners.to_csv(out / "repeat_winners.csv")
    ci, diff, n_ok = bootstrap(oof, models, args.boot, args.seed)
    ci.to_csv(out / "bootstrap_ci.csv", index=False)
    diff.to_csv(out / "bootstrap_diff.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"=== mean over {folds.groupby(['repeat', 'fold']).ngroups} folds ===")
    print(summary.xs("mean", axis=1, level=1).round(3).to_string())
    print("\n=== model ranks per metric (probability ranking; 1 = best) ===")
    print(ranks.to_string())
    print("\n=== model ranks, density ranking ===")
    print(ranks_d.to_string())
    print("\n=== repeat winners (out of 10 repeats) ===")
    print(winners.round(1).to_string())
    print(f"\n=== Wilcoxon, model pairs: significant after Holm = "
          f"{int(mt.significant.sum())}/{len(mt)} ===")
    print(f"=== Wilcoxon, density vs probability: significant after Holm = "
          f"{int(rt.significant.sum())}/{len(rt)} ===")
    print(f"=== bootstrap: {n_ok} draws ===")


if __name__ == "__main__":
    main()
