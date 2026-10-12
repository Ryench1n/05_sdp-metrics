"""Таван сангийн үр дүнг нэгтгэнэ (run_cv.py, analyze.py-ийн дараа ажиллуулна).

Гаралт: results/pooled/
    summary_long.csv      — сан × загвар × ranking × хэмжүүрийн дундаж
    best_models.csv       — сан бүрт хэмжүүр бүрээр шилдэг загвар (probability ranking)
    ranking_effect.csv    — density − probability: дундаж ялгаа, Wilcoxon p (Holm), Cliff's delta
    global_ranking.csv    — 5 сан × 10 repeat = 50 хосолсон утга дээрх Wilcoxon (загвар бүрт)
    significance.csv      — сан бүрт загвар хоорондын ялгааны тоо (Wilcoxon / bootstrap)
    rank_agreement.csv    — сан бүрт хэмжүүрүүдийн загварын эрэмбийн Kendall's W

Ажиллуулах:  python experiments/pooled.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import DATASETS                                   # noqa: E402
from metrics import DECISION_METRICS, RANKING_METRICS       # noqa: E402
from stats import cliffs_delta, holm, wilcoxon_paired       # noqa: E402

METRICS = DECISION_METRICS + RANKING_METRICS
MODELS = ["LR", "RF", "ET", "XGB"]
LOWER_IS_BETTER = {"IFA"}


def main():
    out = ROOT / "results" / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    folds = pd.concat([pd.read_csv(ROOT / "results" / "cv" / d / "folds.csv").assign(dataset=d)
                       for d in DATASETS])

    # 1. Дундаж утгууд
    summary = folds.groupby(["dataset", "model", "ranking"])[METRICS].mean()
    summary.to_csv(out / "summary_long.csv")

    # 2. Сан бүрт шилдэг загвар (probability ranking)
    rows = []
    for d in DATASETS:
        s = summary.loc[d].xs("probability", level="ranking")
        sd = summary.loc[d].xs("density", level="ranking")
        for m in METRICS:
            best = s[m].idxmin() if m in LOWER_IS_BETTER else s[m].idxmax()
            row = {"dataset": d, "metric": m, "best_probability": best}
            if m in RANKING_METRICS:
                row["best_density"] = sd[m].idxmin() if m in LOWER_IS_BETTER else sd[m].idxmax()
            rows.append(row)
    best = pd.DataFrame(rows)
    best.to_csv(out / "best_models.csv", index=False)

    # 3. Ranking-ийн нөлөө сан бүрт (analyze.py-ийн wilcoxon_ranking.csv-ийг нэгтгэнэ)
    eff = pd.concat([pd.read_csv(ROOT / "results" / "cv" / d / "wilcoxon_ranking.csv").assign(dataset=d)
                     for d in DATASETS])
    eff["diff"] = eff.mean_density - eff.mean_probability
    eff["rel_change_pct"] = 100 * eff["diff"] / eff.mean_probability.replace(0, np.nan)
    eff.to_csv(out / "ranking_effect.csv", index=False)

    # 4. Global Wilcoxon: 5 сан × 10 repeat
    rep = folds.groupby(["dataset", "model", "ranking", "repeat"])[RANKING_METRICS + DECISION_METRICS].mean()
    rows = []
    for metric in RANKING_METRICS + ["MCC"]:
        block = []
        for m in MODELS:
            p = rep.xs((m, "probability"), level=["model", "ranking"])[metric].sort_index().values
            dd = rep.xs((m, "density"), level=["model", "ranking"])[metric].sort_index().values
            stat, pv = wilcoxon_paired(dd, p)
            block.append({"metric": metric, "model": m, "n_pairs": len(p),
                          "mean_probability": p.mean(), "mean_density": dd.mean(),
                          "W": stat, "p": pv, "cliffs_delta": cliffs_delta(dd, p)})
        for r, pa in zip(block, holm([r["p"] for r in block])):
            r["p_holm"] = pa
        rows += block
    pd.DataFrame(rows).to_csv(out / "global_ranking.csv", index=False)

    # 5. Загвар хоорондын ялгааны тоо
    rows = []
    for d in DATASETS:
        w = pd.read_csv(ROOT / "results" / "cv" / d / "wilcoxon_models.csv")
        b = pd.read_csv(ROOT / "results" / "cv" / d / "bootstrap_diff.csv")
        b = b[b.type == "model"]
        rows.append({"dataset": d, "wilcoxon_significant": int(w.significant.sum()),
                     "bootstrap_excludes_zero": int(b.excludes_zero.sum()), "total": len(w),
                     "bootstrap_metrics": ", ".join(sorted(b[b.excludes_zero].metric.unique()))})
    pd.DataFrame(rows).to_csv(out / "significance.csv", index=False)

    # 6. Хэмжүүрүүдийн загварын эрэмбийн нийцэл: Kendall's W ба хосуудын Kendall tau
    rows = []
    for d in DATASETS:
        s = summary.loc[d].xs("probability", level="ranking")[METRICS]
        ranks = pd.DataFrame({m: (s[m].rank(ascending=m in LOWER_IS_BETTER)) for m in METRICS})
        k, n = ranks.shape[1], ranks.shape[0]
        R = ranks.sum(axis=1)
        W = 12 * ((R - R.mean()) ** 2).sum() / (k ** 2 * (n ** 3 - n))
        taus = [kendalltau(ranks[a], ranks[b])[0] for a, b in itertools.combinations(METRICS, 2)]
        rows.append({"dataset": d, "kendall_W": W, "mean_pairwise_tau": np.nanmean(taus),
                     "share_negative_tau": float(np.mean(np.array(taus) < 0)),
                     "distinct_best_models": best[best.dataset == d].best_probability.nunique()})
    pd.DataFrame(rows).to_csv(out / "rank_agreement.csv", index=False)

    pd.set_option("display.width", 220)
    print(pd.read_csv(out / "global_ranking.csv").round(4).to_string(index=False))
    print(pd.read_csv(out / "significance.csv").to_string(index=False))
    print(pd.read_csv(out / "rank_agreement.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
