"""Боломжийн шалгалт: threshold болон ranking-ийн аргын өөрчлөлтөд хэмжүүрүүд
хэрхэн хариу үйлдэл үзүүлэхийг ant-1.7 дээр шалгана.

Ажиллуулах:  python experiments/feasibility.py
Гаралт:      results/feasibility.csv, results/random_baseline.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import load            # noqa: E402
from metrics import evaluate, pofb  # noqa: E402

SEED = 42
N_SPLITS = 10
N_PERMUTATIONS = 2000
THRESHOLDS = (0.5, 0.3)
RANKINGS = ("probability", "density")


def main():
    X, y, loc, _, _ = load("ant-1.7")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # 1) Out-of-fold probability-г нэг удаа тооцоолно
    folds = list(cv.split(X, y))
    prob = np.zeros(len(y))
    for tr, te in folds:
        model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        model.fit(X[tr], y[tr])
        prob[te] = model.predict_proba(X[te])[:, 1]

    # 2) Threshold x ranking хослол бүрт fold-уудын дундаж
    rows = []
    for t in THRESHOLDS:
        for r in RANKINGS:
            per_fold = []
            for _, te in folds:
                p = prob[te]
                pred = (p >= t).astype(int)
                score = p if r == "probability" else p / np.maximum(loc[te], 1)
                per_fold.append(evaluate(y[te], pred, score, loc[te]))
            mean = pd.DataFrame(per_fold).mean()
            rows.append({"threshold": t, "ranking": r, **mean.round(4).to_dict()})
    res = pd.DataFrame(rows)

    # 3) Санамсаргүй ranking-ийн PofB20 тархалт (бүх өгөгдөл дээр)
    rng = np.random.default_rng(0)
    rand = np.array([pofb(y, rng.random(len(y)), loc, 0.2) for _ in range(N_PERMUTATIONS)])
    pooled_prob = pofb(y, prob, loc, 0.2)
    baseline = pd.DataFrame([{
        "random_PofB20_mean": round(rand.mean(), 4),
        "random_PofB20_sd": round(rand.std(), 4),
        "pooled_PofB20_probability": round(pooled_prob, 4),
        "pooled_PofB20_density": round(pofb(y, prob / np.maximum(loc, 1), loc, 0.2), 4),
        "share_random_le_probability": round((rand <= pooled_prob).mean(), 4),
    }])

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    res.to_csv(out / "feasibility.csv", index=False)
    baseline.to_csv(out / "random_baseline.csv", index=False)

    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print()
    print(baseline.to_string(index=False))


if __name__ == "__main__":
    main()
