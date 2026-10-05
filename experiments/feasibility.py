"""Боломжийн шалгалт: threshold болон ranking-ийн аргын өөрчлөлтөд хэмжүүрүүд
хэрхэн хариу үйлдэл үзүүлэхийг ant-1.7 дээр, хоёр загвараар шалгана.

Ажиллуулах:  python experiments/feasibility.py
Гаралт:      results/feasibility_<feature_set>.csv, results/random_baseline.csv

"nodes" feature set-д багшийн өгөгдөл data/advisor/ant-1.7_nodes.csv хэрэгтэй.
Файл байхгүй бол зөвхөн "ck" feature set дээр ажиллана.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import ADVISOR_DIR, load      # noqa: E402
from metrics import evaluate, pofb      # noqa: E402

SEED = 42
N_SPLITS = 10
N_PERMUTATIONS = 2000
THRESHOLDS = (0.5, 0.3)
RANKINGS = ("probability", "density")
MODELS = {
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "LR": lambda: make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, random_state=SEED)),
}


def run(feature_set):
    X, y, loc, features, _ = load("ant-1.7", feature_set)
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
    rows, pooled = [], []
    for model_name, make in MODELS.items():
        prob = np.zeros(len(y))
        for tr, te in folds:
            prob[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        density = prob / np.maximum(loc, 1)
        pooled.append({"feature_set": feature_set, "model": model_name,
                       "pooled_PofB20_probability": round(pofb(y, prob, loc, 0.2), 4),
                       "pooled_PofB20_density": round(pofb(y, density, loc, 0.2), 4)})
        for t in THRESHOLDS:
            for r in RANKINGS:
                per_fold = []
                for _, te in folds:
                    p = prob[te]
                    score = p if r == "probability" else p / np.maximum(loc[te], 1)
                    per_fold.append(evaluate(y[te], (p >= t).astype(int), score, loc[te]))
                mean = pd.DataFrame(per_fold).mean().round(4).to_dict()
                rows.append({"model": model_name, "threshold": t, "ranking": r, **mean})
    print(f"\n=== feature set: {feature_set}  ({len(features)} features, N={len(y)}, K={y.sum()}) ===")
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(ROOT / "results" / f"feasibility_{feature_set}.csv", index=False)

    # Санамсаргүй ranking-ийн PofB20 тархалт (тухайн өгөгдөл дээр)
    rng = np.random.default_rng(0)
    rand = np.array([pofb(y, rng.random(len(y)), loc, 0.2) for _ in range(N_PERMUTATIONS)])
    for row in pooled:
        row["random_PofB20_mean"] = round(rand.mean(), 4)
        row["random_PofB20_sd"] = round(rand.std(), 4)
        row["share_random_le_probability"] = round((rand <= row["pooled_PofB20_probability"]).mean(), 4)
    return pooled


def main():
    (ROOT / "results").mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    sets = ["nodes", "ck"] if (ADVISOR_DIR / "ant-1.7_nodes.csv").exists() else ["ck"]
    pooled = []
    for fs in sets:
        pooled += run(fs)
    base = pd.DataFrame(pooled)
    base.to_csv(ROOT / "results" / "random_baseline.csv", index=False)
    print("\n=== pooled PofB20 vs random ranking ===")
    print(base.to_string(index=False))


if __name__ == "__main__":
    main()
