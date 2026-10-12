"""Боломжийн шалгалт ба invariance туршилт.

1. Threshold (0.5, 0.3) болон ranking-ийн арга (probability, density)-ийг
   өөрчлөхөд хэмжүүрүүд хэрхэн хариу үйлдэл үзүүлэхийг RF, LR загвараар шалгана.
2. Machine learning ашиглахгүй хоёр baseline-тай харьцуулна:
   dummy classifier (бүгдийг clean гэж таамаглана) ба size-only baseline
   (модулийг зөвхөн CSM-ээр эрэмбэлнэ).
3. Санамсаргүй ranking-ийн PofB20 тархалтыг (2000 permutation) гаргана.

Ажиллуулах:  python experiments/feasibility.py
Гаралт:      results/feasibility.csv, results/baselines.csv, results/random_baseline.csv
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
from data import load                   # noqa: E402
from metrics import RANKING_METRICS, evaluate, pofb  # noqa: E402

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


def fold_mean(y, folds, pred, score, effort):
    rows = [evaluate(y[te], pred[te], score[te], effort[te]) for _, te in folds]
    return pd.DataFrame(rows).mean().round(4).to_dict()


def main():
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    X, y, effort, features, _ = load("ant-1.7")
    folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X, y))

    rows, pooled = [], []
    for name, make in MODELS.items():
        prob = np.zeros(len(y))
        for tr, te in folds:
            prob[te] = make().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        density = prob / np.maximum(effort, 1)
        pooled.append({"model": name,
                       "pooled_PofB20_probability": pofb(y, prob, effort, 0.2),
                       "pooled_PofB20_density": pofb(y, density, effort, 0.2)})
        for t in THRESHOLDS:
            for r in RANKINGS:
                score = prob if r == "probability" else density
                rows.append({"model": name, "threshold": t, "ranking": r,
                             **fold_mean(y, folds, (prob >= t).astype(int), score, effort)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "feasibility.csv", index=False)

    # Baseline-ууд: dummy (бүгд clean) ба size-only (CSM-ээр эрэмбэлж, медианаас дээшийг defective)
    size_pred = (effort > np.median(effort)).astype(int)
    dummy = fold_mean(y, folds, np.zeros(len(y), int), np.zeros(len(y)), effort)
    # Dummy бүх модульд ижил score өгдөг тул ranking тодорхойгүй: ranking хэмжүүрийг тооцохгүй
    dummy.update({m: np.nan for m in RANKING_METRICS})
    base = pd.DataFrame([
        {"baseline": "dummy (all clean)", **dummy},
        {"baseline": "size-only (CSM)",
         **fold_mean(y, folds, size_pred, effort, effort)},
    ])
    base.to_csv(out / "baselines.csv", index=False)

    rng = np.random.default_rng(0)
    rand = np.array([pofb(y, rng.random(len(y)), effort, 0.2) for _ in range(N_PERMUTATIONS)])
    for row in pooled:
        row["random_PofB20_mean"] = round(rand.mean(), 4)
        row["random_PofB20_sd"] = round(rand.std(), 4)
        # харьцуулалтыг бөөрөнхийлөөгүй утгаар хийнэ
        row["share_random_le_probability"] = round((rand <= row["pooled_PofB20_probability"]).mean(), 4)
        row["share_random_ge_density"] = round((rand >= row["pooled_PofB20_density"]).mean(), 4)
        row["pooled_PofB20_probability"] = round(row["pooled_PofB20_probability"], 4)
        row["pooled_PofB20_density"] = round(row["pooled_PofB20_density"], 4)
    pd.DataFrame(pooled).to_csv(out / "random_baseline.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"=== {len(features)} features, N={len(y)}, K={y.sum()} ===")
    print(res.to_string(index=False))
    print("\n=== baselines ===")
    print(base.to_string(index=False))
    print("\n=== pooled PofB20 vs random ranking ===")
    print(pd.DataFrame(pooled).to_string(index=False))


if __name__ == "__main__":
    main()
