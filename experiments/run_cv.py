"""Үндсэн туршилт: 10×10 repeated stratified cross-validation.

Загвар бүрийг fold бүр дээр сургаж, хоёр ranking-ийн аргаар (probability,
density) бүх хэмжүүрийг тооцоод, туршилтын тохиргоо, fold бүрийн үр дүн,
out-of-fold prediction-ийг автоматаар бүртгэнэ.

Ажиллуулах:
    python experiments/run_cv.py --dataset ant-1.7

Гаралт: results/cv/<dataset>/
    config.json   — тохиргоо, өгөгдлийн мэдээлэл, сангийн хувилбарууд, commit
    folds.csv     — fold бүрийн хэмжүүрүүд (repeat, fold, model, ranking)
    oof.csv.gz    — repeat бүрийн out-of-fold probability
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import DATASETS, EFFORT, load    # noqa: E402
from metrics import evaluate               # noqa: E402
from models import MODEL_NAMES, make_model  # noqa: E402

THRESHOLD = 0.5


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ant-1.7", choices=DATASETS)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y, effort, features, frame = load(args.dataset)
    out = ROOT / "results" / "cv" / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "data": "PROMISE CK (Jureczko & Madeyski)", "effort": EFFORT,
        "n_modules": int(len(y)), "n_defective": int(y.sum()), "n_features": len(features),
        "repeats": args.repeats, "folds": args.folds, "seed": args.seed,
        "split": "RepeatedStratifiedKFold(random_state=seed)",
        "model_seed": "seed + repeat",
        "models": MODEL_NAMES, "threshold": THRESHOLD,
        "rankings": {"probability": "p", "density": f"p / max({EFFORT}, 1)"},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        "git_commit": git_commit(),
    }

    rskf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats,
                                   random_state=args.seed)
    rows, oof = [], []
    probs = {m: np.full((args.repeats, len(y)), np.nan) for m in MODEL_NAMES}
    t0 = time.time()
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        r, f = divmod(i, args.folds)
        for m in MODEL_NAMES:
            model = make_model(m, seed=args.seed + r).fit(X[tr], y[tr])
            p = model.predict_proba(X[te])[:, 1]
            probs[m][r, te] = p
            pred = (p >= THRESHOLD).astype(int)
            for ranking, score in (("probability", p), ("density", p / np.maximum(effort[te], 1))):
                met = evaluate(y[te], pred, score, effort[te])
                rows.append({"repeat": r, "fold": f, "model": m, "ranking": ranking,
                             "n_test": len(te), "k_test": int(y[te].sum()), **met})
        if f == args.folds - 1:
            print(f"repeat {r + 1}/{args.repeats} done ({time.time() - t0:.0f}s)", flush=True)

    for m in MODEL_NAMES:
        for r in range(args.repeats):
            oof.append(pd.DataFrame({"repeat": r, "module": np.arange(len(y)), "model": m,
                                     "y": y, "effort": effort, "prob": probs[m][r]}))

    pd.DataFrame(rows).to_csv(out / "folds.csv", index=False)
    pd.concat(oof).to_csv(out / "oof.csv.gz", index=False)
    config["finished"] = datetime.now().isoformat(timespec="seconds")
    config["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"saved to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
