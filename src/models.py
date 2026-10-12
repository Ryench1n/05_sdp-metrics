"""Судалгаанд ашиглах ангилагчид.

Гиперпараметрийг тохируулаагүй: судалгааны объект нь загвар биш, хэмжүүр.
Ансамбль загваруудын модны тоог ижилхэн (300) тогтоосон.
"""
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

N_TREES = 300


def make_model(name, seed):
    """Нэрээр нь шинэ загвар үүсгэнэ. Scaling нь pipeline дотор тул data leakage үүсэхгүй."""
    if name == "LR":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, random_state=seed))
    if name == "RF":
        return RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
    if name == "XGB":
        return XGBClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1,
                             eval_metric="logloss", verbosity=0)
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["LR", "RF", "ET", "XGB"]
