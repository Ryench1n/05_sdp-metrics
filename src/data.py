"""Өгөгдөл ачаалах.

PROMISE (Jureczko & Madeyski) репозиторийн таван Java төслийн CK metric-ийн
өгөгдлийг (data/promise/<name>.csv) ашиглана. Сан бүр 20 metric агуулах бөгөөд
тэдгээрийн нэг нь кодын мөрийн тоо `loc`. `loc`-ийг effort-ийн төлөөлөл болгон
ашиглахаас гадна feature-д мөн оруулна (PROMISE-ийн стандарт 20 metric).

Хоёртын шошго: bug > 0 → 1.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "promise"
DATASETS = ["ant-1.7", "camel-1.6", "jedit-4.0", "log4j-1.0", "xalan-2.6"]
META = {"name", "version", "name.1", "bug"}
EFFORT = "loc"


def load(name="ant-1.7"):
    """Return (X, y, effort, features, frame)."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    features = [c for c in df.columns if c not in META]
    X = df[features].to_numpy(dtype=float)
    y = (df["bug"] > 0).astype(int).to_numpy()
    return X, y, df[EFFORT].to_numpy(dtype=float), features, df
