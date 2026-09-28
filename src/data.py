"""PROMISE (Jureczko & Madeyski) өгөгдлийн санг ачаалах."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
META_COLS = {"name", "version", "name.1", "bug"}


def load(name="ant-1.7"):
    """Return (X, y, loc, feature_names, frame).

    y  — defect тоо > 0 бол 1 (хоёртын шошго)
    loc — модулийн lines of code (effort-ийн төлөөлөл)
    """
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    features = [c for c in df.columns if c not in META_COLS]
    X = df[features].to_numpy(dtype=float)
    y = (df["bug"] > 0).astype(int).to_numpy()
    loc = df["loc"].to_numpy(dtype=float)
    return X, y, loc, features, df
