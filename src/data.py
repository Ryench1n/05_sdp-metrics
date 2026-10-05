"""Өгөгдөл ачаалах.

Хоёр feature set бий:
  * "nodes" (үндсэн) — удирдагч багшийн өгсөн өгөгдлийн хүснэгтэн хэсэг
    (data/advisor/<name>_nodes.csv). Graph бүтэц (edges)-ийг ашиглахгүй.
    LOC багана байхгүй тул PROMISE-ийн анхны сангаас классын бүтэн
    нэрээр (LongName) холбож авна.
  * "ck" — PROMISE (Jureczko & Madeyski) сангийн 20 metric. Харьцуулалтад
    ашиглана.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ADVISOR_DIR = DATA_DIR / "advisor"
CK_META = {"name", "version", "name.1", "bug"}
NODES_TARGETS = {"bug", "label", "loc"}


def _promise(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df.rename(columns={df.columns[2]: "LongName"})  # 3 дахь багана = классын бүтэн нэр


def load_ck(name="ant-1.7"):
    """PROMISE CK feature set. Return (X, y, loc, features, frame)."""
    df = _promise(name)
    features = [c for c in df.columns if c not in CK_META | {"LongName"}]
    X = df[features].to_numpy(dtype=float)
    y = (df["bug"] > 0).astype(int).to_numpy()
    return X, y, df["loc"].to_numpy(dtype=float), features, df


def load_nodes(name="ant-1.7"):
    """Багшийн өгөгдөл + PROMISE LOC. Return (X, y, loc, features, frame).

    - Тодорхойлогч болон категори багануудыг (ID, Name, Type, Package,
      LongName, isAbstract, Visibility) feature-д оруулахгүй.
    - `bug` нь шошгоны эх сурвалж тул feature-д оруулахгүй (leakage).
    - Утга нь бүхэлдээ давхардсан багануудыг хасна.
    """
    nodes = pd.read_csv(ADVISOR_DIR / f"{name}_nodes.csv")
    loc = _promise(name)[["LongName", "loc"]]
    df = nodes.merge(loc, on="LongName", how="left", validate="one_to_one")
    if df["loc"].isna().any():
        missing = df.loc[df["loc"].isna(), "LongName"].tolist()
        raise ValueError(f"LOC олдоогүй класс: {missing[:5]}")

    numeric = [c for c in df.select_dtypes("number").columns if c not in NODES_TARGETS]
    duplicated = df[numeric].T.duplicated()
    features = [c for c, dup in zip(numeric, duplicated) if not dup]

    X = df[features].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    return X, y, df["loc"].to_numpy(dtype=float), features, df


def load(name="ant-1.7", feature_set="nodes"):
    return load_nodes(name) if feature_set == "nodes" else load_ck(name)
