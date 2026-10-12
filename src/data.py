"""Өгөгдөл ачаалах.

Удирдагч багшийн өгсөн өгөгдлийн хүснэгтэн хэсгийг (data/advisor/<name>_nodes.csv)
ашиглана. Graph бүтэц (edges)-ийг ашиглахгүй. Гадны өөр эх сурвалжаас багана
холбохгүй.

Effort-ийн төлөөлөл: CSM (Class Size Metric = NM + NA, method болон attribute-ын
тоо). Өгөгдөлд LOC багана байхгүй тул шалгах хүчин чармайлтыг классын хэмжээгээр
ойролцоолно.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ADVISOR_DIR = DATA_DIR / "advisor"
EFFORT = "CSM"
TARGETS = {"bug", "label"}


def load(name="ant-1.7"):
    """Return (X, y, effort, features, frame).

    - Тодорхойлогч болон категори багануудыг (ID, Name, Type, Package,
      LongName, isAbstract, Visibility) feature-д оруулахгүй.
    - `bug` нь шошгоны эх сурвалж тул feature-д оруулахгүй (leakage).
    - Утга нь бүхэлдээ давхардсан багануудыг хасна.
    """
    df = pd.read_csv(ADVISOR_DIR / f"{name}_nodes.csv")
    numeric = [c for c in df.select_dtypes("number").columns if c not in TARGETS]
    duplicated = df[numeric].T.duplicated()
    features = [c for c, dup in zip(numeric, duplicated) if not dup]

    X = df[features].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    return X, y, df[EFFORT].to_numpy(dtype=float), features, df
