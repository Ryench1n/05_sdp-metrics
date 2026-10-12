"""PROMISE (Jureczko & Madeyski) CK өгөгдлийн сангуудыг data/promise/ руу татна.

Эх сурвалж: https://github.com/klainfo/DefectData (inst/extdata/terapromise/ck)
Ажиллуулах:  python scripts/download_data.py
"""
from pathlib import Path
from urllib.request import urlopen

BASE = ("https://raw.githubusercontent.com/klainfo/DefectData/master/"
        "inst/extdata/terapromise/ck/")
DATASETS = ["ant-1.7", "camel-1.6", "jedit-4.0", "log4j-1.0", "xalan-2.6"]

out = Path(__file__).resolve().parent.parent / "data" / "promise"
out.mkdir(parents=True, exist_ok=True)
for name in DATASETS:
    target = out / f"{name}.csv"
    if target.exists():
        continue
    with urlopen(BASE + f"{name}.csv") as r:
        target.write_bytes(r.read())
    print("downloaded", target.name)
print("done:", sorted(p.name for p in out.glob("*.csv")))
