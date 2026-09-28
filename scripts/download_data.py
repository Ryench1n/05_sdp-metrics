"""PROMISE (Jureczko & Madeyski) CK өгөгдлийн сангуудыг data/ хавтас руу татна.

Эх сурвалж: https://github.com/klainfo/DefectData (inst/extdata/terapromise/ck)
Ажиллуулах:  python scripts/download_data.py
"""
from pathlib import Path
from urllib.request import urlopen

BASE = ("https://raw.githubusercontent.com/klainfo/DefectData/master/"
        "inst/extdata/terapromise/ck/")
DATASETS = [
    "ant-1.3", "ant-1.4", "ant-1.5", "ant-1.6", "ant-1.7",
    "camel-1.0", "camel-1.2", "camel-1.4", "camel-1.6",
    "ivy-1.4", "ivy-2.0",
    "jedit-4.0", "jedit-4.1", "jedit-4.2", "jedit-4.3",
    "log4j-1.0", "log4j-1.1", "log4j-1.2",
    "lucene-2.0", "lucene-2.2", "lucene-2.4",
    "poi-2.0", "poi-2.5", "poi-3.0",
    "synapse-1.0", "synapse-1.1", "synapse-1.2",
    "velocity-1.4", "velocity-1.6",
    "xalan-2.4", "xalan-2.5", "xalan-2.6", "xalan-2.7",
    "xerces-1.2", "xerces-1.3", "xerces-1.4",
]

out = Path(__file__).resolve().parent.parent / "data"
out.mkdir(exist_ok=True)
for name in DATASETS:
    target = out / f"{name}.csv"
    if target.exists():
        continue
    with urlopen(BASE + f"{name}.csv") as r:
        target.write_bytes(r.read())
    print("downloaded", target.name)
print("done:", len(list(out.glob("*.csv"))), "files in", out)
