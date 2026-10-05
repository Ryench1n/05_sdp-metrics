# sdp-metrics

Бакалаврын судалгааны ажил: **Тэнцвэргүй өгөгдөлд согог таамаглах үнэлгээний хэмжүүрүүдийн шинжилгээ**
*(Evaluation Metrics Analysis for Imbalanced Software Defect Prediction)*

Оюутан: Э.Ренчин (21B1NUM2520) · Удирдагч: Э.Билгүүн · МУИС, МКУТ, 2026 намар

## Судалгааны асуултууд

- **RQ1.** ant-1.7 дээр ангиллын хэмжүүрүүд загваруудыг хэр нийцтэй эрэмбэлж байна вэ?
- **RQ2.** Threshold болон ranking-ийн аргын өөрчлөлтөд decision-based ба ranking-based хэмжүүрүүд хэрхэн хариу үйлдэл үзүүлэх вэ?
- **RQ3.** Шалгалтын төсөвтэй нөхцөлд аль хэмжүүрүүд практик үр дүнг зөв тусгаж байна вэ, PofB ба IFA хоорондын trade-off ямар байна вэ?

## Өгөгдөл

- **Үндсэн:** удирдагч багшийн өгсөн өгөгдлийн ant-1.7 хүснэгт (`data/advisor/ant-1.7_nodes.csv`).
  744 класс, 165 defective (22.2%), давхардлыг хассаны дараа 62 metric.
  Graph бүтэц (edges)-ийг ашиглахгүй. LOC багана байхгүй тул PROMISE ant-1.7-оос
  классын бүтэн нэрээр холбосон (744/744 таарсан, `bug` утга бүгд ижил).
  Багшийн өгөгдлийг git-д оруулаагүй — файлыг гараар `data/advisor/` руу хуулна.
- **Харьцуулалт:** PROMISE ant-1.7-ийн 20 CK metric (`data/ant-1.7.csv`).

## Бүтэц

```
data/            ant-1.7.csv (PROMISE), advisor/ (багшийн өгөгдөл, git-д ороогүй)
scripts/         download_data.py — бусад 35 PROMISE санг татах
src/metrics.py   10 хэмжүүр: Accuracy, Precision, Recall, F1, MCC, G-mean,
                 ROC-AUC, PR-AUC, PofB20, IFA
src/data.py      load_nodes() — багшийн өгөгдөл + LOC; load_ck() — PROMISE CK
experiments/     feasibility.py — 2 загвар × threshold × ranking invariance туршилт
tests/           unit, invariance, data loader test
results/         туршилтын гаралт (CSV)
```

## Ажиллуулах

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                # 9 test
python experiments/feasibility.py  # results/feasibility_*.csv үүснэ
```

## Одоогийн үр дүн (feasibility, үндсэн өгөгдөл)

Stratified 10-fold CV, `random_state=42`, threshold 0.5:

| Model | Ranking | F1 | MCC | ROC-AUC | PR-AUC | PofB20 | IFA |
|---|---|---|---|---|---|---|---|
| RF | probability | 0.561 | 0.485 | 0.834 | 0.640 | 0.140 | 0.0 |
| RF | density | 0.561 | 0.485 | 0.525 | 0.232 | 0.320 | 6.9 |
| LR | probability | 0.469 | 0.380 | 0.787 | 0.571 | 0.128 | 0.3 |
| LR | density | 0.469 | 0.380 | 0.383 | 0.183 | 0.272 | 10.7 |

- Threshold-ийг өөрчлөхөд зөвхөн decision-based хэмжүүрүүд өөрчлөгдөнө.
- Ranking-ийн аргыг өөрчлөхөд зөвхөн ranking-based хэмжүүрүүд өөрчлөгдөнө; ROC-AUC буурч PofB20 өснө.
- Энэ зүй тогтол хоёр загвар, хоёр feature set (багшийн 62 metric, PROMISE 20 CK) дөрвүүлэн дээр давтагдсан.

## Хэмжүүрийн тэмдэглэл

- `PofB20` нь LOC-ийг effort-ийн төлөөлөл болгон ашигладаг. LOC нь төгс төлөөлөл биш гэдгийг Shihab et al. (2013), Lavazza et al. (2025) харуулсан.
- `IFA` нь fold бүрт тооцогдоод дунджилна.
- Density ranking: `score = probability / max(LOC, 1)`.
