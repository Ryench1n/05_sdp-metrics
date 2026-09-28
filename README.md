# sdp-metrics

Бакалаврын судалгааны ажил: **Тэнцвэргүй өгөгдөлд согог таамаглах үнэлгээний хэмжүүрүүдийн шинжилгээ**
*(Evaluation Metrics Analysis for Imbalanced Software Defect Prediction)*

Оюутан: Э.Ренчин (21B1NUM2520) · Удирдагч: Э.Билгүүн · МУИС, МКУТ, 2026 намар

## Судалгааны асуултууд

- **RQ1.** ant-1.7 дээр ангиллын хэмжүүрүүд загваруудыг хэр нийцтэй эрэмбэлж байна вэ?
- **RQ2.** Threshold болон ranking-ийн аргын өөрчлөлтөд decision-based ба ranking-based хэмжүүрүүд хэрхэн хариу үйлдэл үзүүлэх вэ?
- **RQ3.** Шалгалтын төсөвтэй нөхцөлд аль хэмжүүрүүд практик үр дүнг зөв тусгаж байна вэ, PofB ба IFA хоорондын trade-off ямар байна вэ?

## Бүтэц

```
data/            ant-1.7.csv (үндсэн сан) ба эх сурвалжийн тайлбар
scripts/         download_data.py — бусад 35 PROMISE санг татах
src/metrics.py   10 хэмжүүр: Accuracy, Precision, Recall, F1, MCC, G-mean,
                 ROC-AUC, PR-AUC, PofB20, IFA
src/data.py      өгөгдөл ачаалах
experiments/     feasibility.py — threshold × ranking invariance туршилт
tests/           unit test ба invariance test
results/         туршилтын гаралт (CSV)
```

## Ажиллуулах

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                # 8 test
python experiments/feasibility.py  # results/feasibility.csv үүснэ
```

## Одоогийн үр дүн (feasibility)

Random Forest, stratified 10-fold CV, `random_state=42`:

| Threshold | Ranking | F1 | MCC | ROC-AUC | PR-AUC | PofB20 | IFA |
|---|---|---|---|---|---|---|---|
| 0.5 | probability | 0.552 | 0.464 | 0.843 | 0.634 | 0.114 | 0.4 |
| 0.5 | density | 0.552 | 0.464 | 0.535 | 0.258 | 0.282 | 5.2 |
| 0.3 | probability | 0.597 | 0.475 | 0.843 | 0.634 | 0.114 | 0.4 |
| 0.3 | density | 0.597 | 0.475 | 0.535 | 0.258 | 0.282 | 5.2 |

- Threshold-ийг өөрчлөхөд зөвхөн decision-based хэмжүүрүүд өөрчлөгдөнө.
- Ranking-ийн аргыг өөрчлөхөд зөвхөн ranking-based хэмжүүрүүд өөрчлөгдөнө; ROC-AUC буурч PofB20 өснө.
- Probability ranking-ийн PofB20 (pooled 0.127) нь санамсаргүй ranking-ээс (0.199 ± 0.026) муу.

## Хэмжүүрийн тэмдэглэл

- `PofB20` нь LOC-ийг effort-ийн төлөөлөл болгон ашигладаг. LOC нь төгс төлөөлөл биш гэдгийг Shihab et al. (2013), Lavazza et al. (2025) харуулсан.
- `IFA` нь fold бүрт тооцогдоод дунджилна.
- Density ranking: `score = probability / max(LOC, 1)`.
