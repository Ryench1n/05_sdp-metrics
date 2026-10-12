# sdp-metrics

Бакалаврын судалгааны ажил: **Тэнцвэргүй өгөгдөлд согог таамаглах үнэлгээний хэмжүүрүүдийн шинжилгээ**
*(Evaluation Metrics Analysis for Software Defect Prediction on Imbalanced Data)*

Оюутан: Э.Ренчин (21B1NUM2520) · Удирдагч: Э.Билгүүн · МУИС, МКУТ, 2026 намар

## Судалгааны асуултууд

- **RQ1.** Хэмжүүрүүд загваруудыг хэр нийцтэй эрэмбэлж байна вэ?
- **RQ2.** Threshold болон ranking-ийн аргын өөрчлөлтөд decision-based ба ranking-based хэмжүүрүүд хэрхэн хариу үйлдэл үзүүлэх вэ?
- **RQ3.** Шалгалтын төсөвтэй нөхцөлд аль хэмжүүрүүд практик үр дүнг зөв тусгаж байна вэ, PofB ба IFA хоорондын trade-off ямар байна вэ?

## Өгөгдөл (PROMISE, 5 сан)

PROMISE репозиторийн (Jureczko & Madeyski, 2010) таван Java төсөл: ant-1.7, camel-1.6, jedit-4.0,
log4j-1.0, xalan-2.6 (нийт 3 036 модуль, defective 19.5–46.4%). Сан бүр 20 CK metric агуулна.
Effort-ийн төлөөлөл нь `loc` (кодын мөрийн тоо). Шошго: `bug > 0 → 1`.
Татах: `python scripts/download_data.py` (klainfo/DefectData).

## Бүтэц

```
data/promise/    5 PROMISE сан
scripts/         download_data.py
src/metrics.py   12 хэмжүүр: Accuracy, Precision, Recall, F1, MCC, G-mean,
                 ROC-AUC, PR-AUC, PofB10, PofB20, PofB50, IFA
src/models.py    LR, RF, ET (Extra Trees), XGB (XGBoost)
src/stats.py     Wilcoxon, Holm засвар, Cliff's delta, bootstrap CI
src/data.py      load(name) — PROMISE CK, effort = loc
experiments/     run_cv.py --dataset <name> — 10×10 CV (бүртгэлтэй)
                 analyze.py --dataset <name> — сан бүрийн статистик шинжилгээ
                 pooled.py — 5 сангийн нэгтгэсэн шинжилгээ
                 feasibility.py — threshold/ranking invariance, baseline-ууд
                 figures.py — тайлангийн зургууд
tests/           unit, invariance, statistics, data loader test
results/         cv/<dataset>/, pooled/, feasibility.csv, baselines.csv, random_baseline.csv
```

## Ажиллуулах

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                  # 14 test
python experiments/feasibility.py    # ~1 мин
for d in ant-1.7 camel-1.6 jedit-4.0 log4j-1.0 xalan-2.6; do
  python experiments/run_cv.py --dataset $d    # ~4 мин
  python experiments/analyze.py --dataset $d   # ~3 мин
done
python experiments/pooled.py
python experiments/figures.py
```

## Үндсэн туршилт

- 10×10 repeated stratified cross-validation (`random_state=42`), загварын seed = 42 + repeat.
- 4 загвар (LR, RF, ET, XGB), hyperparameter тохируулаагүй, ансамбль загварууд 300 tree.
- Threshold 0.5; ranking: probability ба density (`probability / max(loc, 1)`).
- Wilcoxon signed-rank (repeat-ийн дундаж, n = 10; нэгтгэсэн шалгуурт n = 50), Holm засвар,
  Cliff's delta, paired bootstrap 95% CI (B = 2000), Kendall's W.

## Гол үр дүн (5 сангийн дундаж)

| Model | ROC-AUC (P → D) | PofB20 (P → D) | IFA (P → D) | MCC |
|---|---|---|---|---|
| LR | 0.780 → 0.387 | 0.112 → 0.409 | 0.42 → 7.74 | 0.388 (өөрчлөгдөөгүй) |
| RF | 0.809 → 0.519 | 0.143 → 0.429 | 0.35 → 3.91 | 0.433 (өөрчлөгдөөгүй) |
| ET | 0.813 → 0.569 | 0.149 → 0.429 | 0.39 → 3.16 | 0.430 (өөрчлөгдөөгүй) |
| XGB | 0.773 → 0.640 | 0.157 → 0.363 | 0.53 → 2.46 | 0.385 (өөрчлөгдөөгүй) |

- Density ranking: 120/120 харьцуулалтад ROC-AUC, PR-AUC буурч, PofB өсөж, IFA муудсан (Holm-ийн дараа p < 0.05).
- Сан бүрт хэмжүүрээс хамааран 2–4 өөр загвар шилдэг гарсан (Kendall's W 0.28–0.57).
- Загвар хоорондын 360 харьцуулалтын 209 нь Wilcoxon-оор, 63 нь bootstrap CI-ээр ялгаатай.
