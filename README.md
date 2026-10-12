# sdp-metrics

Бакалаврын судалгааны ажил: **Тэнцвэргүй өгөгдөлд согог таамаглах үнэлгээний хэмжүүрүүдийн шинжилгээ**
*(Evaluation Metrics Analysis for Imbalanced Software Defect Prediction)*

Оюутан: Э.Ренчин (21B1NUM2520) · Удирдагч: Э.Билгүүн · МУИС, МКУТ, 2026 намар

## Судалгааны асуултууд

- **RQ1.** ant-1.7 дээр хэмжүүрүүд загваруудыг хэр нийцтэй эрэмбэлж байна вэ?
- **RQ2.** Threshold болон ranking-ийн аргын өөрчлөлтөд decision-based ба ranking-based хэмжүүрүүд хэрхэн хариу үйлдэл үзүүлэх вэ?
- **RQ3.** Шалгалтын төсөвтэй нөхцөлд аль хэмжүүрүүд практик үр дүнг зөв тусгаж байна вэ, PofB ба IFA хоорондын trade-off ямар байна вэ?

## Өгөгдөл

Удирдагч багшийн өгсөн өгөгдлийн ant-1.7 хүснэгт (`data/advisor/ant-1.7_nodes.csv`):
744 класс, 165 defective (22.2%), давхардлыг хассаны дараа 62 metric.
Graph бүтэц (edges)-ийг ашиглахгүй, гадны эх сурвалжаас багана холбохгүй.
Багшийн өгөгдлийг git-д оруулаагүй — файлыг гараар `data/advisor/` руу хуулна.

**Effort-ийн төлөөлөл:** өгөгдөлд LOC багана байхгүй тул шалгах хүчин чармайлтыг
классын хэмжээ `CSM = NM + NA` (method ба attribute-ын тоо)-аар ойролцоолно.

## Бүтэц

```
data/advisor/    багшийн өгөгдөл (git-д ороогүй)
src/metrics.py   12 хэмжүүр: Accuracy, Precision, Recall, F1, MCC, G-mean,
                 ROC-AUC, PR-AUC, PofB10, PofB20, PofB50, IFA
src/models.py    LR, RF, ET (Extra Trees), XGB (XGBoost)
src/stats.py     Wilcoxon, Holm засвар, Cliff's delta, bootstrap CI
src/data.py      load() — багшийн өгөгдөл, effort = CSM
experiments/     run_cv.py — үндсэн туршилт (10×10 CV, бүртгэлтэй)
                 analyze.py — статистик шинжилгээ
                 feasibility.py — threshold/ranking invariance, baseline-ууд
                 figures.py — тайлангийн зургууд (results/figures/)
tests/           unit, invariance, statistics, data loader test
results/         туршилтын гаралт
```

## Ажиллуулах

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                  # 14 test
python experiments/feasibility.py    # ~10 сек
python experiments/run_cv.py         # ~3 мин; results/cv/
python experiments/analyze.py        # ~3 мин; статистик шинжилгээ
python experiments/figures.py        # тайлангийн зургууд; results/figures/
```

## Үндсэн туршилт

- 10×10 repeated stratified cross-validation (`random_state=42`), загвар бүрийн seed = 42 + repeat.
- 4 загвар (LR, RF, ET, XGB), гиперпараметр тохируулаагүй, ансамбль загварууд 300 tree.
- Threshold 0.5; ranking: probability ба density (`probability / max(CSM, 1)`).
- Fold бүрийн үр дүн, out-of-fold prediction, тохиргоо (`config.json`) автоматаар бүртгэгдэнэ.

Статистик шинжилгээ (`analyze.py`):

- **Wilcoxon signed-rank** (two-sided), repeat түвшний дундаж утгууд дээр (n = 10).
  Нэг repeat доторх fold-ууд хамааралтай тул fold-ийн утгыг шууд ашиглаагүй.
  Олон харьцуулалтад **Holm** засвар.
- **Cliff's delta** effect size (fold түвшний утгууд дээр).
- **Paired bootstrap 95% CI** (B = 2000): дээж бүрт repeat-ийг санамсаргүй сонгож,
  модулиудыг буцаан тавих замаар түүвэрлэнэ. Бүх загвар ижил дээж дээр үнэлэгдэнэ.

## Гол үр дүн (100 fold-ийн дундаж)

| Model | MCC | ROC-AUC | PR-AUC | PofB20 (prob.) | PofB20 (density) |
|---|---|---|---|---|---|
| LR | 0.391 | 0.797 | 0.593 | 0.177 | 0.299 |
| RF | 0.431 | 0.836 | 0.636 | 0.197 | 0.352 |
| ET | 0.403 | 0.831 | 0.620 | 0.206 | 0.341 |
| XGB | 0.408 | 0.799 | 0.596 | 0.216 | 0.307 |

- Хамгийн сайн загвар хэмжүүрээс хамаарна: RF нь Accuracy, MCC, ROC-AUC, PR-AUC-аар,
  XGB нь F1, G-mean, PofB10/20/50-аар тэргүүлнэ. Density ranking-д PofB-ээр RF тэргүүлнэ.
- Density ranking бүх загварт ROC-AUC, PR-AUC-ийг бууруулж, PofB-ийг өсгөж,
  IFA-г нэмэгдүүлсэн (24/24 харьцуулалт Holm-ийн дараа p < 0.05).
- Загвар хоорондын 72 харьцуулалтын 39 нь Wilcoxon-оор ялгаатай боловч bootstrap CI-ээр
  зөвхөн 4 нь (бүгд ROC-AUC) тэгийг агуулаагүй.
