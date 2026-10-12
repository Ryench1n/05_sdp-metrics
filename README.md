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
src/metrics.py   12 хэмжүүр: Accuracy, Precision, Recall, F1, MCC, G-mean,
                 ROC-AUC, PR-AUC, PofB10, PofB20, PofB50, IFA
src/models.py    LR, RF, ET (Extra Trees), XGB (XGBoost)
src/stats.py     Wilcoxon, Holm засвар, Cliff's delta, bootstrap CI
src/data.py      load_nodes() — багшийн өгөгдөл + LOC; load_ck() — PROMISE CK
experiments/     run_cv.py — үндсэн туршилт (10×10 CV, бүртгэлтэй)
                 analyze.py — статистик шинжилгээ
                 feasibility.py — урьдчилсан invariance туршилт (I явц)
tests/           unit, invariance, statistics, data loader test
results/         туршилтын гаралт
```

## Ажиллуулах

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                          # 14 test
python experiments/run_cv.py                 # ~3 мин; results/cv_nodes/
python experiments/analyze.py                # ~3 мин; статистик шинжилгээ
python experiments/run_cv.py --feature-set ck && python experiments/analyze.py --feature-set ck
```

## Үндсэн туршилт

- 10×10 repeated stratified cross-validation (`random_state=42`), загвар бүрийн seed = 42 + repeat.
- 4 загвар (LR, RF, ET, XGB), гиперпараметр тохируулаагүй, ансамбль загварууд 300 tree.
- Threshold 0.5; ranking: probability ба density (probability / LOC).
- Fold бүрийн үр дүн, out-of-fold prediction, тохиргоо (`config.json`) автоматаар бүртгэгдэнэ.

Статистик шинжилгээ (`analyze.py`):

- **Wilcoxon signed-rank** (two-sided), repeat түвшний дундаж утгууд дээр (n = 10).
  Нэг repeat доторх fold-ууд хамааралтай тул fold-ийн утгыг шууд ашиглаагүй.
  Олон харьцуулалтад **Holm** засвар.
- **Cliff's delta** effect size (fold түвшний утгууд дээр).
- **Paired bootstrap 95% CI** (B = 2000): дээж бүрт repeat-ийг санамсаргүй сонгож,
  модулиудыг буцаан тавих замаар түүвэрлэнэ. Бүх загвар ижил дээж дээр үнэлэгдэнэ.

## Гол үр дүн (багшийн өгөгдөл, 100 fold-ийн дундаж)

| Model | MCC | ROC-AUC | PR-AUC | PofB20 (prob.) | PofB20 (density) |
|---|---|---|---|---|---|
| LR | 0.391 | 0.797 | 0.593 | 0.142 | 0.274 |
| RF | 0.431 | 0.836 | 0.636 | 0.151 | 0.329 |
| ET | 0.403 | 0.831 | 0.620 | 0.158 | 0.337 |
| XGB | 0.408 | 0.799 | 0.596 | 0.179 | 0.302 |

- Хамгийн сайн загвар хэмжүүрээс хамаарна: RF нь Accuracy, MCC, ROC-AUC, PR-AUC-аар,
  XGB нь F1, G-mean, PofB10/20/50-аар тэргүүлнэ.
- Density ranking бүх загварт ROC-AUC, PR-AUC-ийг бууруулж, PofB-ийг өсгөж,
  IFA-г нэмэгдүүлсэн (24/24 харьцуулалт Holm-ийн дараа p < 0.05, Cliff's delta их).

## Хэмжүүрийн тэмдэглэл

- `PofB20` нь LOC-ийг effort-ийн төлөөлөл болгон ашигладаг. LOC нь төгс төлөөлөл биш гэдгийг Shihab et al. (2013), Lavazza et al. (2025) харуулсан.
- `IFA` нь fold бүрт тооцогдоод дунджилна.
- Density ranking: `score = probability / max(LOC, 1)`.
