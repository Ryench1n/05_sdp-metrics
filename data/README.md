# Өгөгдөл

`ant-1.7.csv` — судалгааны үндсэн өгөгдлийн сан.

- Эх сурвалж: Jureczko, M., & Madeyski, L. (2010). *Towards identifying software project clusters with regard to defect prediction*. PROMISE 2010. doi:10.1145/1868328.1868342
- Татсан газар: [klainfo/DefectData](https://github.com/klainfo/DefectData) (`inst/extdata/terapromise/ck/`)
- 745 класс, 20 metric, `bug` багана нь defect-ийн тоо
- Хоёртын шошго: `bug > 0 → 1` (166 defective, 22.3%)

Бусад 35 сангийн үзүүлэлт `results/dataset_overview.csv`-д байна. Тэдгээрийг татах:

```
python scripts/download_data.py
```
