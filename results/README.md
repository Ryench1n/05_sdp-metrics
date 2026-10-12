# Results

- `feasibility.csv` — threshold (0.5/0.3) ба ranking (probability/density)-ийн invariance туршилт, RF ба LR, 10-fold
- `baselines.csv` — dummy classifier ба size-only (CSM) baseline
- `random_baseline.csv` — pooled PofB20 ба санамсаргүй ranking-ийн тархалт (2000 permutation)
- `cv/` — үндсэн 10×10 туршилт: `folds.csv`, `oof.csv.gz`, `config.json`, `summary.csv`,
  `wilcoxon_models.csv`, `wilcoxon_ranking.csv`, `model_ranks*.csv`, `repeat_winners.csv`,
  `bootstrap_ci.csv`, `bootstrap_diff.csv`
