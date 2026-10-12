# Results

- `feasibility.csv` — сан бүрт threshold (0.5/0.3) ба ranking-ийн invariance туршилт, RF ба LR, 10-fold
- `baselines.csv` — dummy classifier ба size-only (LOC) baseline
- `random_baseline.csv` — pooled PofB20 ба санамсаргүй ranking (2000 permutation)
- `cv/<dataset>/` — 10×10 туршилт: `folds.csv`, `oof.csv.gz`, `config.json`, `summary.csv`,
  `wilcoxon_models.csv`, `wilcoxon_ranking.csv`, `model_ranks*.csv`, `repeat_winners.csv`,
  `prob_effort_correlation.csv`, `bootstrap_ci.csv`, `bootstrap_diff.csv`
- `pooled/` — 5 сангийн нэгтгэл: `summary_long.csv`, `best_models.csv`, `ranking_effect.csv`,
  `global_ranking.csv`, `significance.csv`, `rank_agreement.csv`
