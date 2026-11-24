# XGBoost + Isolation Forest ensemble

Two detectors with different failure modes, fused per hour: the supervised XGBoost (`models/XGBoost`, features and settings from Nemade et al. 2026) and an unsupervised Isolation Forest. The Transformer is not part of this ensemble.

```bash
python ../XGBoost/train_xgboost.py --dataset-path ../../data/processed --out-dir runs/xgb
# Isolation Forest = the team's (scores read straight from the zip; its .joblib models are never loaded):
python ensemble.py --xgb-dir runs/xgb \
    --teammate-iforest-zip ../Isolation_Forest/iforest_ensemble_outputs.zip --out-dir runs/ensemble
# Optional comparison: this folder's own Isolation Forest + PCA (the paper's recipe)
python isolation_forest.py --dataset-path ../../data/processed --out-dir runs/iforest
python ensemble.py --xgb-dir runs/xgb --iforest-dir runs/iforest --out-dir runs/ensemble_pca
```
The whole run is also in `Ensemble_BATADAL.ipynb` (reads the XGBoost notebook's scores and the team's zip, refits the PCA Isolation Forest, and saves tables and plots to `notebook_outputs/`).

(`export_transformer_scores.py` and the `--transformer-scores` option remain for a later three-way ensemble.)

## Rules that keep Dataset 2 out of every choice
- Isolation Forest fitted on the Dataset 1 train block only.
- Scores are put on a common scale with the calibration block only: `percentile` (share of calibration scores it is >=) or `ratio` (log(1 + score / calibration 99th percentile)).
- Fusion is a fixed rule, not fitted weights: `mean`, `paper` (0.4 XGBoost / 0.1 Isolation Forest, renormalised) or `max`. `max` of calibration percentiles is the fusion the project proposal already specifies.
- Alarm threshold: the lowest one inside the false-alarm budget (<= 1 episode/week, <= 1% of hours) on the calibration block.
- Evaluation as in `models/XGBoost`: positives = `ATT_FLAG = 1` hours, `-999` dropped, false alarms on the validation block (alarm hours per week), XGBoost scored out-of-fold, episodes merged at <= 3 quiet hours for every detector. Seeds are paired (XGBoost seed k with Isolation Forest seed k); scores are matched by `DATETIME`.

## Results (3 seeds, mean; percentile scale)

No precision or F1 (see `models/XGBoost/xgboost_approach.md`): validation alarm hours per week stand in for false alarms. "Outside windows" is the era check, the share of Dataset 2 hours outside every attack window that alarm (budget 1%). The S-score is approximate.

Team Isolation Forest (36 raw sensors, 200 trees, no PCA):

| Score | Attacks detected (of 7) | Recall | Val. alarm h/week | Outside windows | ROC-AUC | PR-AUC | Approx. S |
|---|---|---|---|---|---|---|---|
| XGBoost | 4.3 | 0.25 | 0.9 | 1.5% | 0.69 | 0.48 | 0.54 |
| Isolation Forest | 5.3 | 0.05 | 1.1 | 0.9% | 0.73 | 0.29 | 0.46 |
| mean of percentiles | 4.0 | 0.05 | 0.8 | 0.8% | 0.75 | 0.29 | 0.45 |
| paper weights | 4.3 | 0.09 | 0.8 | 0.9% | 0.72 | 0.34 | 0.50 |
| **max of percentiles** | **6.3** | **0.26** | **1.2** | **1.6%** | **0.88** | **0.58** | **0.63** |

Comparison: our Isolation Forest + PCA (raw sensors + 1-step differences, 20 components, 500 trees):

| Score | Attacks detected | Recall | Val. alarm h/week | Outside windows | ROC-AUC | PR-AUC | Approx. S |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 6.0 | 0.48 | 1.2 | 1.0% | 0.80 | 0.62 | 0.74 |
| max of percentiles | 7.0 | 0.69 | 0.8 | 1.7% | 0.95 | 0.87 | 0.80 |

## Reading it
- Max fusion beats both detectors alone in either case: the two catch different hours (XGBoost mostly attack 4), so alarming when either is extreme helps, and averaging dilutes a one-detector spike below the threshold.
- The team Isolation Forest is a static baseline on raw sensors and is weak on this protocol (recall 0.05). Adding the paper's two steps (differences and PCA) is what lifts it; the paper reports the same (F1 0.245 without PCA, 0.565 with). If the ensemble is to be strong, use the PCA version or add those steps to the team notebook.
- Six fusion variants were run per Isolation Forest; the best is the proposal's own rule, but treat the figures as optimistic until confirmed on the Test set. Only 7 attacks exist, and precision/F1 are left to the Test set because they depend on the attack-to-clean mix.
- Era check (Dataset 1 is 2014, Dataset 2 is 2016): our Isolation Forest's AUC is 0.80 against both 2014 and 2016 clean hours. The ensemble's `max` fusion alarms on 1.6-1.7% of the Dataset 2 hours outside every attack window against a 1% budget, so a small part of its alarms may reflect the year rather than attacks; the individual detectors sit at 0.9-1.5%.
- The percentile scale tops out at 1, so the calibrated thresholds sit just below 1 and the attack-window plots look flat near the top; the `ratio` scale has no ceiling but gave weaker alarms at the same budget.
