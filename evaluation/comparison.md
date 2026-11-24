# Model comparison on Dataset 2

All detectors are scored by one script, `evaluation/compare_models.py`, with the same rules. Each model keeps its own hourly scores and alarms, with a threshold set on its own clean calibration data. The script only counts those alarms in the same way for every model.

## How to run

```
python evaluation/compare_models.py
```

It uses the Python standard library only. The XGBoost and ensemble rows read Suhas's runs from `results/xgboost_iforest_ensemble/`, which are on the `xgboost-iforest-ensemble` branch. Until that branch is merged, point the script at a copy with `--ensemble-dir <folder>`. Without the folder, those rows are left out.

It writes `evaluation/results/comparison_gap2_grace3.csv` and `per_attack_gap2_grace3.csv`.

## Shared settings

| Setting | Value |
|---|---|
| Hours scored | 4,154. These are the hours every model has a score for. The first 23 hours of Dataset 2 are dropped, and no attack falls there |
| Labels | `data/processed/dataset2_eval.csv`, from the 7 attack windows (492 attack hours) |
| Grace hours | 3 hours after each attack are left out |
| Episode rule | Alarm runs with 2 quiet hours or fewer between them count as one episode |
| Budget | At most 1 false-alarm episode per week |
| Seeds | Isolation Forest, XGBoost and the ensemble: mean of 3 seeds |
| S score | S = 0.5 × S_TTD + 0.5 × S_CLF, with S_TTD = 1 − mean(delay / attack length), where a missed attack counts as 1, and S_CLF = (recall + TNR) / 2 |

## Results

| Model | Type | Attacks detected (of 7) | Mean delay (h) | Precision | Recall | F1 | ROC-AUC | PR-AUC | False alarms / week | S_TTD | S_CLF | **S** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Transformer AE** | Unsupervised, deep | **7** | **1.7** | **0.906** | **0.630** | **0.743** | **0.919** | **0.792** | 1.15 | **0.979** | **0.811** | **0.895** |
| Ridge | Unsupervised, linear | **7** | 7.7 | 0.899 | 0.598 | 0.718 | 0.865 | 0.735 | 1.34 | 0.885 | 0.794 | 0.840 |
| Ensemble XGB + IF (max) | Supervised + unsupervised | **7** | 20.6 | 0.821 | 0.492 | 0.615 | 0.860 | 0.619 | 0.92 | 0.754 | 0.739 | 0.746 |
| IF + PCA (Suhas) | Unsupervised | 6.0 | 12.1 | 0.813 | 0.295 | 0.433 | 0.674 | 0.430 | 1.26 | 0.749 | 0.643 | 0.696 |
| XGBoost | Supervised | 4.3 | 17.5 | 0.732 | 0.259 | 0.383 | 0.748 | 0.445 | **0.85** | 0.463 | 0.623 | 0.543 |
| Isolation Forest (static) | Unsupervised | 5.3 | 34.9 | 0.293 | 0.028 | 0.051 | 0.645 | 0.208 | 1.40 | 0.404 | 0.509 | 0.457 |

Detection delay per attack, in hours ("miss" means not detected; models with seeds show seed 0):

| Attack | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Transformer AE | 0 | 0 | 0 | 0 | 5 | 2 | 5 |
| Ridge | 0 | 0 | 39 | 0 | 0 | 10 | 5 |
| Ensemble XGB + IF | 0 | 3 | 25 | 10 | 6 | 11 | 45 |
| IF + PCA | 0 | miss | 0 | 20 | 6 | 11 | 45 |
| XGBoost | miss | 3 | 25 | 10 | 8 | miss | miss |
| Isolation Forest (static) | 25 | miss | 26 | 42 | miss | 29 | 45 |

## What this shows

- **The Transformer is best overall.** It catches all 7 attacks, 4 of them in the first hour, and is best on almost every metric.
- **Ridge comes second.** This simple linear model also catches all 7 attacks, but it is slow on attack 3.
- **The ensemble works as an ensemble.** XGBoost and the PCA Isolation Forest miss different attacks (XGBoost misses 1, 6 and 7; IF + PCA misses 2), so taking the max of the two catches all 7. It is also the only model besides XGBoost under the 1-alarm-per-week budget on Dataset 2. Its delays are much longer than the Transformer's, though.
- **PCA matters for the Isolation Forest.** Adding sensor differences and PCA lifts S from 0.46 to 0.70.
- **Of the six detectors, only XGBoost and the ensemble stay within the false-alarm budget on Dataset 2.**

## Caveats

- **XGBoost is supervised.** It is trained on Dataset 2 attacks, and each attack is scored by a model that did not see that attack (leave-one-attack-out). The other detectors never see an attack in training. Its numbers are fair as out-of-fold results, but with only 7 attacks they are noisy.
- **The ensemble keeps Suhas's threshold rule.** Its threshold search merges alarms within 3 quiet hours on calibration, as in `models/XGBoost/xgb_metrics.py`. Only the Dataset 2 scoring uses the shared rules above.
- **These numbers differ from Suhas's own write-up.** He counts only the `ATT_FLAG` hours as attacks and counts false alarms on the validation block. The table above counts the full attack windows and the false alarms on Dataset 2, the same way as for every other model.

## Checks

The script reproduces each model's own reported numbers:

- Transformer: P 0.906, R 0.630, ROC-AUC 0.919, 25 episodes, S 0.895
- Isolation Forest (static): 5.33 attacks, S 0.457
- Ridge: S 0.840, 7 attacks, matching `models/Ridge/evaluation/metrics.json`. F1 differs by 0.002 because 23 fewer hours are scored
- XGBoost, IF + PCA and the ensemble: the attacks detected and the mean delay match Suhas's `all_runs.csv` for every seed. Examples: ensemble seed 0 has 7 attacks and a 14.3 h delay; XGBoost seed 1 has 5 attacks and a 29.6 h delay
