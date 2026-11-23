# Model comparison on Dataset 2

All detectors are scored by one script, `evaluation/compare_models.py`, with the same rules. Each model keeps its own hourly scores and alarms, with a threshold set on its own clean calibration data. The script only counts those alarms in the same way for every model.

## How to run

```
python evaluation/compare_models.py
```

It uses the Python standard library only. It writes `evaluation/results/comparison_gap2_grace3.csv` and `per_attack_gap2_grace3.csv`.

## Shared settings

| Setting | Value |
|---|---|
| Hours scored | 4,166. These are the hours every model has a score for. The Transformer has no score for the first 11 hours of Dataset 2, and no attack falls there |
| Labels | `data/processed/dataset2_eval.csv`, from the 7 attack windows |
| Grace hours | 3 hours after each attack are left out |
| Episode rule | Alarm runs with 2 quiet hours or fewer between them count as one episode |
| Budget | At most 1 false-alarm episode per week |
| Isolation Forest | Mean of 3 seeds |
| S score | S = 0.5 × S_TTD + 0.5 × S_CLF, with S_TTD = 1 − mean(delay / attack length), where a missed attack counts as 1, and S_CLF = (recall + TNR) / 2 |

## Results

| Model | Attacks detected (of 7) | Mean delay (h) | Precision | Recall | F1 | ROC-AUC | PR-AUC | False alarms / week | S_TTD | S_CLF | **S** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Transformer AE | **7** | **1.7** | **0.906** | **0.630** | **0.743** | **0.919** | **0.791** | **1.15** | **0.979** | **0.811** | **0.895** |
| Ridge | **7** | 8.3 | 0.904 | 0.610 | 0.728 | 0.866 | 0.745 | 1.33 | 0.876 | 0.800 | 0.838 |
| Isolation Forest | 5.3 | 34.9 | 0.293 | 0.028 | 0.051 | 0.645 | 0.207 | 1.40 | 0.404 | 0.509 | 0.457 |

Detection delay per attack, in hours ("miss" means not detected; Isolation Forest is seed 0):

| Attack | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Transformer AE | 0 | 0 | 0 | 0 | 5 | 2 | 5 |
| Ridge | 0 | 0 | 43 | 0 | 0 | 10 | 5 |
| Isolation Forest | 25 | miss | 26 | 42 | miss | 29 | 45 |

## What this shows

- **The Transformer is best on every metric.** It catches all 7 attacks, 4 of them in the first hour, and has the fewest false alarms.
- **Ridge is close on precision and recall.** It also catches all 7 attacks, but it is much slower on attack 3 (43 h) and attack 6 (10 h). This is why its S_TTD is lower.
- **The static Isolation Forest does poorly.** It looks at one hour at a time and does not model how sensors relate to each other, so it flags very few attack hours (recall 0.03) and misses attacks 2 and 5.
- **No model stays inside the false-alarm budget on Dataset 2.** All three are inside it, or close, on their calibration data, but they are over 1 episode per week on Dataset 2. The Transformer is the closest.

## Checks

When each model is scored on its own hours with its own settings (`--all-hours`, plus `--merge-gap 0 --grace 0` for Ridge), the script reproduces the numbers each model reported:

- Transformer: P 0.906, R 0.630, ROC-AUC 0.919, 25 episodes, S 0.895
- Isolation Forest: 5.33 attacks, S 0.457
- Ridge: F1 0.725, ROC-AUC 0.866, 32 episodes

## Still to add

- XGBoost and the XGBoost + Isolation Forest ensemble (Suhas). To add them, add their `hourly_scores` file to `load_models()` in the script.
