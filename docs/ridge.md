# Ridge regression baseline

## 1. Purpose and files

This is our simplest model. For each of the 28 continuous sensors, we train a Ridge regression that predicts that sensor from the other 35 features in the same hour. If the prediction errors are big, the hour looks unusual, and we raise an alarm.

All steps are in `notebooks/Ridge.ipynb`. Every number below comes from the output of that notebook. It comes after the data pipeline (`docs/pipeline.md`).

- Data used: the files in `data/processed/`: `train_scaled.csv`, `val_scaled.csv`, `cal_scaled.csv` and `dataset2_eval.csv` (Dataset 2 with the labels from the 7 attack windows).
- Training is used to fit the models, Validation to pick alpha and the normal error size, Calibration to set the threshold, and Dataset 2 to check the results.
- The Test set is not used. It stays untouched until all choices and thresholds are frozen.

## 2. Summary

| Step | Result |
|---|---|
| 1. Load | Training 5,256 × 38, Validation 1,728 × 38, Calibration 1,729 × 38, Dataset 2 4,177 × 40. 28 continuous + 8 on/off = 36 features |
| 2. One sensor (`L_T1`) | Training and Validation errors almost the same (0.0713 and 0.0716). Error inside attacks 0.2547 |
| 3. All 28 sensors | Every attack makes at least one sensor react |
| 4. Alpha | 23 of 28 sensors picked 0.0001. Each sensor uses its own best alpha |
| 5. Score | Error / normal error per sensor, then the average of the top 3 sensors per hour |
| 6. Threshold | 7.276 (10 episodes on Calibration, inside the budget) |
| 7. Dataset 2 | All 7 attacks detected. F1 0.725, ROC-AUC 0.866 |
| 8. False alarms | 32 episodes in 21.9 weeks, mostly from `P_J422` and pump 6 |
| 9. Without `P_J422` | About the same. We keep all 28 sensors |
| 10. Save | 3 files in `models/Ridge/` |

## 3. One sensor first: `L_T1`

We first fit one model (alpha 1) for `L_T1` (tank 1 level), to check that the idea works.

| Data | Mean absolute error |
|---|---|
| Training | 0.0713 |
| Validation | 0.0716 |
| Dataset 2, outside attacks | 0.074 |
| Dataset 2, inside attacks | 0.2547 |

- Training and Validation are almost the same, so the model is not overfitting.
- Attacks 3 (0.4014) and 4 (0.6264) are clearly caught. These attacks change tank 1 directly.
- Attack 2 (0.4671) also shows up. Attack 2 fakes the pump 10/11 readings, and those readings are inputs to the `L_T1` model, so the prediction goes wrong.
- Attack 7 (0.1044) is mostly missed on `L_T1`. In attack 7, `L_T1` and pumps 1/2 were replayed together, so the inputs and the target still fit each other.

**Caution:** a big error on a sensor doesn't always mean that sensor is attacked. Attack 2 does not touch `L_T1`, but its error is still big.

## 4. All 28 sensors

We then fit one model per continuous sensor (alpha 1 at this point).

- For every sensor, the Validation error and the Dataset 2 outside-attack error are about the same.
- Tanks have much bigger normal errors than flows and pressures (for example `L_T4` 0.150 vs `F_PU6` 0.002 on Validation). A tank level depends on the past hours, and Ridge only sees the current hour.
- Every attack has at least one sensor that reacts:

| Attack | Top sensor | Mean error |
|---|---|---|
| 1 | `L_T7` | 0.238 |
| 2 | `L_T1` | 0.467 |
| 3 | `L_T1` | 0.401 |
| 4 | `L_T1` | 0.626 |
| 5 | `F_PU7` | 0.242 |
| 6 | `L_T4` | 0.485 |
| 7 | `L_T4` | 0.285 |

- The raw errors can't be added up directly, because they have very different sizes. A tank error of 0.15 is normal, but for a flow it would be huge. The score in section 6 fixes this.

## 5. Settings: alpha

- We first tried alpha from 0.01 to 100. The best was at the smallest value, so we extended the range down to 0.0001. The final range is 0.0001, 0.001, 0.01, 0.1 and 1.
- Alpha is picked per sensor, by the smallest mean absolute error on Validation.
- 23 of 28 sensors picked 0.0001. Two picked 0.001 (`F_PU6`, `P_J269`), one picked 0.01 (`F_PU11`) and two picked 0.1 (`L_T7`, `P_J422`).
- Below 0.01 the errors barely change (for example `L_T1`: 0.0609 at 0.01, 0.0602 at 0.0001). So Ridge needs very little penalty here.
- Each sensor uses its own best alpha. All later steps use these models.

## 6. Settings: score and threshold

**Score per hour**
1. For each sensor, divide the absolute error by that sensor's mean absolute error on Validation (`normal_size`).
2. Take the average of the 3 biggest scaled errors in that hour.

This is the same top-3 rule as the fusion in the proposal. An attack usually changes a few sensors, so averaging all 28 would hide it. Using 3 instead of 1 makes one noisy sensor less likely to cause an alarm alone.

**Threshold**
- We started with the 99th percentile of the Calibration scores: 6.81. This gave 18 alarm hours in 17 episodes over 10.3 weeks. Calibration has no attacks, so all of these are false alarms. That is over our budget of 1 episode per week (10 episodes).
- So we raised it to the lowest value that gives 10 or fewer episodes: **7.276**. Calibration then has 11 alarm hours in 10 episodes.
- The false alarms at the 99th percentile came from: `P_J422` 12, `F_PU6` 2, and `P_J317`, `P_J269`, `P_J306` and `F_PU10` 1 each.
- The threshold is set on Calibration only. Dataset 2 is not used to set it.

## 7. Results on Dataset 2

| Metric | Dataset 2 |
|---|---|
| ROC-AUC | 0.866 |
| PR-AUC | 0.741 |
| Precision | 0.893 |
| Recall | 0.61 |
| F1 | 0.725 |
| False alarm hours | 36 of 3,685 |
| False alarm episodes | 32 in 21.9 weeks |

Per attack (delay = hours from the start of the attack to the first alarm):

| Attack | Hours | Alarm hours | Detected | Delay (hours) |
|---|---|---|---|---|
| 1 | 50 | 29 | Yes | 0 |
| 2 | 24 | 19 | Yes | 0 |
| 3 | 60 | 10 | Yes | 43 |
| 4 | 94 | 86 | Yes | 0 |
| 5 | 60 | 60 | Yes | 0 |
| 6 | 94 | 70 | Yes | 10 |
| 7 | 110 | 26 | Yes | 5 |

- All 7 attacks are detected, and 4 of them in their first hour.
- Attack 3 is caught very late (43 hours). Attack 7 has only 26 alarm hours out of 110.
- Attacks 2 and 7 have no labels in the original Dataset 2 file (`ATT_FLAG`). Our labels come from the 7 attack windows. Ridge still catches both.

**Caution:**
- 32 false alarm episodes in 21.9 weeks is about 1.5 per week, which is over the budget, even though Calibration was inside it.
- These are development scores. We looked at Dataset 2 while building the model. The final numbers come from the Test set.

## 8. False alarm findings

| Group | False alarm hours |
|---|---|
| Far from any attack | 29 |
| Within 24 hours before an attack | 6 |
| Within 24 hours after an attack | 1 |

| Top sensor | False alarm hours |
|---|---|
| `P_J422` | 16 |
| `F_PU6` | 6 |
| `P_J415` | 5 |
| `F_PU1` | 3 |
| `F_PU10` | 3 |
| `F_PU11`, `P_J307`, `P_J14` | 1 each |

- **Not after-effects.** Only 1 false alarm is right after an attack (2016-09-16 01:00, the hour after attack 1 ends).
- **`P_J422` and pump 6 cause most of them.** `P_J422` has a very small normal error, about 0.001 (0.0011 in the step 4 table), so a small change gives a big scaled error. The `F_PU6` false alarms come in pairs of hours in a row, at the times pump 6 runs normally. Pump 6 runs only rarely, so its running hours look unusual.
- **Some are exactly 1 hour before an attack starts:** 2016-09-26 10:00 (attack 2), 2016-10-29 18:00 (attack 4) and 2016-12-14 14:00 (attack 7). These might be the start of the attack, just before the labelled window.
- **3 are in the first hours of Dataset 2:** 2016-07-04 03:00, 04:00 and 06:00.

## 9. Without `P_J422`

We tried the score with 27 sensors (all except `P_J422`). The models are the same, only the score changes. The threshold was set again on Calibration with the same rule.

| | All 28 sensors | Without `P_J422` |
|---|---|---|
| Threshold | 7.276 | 6.031 |
| Calibration alarm hours / episodes | 11 / 10 | 11 / 10 |
| ROC-AUC | 0.866 | 0.865 |
| PR-AUC | 0.741 | 0.747 |
| Precision | 0.893 | 0.882 |
| Recall | 0.610 | 0.624 |
| F1 | 0.725 | 0.731 |
| Dataset 2 false alarm hours | 36 | 41 |
| Dataset 2 false alarm episodes | 32 | 35 |

- The threshold dropped, and other sensors' false alarms took the place of the `P_J422` ones. The results stayed about the same, and the delays for all 7 attacks did not change.
- Calibration showed no difference, and we only make choices on Calibration. So **we kept all 28 sensors**.

## 10. Saved files

Folder: `models/Ridge/`

| File | What it is |
|---|---|
| `ridge_scores.csv` | Hourly score and alarm (1 or 0) for Validation, Calibration and Dataset 2. Columns: `DATETIME`, `split`, `score`, `alarm`, `label`, `attack_id` (the last two only for Dataset 2). 7,634 rows. For fusion and for comparing with the other models |
| `ridge_errors.csv` | The raw error (real minus predicted) of each of the 28 sensors for every hour in the same three blocks, with `DATETIME` and `split`. For checking which sensor caused an alarm |
| `ridge_model.joblib` | A dictionary with the 28 fitted models (`models`), the best alphas (`alpha`), `normal_size`, the `threshold` (7.276) and the feature list (`features`). For scoring the Test set at the end |

Alarm hours in `ridge_scores.csv`: Validation 6 out of 1,728, Calibration 11, Dataset 2 336.

**Caution:** Validation was used to choose the alphas and `normal_size`, so its 6 alarm hours are not a fair test. They only show the threshold is not far off on another normal block. Calibration's 11 alarm hours and 10 episodes meet the budget by design, since we chose the threshold on Calibration. The honest check on normal hours the threshold never saw is Dataset 2 outside the attack windows: 36 false alarm hours and 32 episodes in 21.9 weeks, which is over the episode budget.

**How to load**

```python
import pandas as pd
import joblib

scores = pd.read_csv("../models/Ridge/ridge_scores.csv", parse_dates=["DATETIME"])
errors = pd.read_csv("../models/Ridge/ridge_errors.csv", parse_dates=["DATETIME"])

ridge = joblib.load("../models/Ridge/ridge_model.joblib")
models = ridge["models"]          # one Ridge model per continuous sensor
normal_size = ridge["normal_size"]
threshold = ridge["threshold"]
features = ridge["features"]

# predict one sensor
target = "L_T1"
inputs = [c for c in features if c != target]
pred = models[target].predict(df[inputs])   # df: a scaled table, e.g. from data/processed/
```

The paths assume the code runs from the `notebooks` folder. The Test set must go through the same rounding, column drop and scaler as in `docs/pipeline.md` before it is scored.

## 11. Ridge with the shared evaluation

We also scored Ridge with Vincent's shared code (`calibrate` and `run_evaluation` in `models/Transformer/evaluate_recon_model.py`), so Ridge is scored the same way as the Transformer. The steps are in `notebooks/Ridge_shared_eval.ipynb`, and the numbers in this section come from its output.

- We use the squared errors from `ridge_errors.csv`, because the shared code expects bigger = worse and Ridge's errors are signed.
- The threshold is set on Calibration for three scores (`topk_percentile`, `mean`, `topk_ratio`), with the top 3 sensors, at most 1% of hours under alarm and at most 1 episode per week.
- We use the team's shared settings, the same as the Transformer and Isolation Forest: merge gap 2 (alarm runs separated by 2 or fewer quiet hours count as one episode) and grace hours 3 (the 3 hours after each attack ends aren't counted). So Dataset 2 has 492 attack hours and 3,664 clean hours.
- All 7 attacks are detected with all three scores.
- Our first run used merge gap 0 and grace 0. With 2 and 3, the episodes per week dropped slightly (1.46 to 1.38 for `topk_percentile`), the threshold stayed the same, and Ridge is still outside the budget on Dataset 2.

| | Our Ridge (section 7) | `topk_ratio` | `topk_percentile` |
|---|---|---|---|
| Precision | 0.893 | 0.893 | 0.894 |
| Recall | 0.610 | 0.630 | 0.598 |
| F1 | 0.725 | 0.739 | 0.716 |
| ROC-AUC | 0.866 | 0.864 | 0.864 |
| PR-AUC | 0.741 | 0.749 | 0.731 |
| False alarm episodes per week | 1.46 | 1.33 | 1.38 |

`topk_ratio` is closest to our method, and the numbers are almost the same, so both pieces of code agree. Our own Ridge (section 7) counts episodes without a merge gap or grace hours, so its 1.46 per week uses a slightly stricter rule. The `topk_percentile` threshold is 0.9975, close to its ceiling of 1.

**Budget:** all three scores are outside the budget on Dataset 2, same as in section 7. They all have more than 1 false alarm episode per week (`topk_percentile` 1.38, `mean` 1.24, `topk_ratio` 1.33). On Validation, `topk_percentile` and `topk_ratio` are also outside the budget. Only `mean` stays inside.

**Ridge vs the Transformer** (both with `topk_percentile` and the same settings: top 3 sensors, merge gap 2, grace hours 3). The Transformer numbers are from `models/Transformer`, the frozen 12-hour model, seed 0.

| | Ridge | Transformer |
|---|---|---|
| Precision | 0.894 | 0.906 |
| Recall | 0.598 | 0.630 |
| F1 | 0.716 | 0.743 |
| ROC-AUC | 0.864 | 0.919 |
| PR-AUC | 0.731 | 0.792 |
| S score | 0.840 | 0.895 |
| Mean delay (hours) | 7.7 | 1.7 |
| False alarm episodes per week | 1.38 | 1.15 |

The Transformer is better on every measure, mostly in ranking (ROC-AUC, PR-AUC) and speed (mean delay). This is likely because it sees the last 12 hours, while Ridge sees one hour at a time. Both are outside the budget on Dataset 2.

**Caution:** the S score is about 0.84 to 0.85 (0.840, 0.846 and 0.848). The competition's S scores are on the Test set, so ours (on Dataset 2) can't be compared with them yet.

**Files:** everything is saved in `models/Ridge/evaluation/`: `metrics.json`, `per_attack.csv`, `hourly_scores.csv`, and the plots (calibration and timeline for each score, one plot per attack, top sensors per attack, detection delay and three false alarms).

## 12. Still open

1. **Dataset 2 is over the false alarm budget.** 32 episodes in 21.9 weeks is about 1.5 per week, even though Calibration is inside the budget.
2. **A 2-hours-in-a-row alarm rule.** Raising an alarm only when 2 hours in a row are above the threshold could remove many single-hour false alarms. This rule should be the same for every model, so it is a team decision.
3. **Attack 3 is detected 43 hours late.**
4. **The Test set is scored only at the end**, after all choices and thresholds are frozen.
