# Test set scoring

## 1. Purpose and files

This step scores every frozen model on new data with fixed settings. Nothing is trained, refitted or recalibrated. We only load saved models and saved thresholds, score the hours, and count the alarms the same way for every model.

All steps are in `notebooks/Test_Scoring.ipynb`. Every number below comes from the output of that notebook. It comes after the shared comparison (`evaluation/comparison.md`) and the freeze prep on the `freeze-prep` branch.

We use the notebook twice:
1. **Dry run on Dataset 2.** Every model must give the same numbers as the shared comparison (`evaluation/results/comparison_gap2_grace3.csv`). This shows the notebook scores each model the same way as before.
2. **One run on the Test set**, only after the team freezes all choices and thresholds. The Test set is scored once only.

- Data used: `data/processed/dataset2_eval.csv` for the dry run. `data/processed/test_eval.csv` (made by `notebooks/Test_Prep.ipynb`) and the raw `data/BATADAL_test_dataset.csv` for the Test run.
- Results go to `evaluation/results/<RUN_ON>/`.

## 2. Summary

| Step | Result |
|---|---|
| 1. Setup | One switch (`RUN_ON`, `UNLOCK_TEST`) and all frozen settings in one cell |
| 2. Load | Dataset 2: 4,177 rows, 492 attack hours, attacks 1 to 7. No attack hour in the first 23 hours |
| 3. Ridge | Errors match `ridge_errors.csv` (largest difference 4.7e-13). S 0.840 |
| 4. Transformer | Vincent's Test path on raw Dataset 2 gives the same errors as `dataset2_eval.csv` (difference 0). S 0.895 |
| 5. XGBoost, IF + PCA, ensemble | Saved models reproduce the saved scores (XGBoost about 1e-10, IF + PCA about 1e-16). Mean S 0.543, 0.696 and 0.746 |
| 6. Summary | One table for all models, plus a per-attack table |
| 7. Check | All 5 models match the shared comparison |

## 3. What is frozen

The freeze is commit `bfa2f8d` on `freeze-prep`, on top of `origin/main` at `116379b` (which includes Suhas's branch, PR #4).

| Item | Frozen choice | File |
|---|---|---|
| Ridge | 28 per-sensor Ridge models | `models/Ridge/ridge_model.joblib` |
| Ridge calibration | `topk_percentile` threshold 0.997494 | `models/Ridge/calibration/calibration.pt` (made by `models/Ridge/freeze_calibration.py`) |
| Transformer | 12-hour window, seed 0, step 25,000 | `models/Transformer/Notebook/Model Weights/best_model.pt` |
| Transformer calibration | `topk_percentile` threshold 0.998448 | `models/Transformer/Notebook/Model Weights/calibration/calibration.pt` |
| XGBoost | 5 fold models per seed (fold 1, 3, 4, 5, 6), scored in turn by row position | `results/xgboost_iforest_ensemble/xgb/seed{k}/models/` |
| IF + PCA | Suhas's scaler, PCA (20 components) and Isolation Forest (500 trees), seeds 0, 1, 2 | `models/Ensemble/iforest_pca_models/seed{k}.joblib` |
| Calibration references | Each detector's saved Calibration scores | `xgb/seed{k}/xgb_hourly_scores.csv`, `iforest/seed{k}/iforest_hourly_scores.csv` |
| Ensemble | Max of the XGBoost and IF + PCA Calibration percentiles, seed k with seed k | |
| Thresholds (XGBoost, IF + PCA, ensemble) | Percentile thresholds set on Calibration | `models/Ensemble/notebook_outputs/all_runs.csv` (his IF + PCA, percentile scale) |
| Score | `topk_percentile`, top 3 sensors (Ridge and Transformer) | |
| Budget | At most 1% of hours and 1 false-alarm episode per week, on Calibration | |
| Episodes | Merge gap 2: alarm runs with 2 or fewer quiet hours between them are one episode | |
| Grace | The 3 hours after each attack are not counted | |
| Warm-up | The first 23 hours are not scored, but they are fed in as history | |

The thresholds for XGBoost, IF + PCA and the ensemble, per seed:

| Detector | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| XGBoost | 0.992380 | 0.991794 | 0.992380 |
| IF + PCA | 0.992966 | 0.992966 | 0.992966 |
| Ensemble (max) | 0.995897 | 0.996483 | 0.996483 |

We use the percentile thresholds from `all_runs.csv`, not the per-fold thresholds in `xgb/seed{k}/models/thresholds.csv`. The per-fold ones score XGBoost model by model. The shared comparison and the ensemble use the percentile scale.

## 4. How the switch works

- `RUN_ON = "dataset2"`: the dry run. This is the default.
- `RUN_ON = "test"`: the Test run. It only works if `UNLOCK_TEST = True` is also set by hand. Otherwise the first cell stops with an error.
- Nothing else changes between the two runs. The models, thresholds and settings are the same.
- The checks that need saved Dataset 2 results (3b, 5b, 5c, 7) only run on Dataset 2. On the Test set they print that they are skipped. The Calibration checks in 5b and 5c run in both runs.

## 5. How each model is scored on new data

**Ridge.** Each of the 28 sensors is predicted from the other 35 features, one hour at a time. We keep the squared errors, load the frozen calibration and run Vincent's `run_evaluation` with `grace_hours = 3` on the hours from 23 onward.

**Transformer.**
- Dataset 2: we score `dataset2_eval.csv` with Vincent's loader, as his inference notebook does.
- Test set: Vincent's loader only takes the RAW Test file, with `build_dataloaders(test_path=..., unlock_test=True, window=12)`. It rounds to 2 decimals, drops the 7 columns and scales with `scaler.joblib` itself. Its Test hours have no labels, so we join our `label` and `attack_id` from `test_eval.csv` by `DATETIME`. The notebook checks that the join covers every scored hour and all 407 attack hours.
- We checked this Test path on the raw Dataset 2 file, which has the same format as the Test file. It gives exactly the same errors as `dataset2_eval.csv`.
- Then the frozen calibration and `run_evaluation` with `grace_hours = 3`, on the hours from 23 onward.

**XGBoost.**
- We build Suhas's features with `build_features()` from `models/XGBoost/xgb_data.py`: lags, differences and rolling statistics over up to 24 hours, without time features. This function ignores `ATT_FLAG`, `label` and `attack_id`, so it works on the Test file as it is.
- Each hour is scored by one fold model, taken in turn by row position: row 0 by fold 1, row 1 by fold 3, row 2 by fold 4, row 3 by fold 5, row 4 by fold 6, then fold 1 again. This is the rule in `single()` in `train_xgboost.py`, which Suhas used for Validation and Calibration. So an hour's score depends on its row position. On the Test set, row 0 is the first scored hour.
- On Dataset 2 the dry run copies Suhas's saved scores instead. Each attack window is scored by the fold model that never saw that attack (attacks 2 and 7 by the final model), and the other hours go in turn. That choice uses the attack labels, so it can only be done on Dataset 2. On the Test set no fold model has seen any Test hour, so every hour goes in turn.
- We checked the in-turn rule on the Calibration block. It reproduces Suhas's saved Calibration scores to about 1e-10.

**IF + PCA.** We load the saved scaler, PCA and Isolation Forest per seed and score each hour as `-forest.score_samples(...)`, the same as `isolation_forest.py`.

**Percentiles and the ensemble.**
- Each detector's score becomes a percentile: the share of its frozen Calibration scores at or below it (as `normalise()` in `models/Ensemble/ensemble.py`).
- The ensemble score is the larger of the two percentiles.
- An alarm is raised when a score is above its frozen threshold.
- We read the saved scores and thresholds exactly (`float_precision="round_trip"`). Some hours have a percentile exactly equal to the threshold. Our first dry run read the thresholds with pandas' default parser, which was 1 bit too low. That gave 5 extra XGBoost alarms in seed 0 and 3 in seed 2, and a mean S of 0.565 instead of 0.543. With exact reading, everything matches.

**Counting.**
- All five models in the summary table are counted by `evaluate()` from `evaluation/compare_models.py`, the code behind the shared comparison, with merge gap 2 and grace 3.
- For Ridge and the Transformer it takes their `topk_percentile` scores and alarms from `run_evaluation`. Both give the same S, precision, recall, F1, delay and episodes.
- Only ROC-AUC and PR-AUC differ a little (PR-AUC by about 1.5e-4), because the two files use slightly different formulas for the area.
- We do not use `ensemble.evaluate()`, because it needs `ATT_FLAG` values, which the Test file does not have.
- For XGBoost, IF + PCA and the ensemble we report each seed and the mean over seeds 0, 1 and 2.

## 6. The warm-up rule

- The XGBoost and IF + PCA features use up to 24 hours of history, so hour 23 is the first hour with full features. The 12-hour Transformer needs 11 hours of history.
- We build the features on **all** hours and score from hour 23 onward. If we cut the first 23 hours before building the features, XGBoost would lose another 23 hours.
- Every model is scored on the same hours: 4,154 on Dataset 2, and 2,066 on the Test set (2,089 − 23).
- No attack falls in the first 23 hours. The Test set starts on 2017-01-04 00:00 (`docs/audit.md`), and attack 8 starts on 2017-01-16 09:00.

## 7. Which score we report

For Ridge and the Transformer we report `topk_percentile` only. `run_evaluation` also prints `mean` and `topk_ratio`. On the Test set we do not switch to one of them, even if it looks better, because choosing after seeing Test results would break the freeze.

"Inside budget" means at most 1 false-alarm episode per week **and** at most 1% of normal hours under alarm, on the scored data.

## 8. How to run it

1. Use the environment in `requirements.txt` (Python 3.13.5). On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch. The notebook's first cell does this.
2. Open `notebooks/Test_Scoring.ipynb` and run all cells. With `RUN_ON = "dataset2"` the last check must print `match True` for every model.
3. For the Test run, after the freeze: set `RUN_ON = "test"` and `UNLOCK_TEST = True` in the first cell, then run all cells once.
4. The results are in `evaluation/results/<RUN_ON>/`:
   - `overall.csv`, `per_attack.csv` and `settings.json` (the run, the git commit, the frozen files, the thresholds and the library versions);
   - `ridge/` and `transformer/`: the full `run_evaluation` output (metrics, per-attack results, hourly scores, plots);
   - `xgboost/`, `iforest_pca/` and `ensemble/`: hourly scores per seed, results per seed and per-attack results.

## 9. Results

### Dataset 2 dry run

4,154 scored hours, merge gap 2, grace 3, `topk_percentile` for Ridge and the Transformer, mean over 3 seeds for the others.

| Model | S | Precision | Recall | F1 | ROC-AUC | PR-AUC | Mean delay (h) | False-alarm episodes / week | Normal hours alarmed | Attacks detected | Inside budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ridge | 0.840 | 0.899 | 0.598 | 0.718 | 0.865 | 0.735 | 7.7 | 1.34 | 0.91% | 7 | No |
| Transformer (12 h, seed 0) | 0.895 | 0.906 | 0.630 | 0.743 | 0.919 | 0.792 | 1.7 | 1.15 | 0.88% | 7 | No |
| XGBoost | 0.543 | 0.732 | 0.259 | 0.383 | 0.748 | 0.445 | 17.5 | 0.85 | 1.28% | 4.3 | 0 of 3 seeds |
| IF + PCA | 0.696 | 0.813 | 0.295 | 0.433 | 0.674 | 0.430 | 12.1 | 1.26 | 0.92% | 6.0 | 1 of 3 seeds |
| Ensemble (max) | 0.746 | 0.821 | 0.492 | 0.615 | 0.860 | 0.619 | 20.6 | 0.92 | 1.45% | 7.0 | 0 of 3 seeds |

Check against the shared comparison:

| Model | Ours: S / attacks / delay / episodes per week | Shared comparison | Match |
|---|---|---|---|
| Ridge | 0.840 / 7 / 7.7 / 1.34 | 0.840 / 7 / 7.7 / 1.34 | Yes |
| Transformer | 0.895 / 7 / 1.7 / 1.15 | 0.895 / 7 / 1.7 / 1.15 | Yes |
| XGBoost | 0.543 / 4.3 / 17.5 / 0.85 | 0.543 / 4.3 / 17.5 / 0.85 | Yes |
| IF + PCA | 0.696 / 6.0 / 12.1 / 1.26 | 0.696 / 6.0 / 12.1 / 1.26 | Yes |
| Ensemble (max) | 0.746 / 7.0 / 20.6 / 0.92 | 0.746 / 7.0 / 20.6 / 0.92 | Yes |

Every metric matches to within 1e-4. The largest difference is the Transformer's ROC-AUC and PR-AUC (about 2e-6), because our torch version is slightly different from Vincent's.

Per attack on Dataset 2 (delay from the real start, and hours caught out of the attack's hours; for the 3-seed models, seed 0 / 1 / 2):

| Attack (hours) | Ridge | Transformer | XGBoost | IF + PCA | Ensemble (max) |
|---|---|---|---|---|---|
| 1 (50) | 0 h, 14 | 0 h, 45 | miss / miss / miss | 0 h, 38 / 38 / 38 | 0 h, 38 / 38 / 38 |
| 2 (24) | 0 h, 19 | 0 h, 19 | 3 h, 19 / 19 / 19 | miss / miss / miss | 3 h, 16 / 19 / 16 |
| 3 (60) | 39 h, 17 | 0 h, 27 | 25 h, 18 / 18 / 17 | 0 h, 5 / 4 / 3 | 25 / 25 / 45 h, 17 / 18 / 15 |
| 4 (94) | 0 h, 89 | 0 h, 85 | 10 h, 72 / 75 / 78 | 20 h, 6 / 8 / 4 | 10 h, 73 / 73 / 74 |
| 5 (60) | 0 h, 60 | 5 h, 33 | 8 h, 16 / 16 / 15 | 6 h, 18 / 18 / 18 | 6 h, 18 / 19 / 19 |
| 6 (94) | 10 h, 69 | 2 h, 76 | miss / miss / miss | 11 h, 75 / 75 / 75 | 11 h, 75 / 75 / 75 |
| 7 (110) | 5 h, 26 | 5 h, 25 | miss / 102 h / miss, 0 / 1 / 0 | 45 / 17 / 45 h, 4 / 5 / 4 | 45 / 101 / 101 h, 4 / 3 / 3 |

### Test results

To be added after the single Test run.

## 10. Cautions

- XGBoost on the Test set uses the in-turn rule for every hour, but the dry run on Dataset 2 used Suhas's out-of-fold rule for the attack windows. The in-turn code is checked on the Calibration block, not on attack hours.
- The saved IF + PCA models are tied to scikit-learn 1.9.0, and the Transformer was checked with torch 2.13.0. Use `requirements.txt`.
- On the Test set the Transformer reads the raw Test file and the other models read `test_eval.csv`. Both are prepared the same way (checked on raw Dataset 2), but they are two separate files.
- No model is inside the budget on Dataset 2 in every seed. We keep the frozen thresholds anyway, because changing them after looking at Dataset 2 or Test results would break the freeze.
