# XGBoost detector

Supervised detector after Nemade et al. (2026), "Cyber-Physical Attack Detection in Water Distribution Systems Using a Hybrid Ensemble of XGBoost, Isolation Forest, and LSTM Autoencoder on the BATADAL Dataset" (JIDMIS 3(3s)). It is the supervised member of the planned three-model ensemble (Transformer reconstruction + XGBoost + Isolation Forest).

```bash
cd models/XGBoost
python train_xgboost.py --dataset-path ../../data/processed --out-dir runs/xgb --seeds 0 1 2
# or run XGBoost_BATADAL.ipynb (same code; outputs kept in notebook_outputs/)
```

## Labels

| Data | Use |
|---|---|
| Dataset 1 (all normal) | negatives: train block trains, calibration block sets the threshold, validation block measures false alarms |
| Dataset 2, `ATT_FLAG = 1` | positives (219 hours) |
| Dataset 2, `ATT_FLAG = -999` | **dropped**: never a training or hour-level evaluation label |

The 219 labeled hours cover only 5 of the 7 real attacks (1, 3, 4, 5, 6). Attacks 2 and 7 have no labeled hour, so no model trains on them: they are a true hold-out. The `-999` hours inside a real attack window are scored only to measure detection delay.

## Method (what follows the paper, what does not)

Follows the paper: 400 trees, depth 6, learning rate 0.05, 0.8 row and column subsampling, `scale_pos_weight` = negatives / positives; lag 1-3, first differences at 1 and 3, rolling mean/std/deviation (3, 6, 24 h) for tank levels and pressures, cyclical hour and weekday (the last off by default, see below). This gives 391 features from the 36 non-constant sensors (the paper reports 376 from 43); the default run leaves out the four hour/weekday features and uses 387.

Differs from the paper, on purpose:

- **No in-sample scoring.** The paper trains on 80% of a 1,314-row sample and then reports on all 4,177 hours, with the threshold picked to maximise F1 on the same series. Here, the model that scores an attack never trained on it (leave-one-attack-out), the threshold comes from the calibration block under the project's false-alarm budget (at most 1 episode per week, at most 1% of hours), and false alarms are counted on the validation block.
- **Trailing windows only**, so the score for hour t uses hours up to t (the paper's 3-hour smoothing is centred).
- **Features are built inside each contiguous block**, never across the 24-hour gaps, and the first 23 hours of each block are dropped. The scored hours are exactly those the 24-hour Transformer scores, so scores line up for the ensemble.
- **No negative subsampling**: all of the Dataset 1 train block is used, balanced with `scale_pos_weight`.
- Trees do not need scaling; the processed files are already Training-scaled.

## Results (3 seeds, mean)

Default run: no time features, `confirmed_only` labels, episodes merge alarms at most 2 quiet hours apart (`MERGE_GAP_HOURS`) and 3 grace hours after each attack are left out of the Dataset 2 false-alarm count (`GRACE_HOURS`); both match the Transformer evaluation and are fixed for every detector.

| | no time features (default) | with `--time-features` |
|---|---|---|
| Attacks detected (of 7) | 5.0 | 4.0 |
| Unseen attacks 2, 7 detected (of 2) | 1.3 | 1.0 |
| Mean delay, detected only (h) | 15.1 | 10.9 |
| Recall on confirmed labeled hours | 0.28 | 0.28 |
| Recall on schedule labels (every window hour) | 0.27 | 0.25 |
| Validation alarm hours per clean week | 1.47 | 1.06 |
| ROC-AUC / PR-AUC (confirmed labels) | 0.69 / 0.48 | 0.70 / 0.49 |
| ROC-AUC (schedule labels) | 0.75 | 0.75 |
| Dataset 2 normal hours that alarm (era check) | 1.5% | 1.4% |
| Dataset 2 normal-hour false-alarm episodes per week | 1.15 | 0.85 |
| Approximate S-score | 0.60 | 0.55 |

How to read these:

- **No precision or F1.** Their true positives come from Dataset 2 and their false positives from the validation block, so they only reflect the chosen 219:1,705 mix. Development tables give alarm hours per clean week; compute precision and F1 on the Test set only. The S-score is "approximate" for the same reason: recall (Dataset 2) and TNR (validation) come from different series, and the official one needs one labelled series.
- **AUC compares like with like.** Every clean hour is scored by one fold model (round-robin), as every attack hour is; averaging the fold models would give clean hours lighter-tailed scores and inflate the AUC.
- **Time features were dropped by default** because fewer attacks are caught with them (4.0 vs 5.0). That choice was made by looking at Dataset 2 results, so it is a tuned choice, not a pre-registered one. Seeds also disagree on which attacks are caught.
- The paper's F1 of 0.93 is an in-sample figure and is not comparable to these.

### Era check

Dataset 1 (all negatives) is from 2014 and Dataset 2 (all positives) from 2016. As in the team's Transformer evaluation, the Dataset 2 normal hours (outside every attack window, minus 3 grace hours after each attack) are scored too; they are never trained on under `confirmed_only`. They alarm on 1.4-1.5% of hours at 0.85-1.15 episodes per week, against a budget of 1% and 1 episode per week set on 2014 data. That is a mild excess, not the large one that would mean the model mostly reacts to the year, but it is not zero. Part of it can be genuine drift between years. False alarms are still also reported on the validation block (2014): 0.84 episodes per week.

### Label-rule sensitivity (`--label-rule`)

Trained under three rules, all evaluated on the same labels (3 seeds, mean):

| | `confirmed_only` | `minus999_normal` | `schedule` |
|---|---|---|---|
| Attacks detected (of 7) | 5.0 | 5.0 | 7.0 |
| Mean delay (h) | 15.1 | 10.2 | 23.6 |
| Recall, confirmed labels | 0.28 | 0.26 | 0.62 |
| Recall, schedule labels | 0.27 | 0.21 | 0.46 |
| Validation alarm hours per week | 1.5 | 1.5 | 4.7 |
| Dataset 2 normal-hour episodes per week | 1.15 | 0.85 (in-sample) | 1.68 |
| ROC-AUC, confirmed / schedule | 0.69 / 0.75 | 0.73 / 0.77 | 0.96 / 0.85 |
| Dataset 2 normal-hour alarm rate | 1.5% | 0.7% (trained on, in-sample) | 2.6% |

`schedule` trains on every hour of every window, so the model also sees attack onsets and detects all 7, but its alarms on clean data run well above the budget (4.7 alarm hours per week on validation, 2.6% of Dataset 2 normal hours and 1.7 episodes per week there), (a likely reason, not tested here: window hours that look normal are taught as attacks). Its figures are not directly comparable with the other two for that reason. `minus999_normal` trains on the outside-window Dataset 2 hours, so its era figure is in-sample and should not be read as a check.

## Known limits

- Labeled hours are the middle of some attacks (for example 7 of the 60 hours of attack 5), so under `confirmed_only` the onset is not in the positives; the `schedule` rule covers it.
- XGBoost gives one score per hour, not per sensor, so it cannot enter the proposal's per-sensor fusion. It is used only in the system-level fusion variant and in the XGBoost + Isolation Forest ensemble.
- Only 7 attacks exist, and the S-score and any F1 must be recomputed on the Test set.

## Why the paper's numbers are higher (`paper_protocol_check.py`)

Re-running the paper's recipe on Dataset 2 (1,095 random normal rows + 219 attack rows, chronological 80/20 of that sample, 400 trees, threshold maximising F1 on the full series) reproduces its XGBoost row (paper: P 0.958, R 0.626, F1 0.757, AUC 0.814; here P 0.98-1.00, R 0.635, F1 0.77, AUC 0.84-0.87). The same model scored only on rows it never trained on: recall 0, AUC 0.55-0.65. The paper's gain is in-sample scoring on Dataset 2, plus a threshold tuned on the evaluated series, plus a centred 3-hour smoothing of the ensemble score. Its Isolation Forest + PCA row also reproduces (AUC 0.79 vs 0.80) and, being unsupervised, scores the same when trained on Dataset 1 only (AUC 0.80), so that recipe carries over to this project.
