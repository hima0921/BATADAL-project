# Scoring and metrics

This document explains how the Transformer reconstruction model turns its output into alarms, and what each number in its results means. No knowledge of the model's internals is needed to follow it.

The scoring code does not depend on the Transformer. Any detector that produces one error per sensor per hour can be scored, calibrated and evaluated with it (section 8), which keeps results from different detectors comparable.

## 1. Purpose and files

The model only reconstructs windows of sensor readings. Everything that makes it a detector happens afterwards, in three steps: scoring, calibration and evaluation. This document describes those steps as they are implemented.

| File | What it does |
|---|---|
| `evaluate_recon_model.py` | Holds all scoring, threshold and metric code. Scores Dataset 2 and writes the results and plots |
| `calibrate_recon_model.py` | Runs only the calibration step and saves it as `calibration.pt`. It imports its functions from the evaluation script, so the two cannot disagree |
| `data/processed/dataset2_eval.csv` | Dataset 2 with a `label` column (1 inside the 7 real attack windows, 0 elsewhere) and an `attack_id` column |

**What each step reads**

| Step | Data | Why |
|---|---|---|
| Calibration | Calibration block of Dataset 1 (last 20%) | Clean data the model never trained on. Sets the references and the threshold |
| Transfer check | Validation block of Dataset 1 (middle 20%) | Checks that the threshold also holds on other clean data |
| Evaluation | Dataset 2 | The only labelled attacks available before the Test set is unlocked |

The Test set is never read by either script.

## 2. Summary

1. The model reconstructs each window. The squared error of the **last hour** gives one error per sensor for that hour.
2. Each sensor's error is compared with that sensor's own errors on clean data, and the result is combined into **one score per hour**.
3. The **threshold** is the lowest value that keeps false alarms on clean data inside the budget: at most 1 alarm episode per week and at most 1% of hours under alarm.
4. An hour is an **alarm** when its score is above the threshold.
5. The alarms on Dataset 2 are compared with the real attack windows to give the metrics in section 6.

The threshold is fixed before Dataset 2 is scored and is never adjusted afterwards.

## 3. Scoring

### Step 1: error per sensor, per hour

- **The model reconstructs the whole window in one pass**, 12 or 24 hours of all 36 features.
- **Only the last hour of each window is scored.** The error is `(reconstruction - real value)²` for each of the 36 features at that hour.
- **Why the last hour.** Windows slide by one hour, so every hour is the last hour of exactly one window. Scoring it there uses only data up to that hour, so an alarm at hour *t* could have been raised at hour *t* in a live system. Averaging over every window that covers the hour would use later hours and make the detection delay look better than it is.
- **Consequence.** The first `W-1` hours of each file have no window ending at them and get no score. In Dataset 2 that is the first 23 hours, about ten weeks before attack 1.

The result is a table with one row per scored hour and one column per sensor.

### Step 2: one score per hour

Three scores are computed side by side from the same error table. The threshold, the alarms and every metric are produced separately for each.

| Score | How it is built | Range | Where it comes from |
|---|---|---|---|
| `topk_percentile` | Each sensor's error becomes a percentile of that sensor's calibration errors. The score is the mean of the 3 highest percentiles | 0 to 1 | The proposal's score |
| `mean` | Plain average of the 36 raw errors | 0 and up | The proposal's comparison |
| `topk_ratio` | Each sensor's error is divided by that sensor's 99th-percentile calibration error. The score is the mean of the 3 highest ratios | 0 and up | Not in the proposal. Added to test a score with no ceiling |

**Why not just average the errors.** Sensors are not equally easy to reconstruct. A sensor the model always reconstructs poorly has large errors on normal data, and it would dominate a plain average. The other two scores first ask how unusual an error is *for that sensor*.

**The percentile.** For one sensor, it is the share of its calibration errors that are less than or equal to the new error. 0.95 means the error is larger than 95% of what that sensor showed on clean data.

**The top 3.** An attack usually touches a few sensors, not all 36. Averaging only the 3 most unusual sensors keeps a local disturbance from being diluted by 33 normal ones. The number is the `--top-k` option.

**Example.** At one hour, the three highest sensor percentiles are 1.00, 0.98 and 0.96. The `topk_percentile` score is their mean, 0.98.

**The ceiling.** A percentile cannot go above 1. Any error at or above the largest calibration error for that sensor gives exactly 1, however large it is. So the percentile score cannot tell a large error from an enormous one. `topk_ratio` was added because it has no ceiling. In the one full run so far (24-hour window, one seed), the percentile score still gave the best results, so the proposal's score remains the main one.

## 4. Calibration

Calibration uses the calibration block only. It produces two things.

**Per-sensor references**

- The sorted calibration errors of each sensor, used for the percentiles.
- The 99th percentile of each sensor's calibration errors, used for the ratios.

**One threshold per score**

- **Alarm rule.** An hour is an alarm when `score > threshold`.
- **Episode.** A run of back-to-back alarm hours is one episode. With `--merge-gap-hours g`, two runs separated by at most `g` quiet hours count as one. The default is 0.
- **Budget.** Two limits, both from the proposal, and both must hold:
  - at most 1% of clean hours under alarm (`--max-alarm-fraction`)
  - at most 1 alarm episode per week (`--max-episodes-per-week`)
- **Search.** Every distinct calibration score is tried as a threshold, from the highest down. The highest gives no alarms at all. The search keeps lowering the threshold while both limits hold, and stops at the first one that breaks either. The result is the lowest threshold inside the budget, which is the most sensitive setting the budget allows.

**What the budget means in hours.** With the 24-hour window the calibration block has 1,706 scored hours, about 10.2 weeks. The budget therefore allows at most 17 alarm hours and at most 10 episodes on that block.

**Transfer check.** The same threshold is applied to the validation block and the two false-alarm figures are reported, with a yes or no for "inside budget". This check is slightly optimistic, because the validation block was also used to choose the checkpoint.

**Ceiling warning.** If the `topk_percentile` threshold comes out at 1, no score can exceed it and that score can never raise an alarm. Both scripts print a note when this happens.

**Frozen calibration.** `calibrate_recon_model.py` saves the references, thresholds and settings in `calibration.pt`. Passing it to the evaluation script with `--calibration` reuses them unchanged. Without it, the evaluation script calibrates on the same block itself, with the same code.

## 5. Labels used in evaluation

- **Attack hour:** `label` is 1. These are the hours inside the 7 real attack windows, 492 hours in total.
- **Clean hour:** `label` is 0.
- **Left out:** any other label value, and the grace hours. `--grace-hours n` leaves out the `n` hours after each attack ends, where the system may still be recovering. The default is 0.
- **Attack number:** `attack_id` is 1 to 7 inside an attack and 0 elsewhere. It is used for the per-attack figures.

## 6. Metrics

Every metric is reported once per score type.

### Per attack

| Metric | Definition | How to read it |
|---|---|---|
| Attacks detected | An attack is detected if at least one alarm falls inside its real window | Out of 7. An alarm just before or just after the window does not count |
| Missed attacks | The numbers of the attacks with no alarm inside their window | |
| Detection delay | Hours from the attack's real start to the first alarm inside it | 0 means caught in the first hour. Lower is better |
| Mean and median delay | Over the detected attacks only | A missed attack is not in the average, so read this together with attacks detected |

The delay is measured from the real start in the attack schedule, not from the first hour flagged in the file.

### False alarms

| Metric | Definition | How to read it |
|---|---|---|
| False-alarm episodes | Alarm episodes among the clean hours | An alarm run that continues past the end of an attack counts as one, unless grace hours are set |
| Episodes per week | Episodes ÷ (clean hours ÷ 168) | Budget: at most 1 |
| Clean hours under alarm | Alarm hours among clean hours ÷ clean hours | Budget: at most 1% |
| Inside budget on Dataset 2 | Yes if both figures are within the budget | The threshold was set on Dataset 1. This shows whether it still holds on Dataset 2's clean hours |

### Hour by hour

Each scored hour falls into one of four groups.

| | Alarm | No alarm |
|---|---|---|
| **Attack hour** | True positive (TP) | False negative (FN) |
| **Clean hour** | False positive (FP) | True negative (TN) |

| Metric | Formula | Question it answers |
|---|---|---|
| Precision | TP ÷ (TP + FP) | When there is an alarm, how often is it during an attack? |
| Recall | TP ÷ (TP + FN) | What share of attack hours are under alarm? |
| F1 | 2 × precision × recall ÷ (precision + recall) | One number for both. It is low if either is low |
| True negative rate | TN ÷ (TN + FP) | What share of clean hours are quiet? |

**No point adjustment.** Some published results count a whole attack as detected, hour by hour, once any single hour inside it is flagged. That raises recall and F1 sharply. Here one alarm hour counts as one hour. Recall here is therefore lower than figures reported with point adjustment, and cannot be compared with them.

**Recall and attacks detected are different things.** A detector can catch all 7 attacks and still have low recall, if it alarms for only part of each one.

### Without a threshold

These two use the scores directly, on the attack and clean hours, and do not depend on the threshold.

| Metric | Meaning | Reference points |
|---|---|---|
| ROC-AUC | The chance that a randomly chosen attack hour has a higher score than a randomly chosen clean hour | 0.5 is random, 1.0 is perfect |
| PR-AUC | Average precision: the area under the precision-recall curve | A random score gives about the share of attack hours, roughly 0.12 on Dataset 2. 1.0 is perfect |

They measure how well the score separates attack hours from clean hours. They say nothing about whether the calibrated threshold is a good one. PR-AUC is the stricter of the two here, because attack hours are a small share of the data.

### BATADAL S-score

The ranking score of the original competition [1].

```
S     = 0.5 × S_TTD + 0.5 × S_CLF
S_TTD = 1 − mean over attacks of (delay ÷ attack length)
S_CLF = (recall + true negative rate) ÷ 2
```

- **S_TTD** rewards early detection. An attack caught in its first hour contributes 0 to the mean. A missed attack contributes 1, as if it were detected at its last hour.
- **S_CLF** is the average of the share of attack hours under alarm and the share of clean hours that are quiet.
- **Range:** 0 to 1, higher is better.
- **Caution:** a detector that alarms every hour scores 0.75 and one that never alarms scores 0.25, so S should always be read next to the false-alarm figures.

## 7. Output files

`calibrate_recon_model.py` writes:

| File | Contents |
|---|---|
| `calibration.pt` | References, thresholds and settings, for reuse |
| `calibration.json` | The thresholds and false-alarm figures, readable |
| `calibration_<score>.png` | Calibration and validation scores against the threshold |

`evaluate_recon_model.py` writes:

| File | Contents |
|---|---|
| `metrics.json` | Every metric, as `<score>_<metric>` |
| `per_attack.csv` | Detected or missed, and the delay, for each attack and score |
| `hourly_scores.csv` | Date, label, attack number, score and alarm for every scored hour |
| `sensor_errors.npz` | The per-sensor error table and references, for combining with other detectors |
| `timeline_<score>.png` | The score over Dataset 2 with the threshold, attack windows and alarms |
| `attack_<n>.png` | Close-up of each attack with the sensors that contributed most |
| `false_alarm_<n>.png` | Close-up of the three largest false alarms |
| `top_sensors_per_attack.png` | The 5 sensors with the highest error in each attack |
| `detection_delay.png` | Delay per attack for the three scores |

**Sensor contributions.** In the close-ups, each sensor is shown as its error divided by its 99th-percentile calibration error. A value of 1 is the edge of that sensor's normal range, and 10 is ten times beyond it.

## 8. Using the same scoring for another detector

The functions in `evaluate_recon_model.py` take plain arrays, not a model. Any detector that can produce one error per sensor per hour can use them.

What a detector has to supply:

| Array | Shape | Contents |
|---|---|---|
| Calibration errors | hours × sensors | Errors on the calibration block of Dataset 1 |
| Validation errors | hours × sensors | Errors on the validation block |
| Dataset 2 errors | hours × sensors | Errors on Dataset 2, with the date, label and attack number of each scored hour |

For a forecasting detector, the error would be the squared difference between the forecast and the real value at each hour.

```python
from evaluate_recon_model import calibrate, run_evaluation

calibration = calibrate(calib_errors, val_errors)
run_evaluation(calibration, dev_errors, dev_when, dev_labels, dev_attack_id,
               feature_names, out_dir="runs/other_detector/evaluation")
```

This gives the same thresholds, metrics, result files and plots as for the Transformer, so the tables can be put side by side.

To combine detectors, `sensor_errors.npz` from each run holds the error table and the calibration references. Scores can be combined from those files without running any model again.

Two conditions for a fair comparison:

- Every detector uses the same blocks of `data/processed/` and the same `dataset2_eval.csv`.
- Every detector uses the same budget, episode rule and grace hours.

## 9. Open decisions

1. **Episode rule and grace hours.** Both are 0 now. They change the false-alarm figures for every detector, so one setting is needed for all of them, fixed before the Test set is used.
2. **Which score is reported.** The proposal names `topk_percentile`. The other two are comparisons. The percentile score should be the headline figure for every detector.
3. **Number of top sensors.** The default is 3. It is a choice, and it should be the same for every detector or tuned on Dataset 2 only.
4. **Test set evaluation.** Neither script can score the Test set yet. That needs the test attack windows and an explicit unlock, and it should happen once, after items 1 to 3 are frozen.

## 10. Limits to keep in mind

1. **The close-up plots show the `topk_ratio` score.** The attack and false-alarm close-ups use that score and its alarms, not `topk_percentile`. They should be switched to the score that is reported (item 2 in section 9).
2. **The calibration block is short.** About 10 weeks of data set the threshold. A 1% budget is 17 hours, so a few unusual hours in that block move the threshold noticeably.
3. **The percentile score is coarse near the top.** Its values near 1 come in small fixed steps, and many attack hours can tie at exactly 1.
4. **The transfer check is optimistic.** The validation block also chose the checkpoint.
5. **Comparison with published results.** Published BATADAL scores often use thresholds chosen with knowledge of the attacks, or point adjustment. The figures here use neither, so they are expected to be lower, and the report should say why.

## 11. Reference

[1] R. Taormina et al., "The Battle of the Attack Detection Algorithms: Disclosing cyber attacks on water distribution networks," *Journal of Water Resources Planning and Management*, 144(8), 2018.
