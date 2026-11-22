# XGBoost detector

Supervised detector after Nemade et al. (2026), "Cyber-Physical Attack Detection in Water Distribution Systems Using a Hybrid Ensemble of XGBoost, Isolation Forest, and LSTM Autoencoder on the BATADAL Dataset" (JIDMIS 3(3s)). It is the supervised member of the planned three-model ensemble (Transformer reconstruction + XGBoost + Isolation Forest).

```bash
cd models/XGBoost
python train_xgboost.py --dataset-path ../../data/processed --out-dir runs/xgb --seeds 0 1 2
```

## Labels

| Data | Use |
|---|---|
| Dataset 1 (all normal) | negatives: train block trains, calibration block sets the threshold, validation block measures false alarms |
| Dataset 2, `ATT_FLAG = 1` | positives (219 hours) |
| Dataset 2, `ATT_FLAG = -999` | **dropped**: never a training or hour-level evaluation label |

The 219 labeled hours cover only 5 of the 7 real attacks (1, 3, 4, 5, 6). Attacks 2 and 7 have no labeled hour, so no model trains on them: they are a true hold-out. The `-999` hours inside a real attack window are scored only to measure detection delay.

## Method (what follows the paper, what does not)

Follows the paper: 400 trees, depth 6, learning rate 0.05, 0.8 row and column subsampling, `scale_pos_weight` = negatives / positives; lag 1-3, first differences at 1 and 3, rolling mean/std/deviation (3, 6, 24 h) for tank levels and pressures, cyclical hour and weekday. This gives 391 features from the 36 non-constant sensors (the paper reports 376 from 43).

Differs from the paper, on purpose:

- **No in-sample scoring.** The paper trains on 80% of a 1,314-row sample and then reports on all 4,177 hours, with the threshold picked to maximise F1 on the same series. Here, the model that scores an attack never trained on it (leave-one-attack-out), the threshold comes from the calibration block under the project's false-alarm budget (at most 1 episode per week, at most 1% of hours), and false alarms are counted on the validation block.
- **Trailing windows only**, so the score for hour t uses hours up to t (the paper's 3-hour smoothing is centred).
- **Features are built inside each contiguous block**, never across the 24-hour gaps, and the first 23 hours of each block are dropped. The scored hours are exactly those the 24-hour Transformer scores, so scores line up for the ensemble.
- **No negative subsampling**: all of the Dataset 1 train block is used, balanced with `scale_pos_weight`.
- Trees do not need scaling; the processed files are already Training-scaled.

## First results (3 seeds, mean)

| | with time features | `--no-time-features` |
|---|---|---|
| Attacks detected (of 7) | 4.0 | 5.0 |
| Unseen attacks 2, 7 detected (of 2) | 1.0 | 1.3 |
| Mean delay, detected only (h) | 10.9 | 15.1 |
| Recall on labeled hours | 0.275 | 0.277 |
| Precision (219 positives vs 1,705 validation hours) | 0.85 | 0.81 |
| True-negative rate (validation) | 0.994 | 0.991 |
| ROC-AUC / PR-AUC | 0.67 / 0.47 | 0.66 / 0.46 |
| S-score | 0.55 | 0.60 |

Read these with care: the precision depends on the arbitrary 219:1,705 mix, only 7 attacks exist, and seeds disagree on which attacks are caught (attack 1 and 6 are usually missed).
The paper's F1 of 0.93 is an in-sample figure and is not comparable to these.

## Known limits

- With `-999` dropped, all negatives come from 2014 (Dataset 1) and all positives from 2016. A classifier could partly learn the era instead of the attack. The weak out-of-attack scores suggest this is not what is being learned, but it has not been tested directly.
- Labeled hours are the middle of some attacks (for example 7 of the 60 hours of attack 5), so the "start" of an attack is not in the positives.
- Not yet done: Isolation Forest (+PCA), the ensemble, and any use of the Test set.

## Why the paper's numbers are higher (`paper_protocol_check.py`)

Re-running the paper's recipe on Dataset 2 (1,095 random normal rows + 219 attack rows, chronological 80/20 of that sample, 400 trees, threshold maximising F1 on the full series) reproduces its XGBoost row (paper: P 0.958, R 0.626, F1 0.757, AUC 0.814; here P 0.98-1.00, R 0.635, F1 0.77, AUC 0.84-0.87). The same model scored only on rows it never trained on: recall 0, AUC 0.55-0.65. The paper's gain is in-sample scoring on Dataset 2, plus a threshold tuned on the evaluated series, plus a centred 3-hour smoothing of the ensemble score. Its Isolation Forest + PCA row also reproduces (AUC 0.79 vs 0.80) and, being unsupervised, scores the same when trained on Dataset 1 only (AUC 0.80), so that recipe carries over to this project.
