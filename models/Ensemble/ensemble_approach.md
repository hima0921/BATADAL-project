# Isolation Forest and the ensemble

Detectors: Transformer reconstruction (`models/Transformer`), XGBoost (`models/XGBoost`), Isolation Forest + PCA (`isolation_forest.py`). `ensemble.py` fuses their per-hour scores.

```bash
python ../XGBoost/train_xgboost.py --dataset-path ../../data/processed --out-dir runs/xgb
python isolation_forest.py --dataset-path ../../data/processed --out-dir runs/iforest
# once per trained Transformer checkpoint (run in the Transformer environment):
python export_transformer_scores.py --dataset-path ../../data/processed \
    --model-checkpoint ../Transformer/runs/recon_24h/best_model.pt --out transformer_hourly_scores.csv
python ensemble.py --xgb-dir runs/xgb --iforest-dir runs/iforest \
    --transformer-scores transformer_hourly_scores.csv --out-dir runs/ensemble
```

## Rules that keep Dataset 2 out of every choice
- The Isolation Forest is fitted on the Dataset 1 train block only (the paper fits it on Dataset 2's own `-999` rows).
- Scores are put on a common scale with the calibration block only (`percentile`, or `ratio` = log(1 + score / calibration 99th percentile)).
- Fusion is a fixed rule, not fitted weights: `mean`, `paper` (0.4 / 0.1 / 0.5, renormalised over the detectors present) or `max`. `max` of calibration percentiles is the fusion the project proposal already specifies.
- The alarm threshold is the lowest one inside the false-alarm budget on the calibration block.
- Evaluation is as in `models/XGBoost`: positives = `ATT_FLAG = 1` hours, `-999` dropped, false alarms on the validation block, XGBoost scored out-of-fold.

## First result: XGBoost + Isolation Forest (3 seeds, mean; Transformer not yet included)

| Score | Attacks detected | Recall | Precision | F1 | ROC-AUC | PR-AUC | S-score |
|---|---|---|---|---|---|---|---|
| XGBoost | 4 | 0.25 | 0.89 | 0.39 | 0.70 | 0.49 | 0.54 |
| Isolation Forest | 6 | 0.48 | 0.90 | 0.63 | 0.80 | 0.62 | 0.74 |
| mean of percentiles | 7 | 0.10 | 0.69 | 0.18 | 0.81 | 0.39 | 0.68 |
| paper weights | 6 | 0.12 | 0.69 | 0.20 | 0.74 | 0.40 | 0.60 |
| **max of percentiles** | **7** | **0.69** | **0.96** | **0.80** | **0.95** | **0.87** | **0.80** |

The two detectors catch different hours (XGBoost: attack 4; Isolation Forest: most of the rest), so a rule that alarms when either is extreme helps and averaging hurts: a mean dilutes a one-detector spike below the threshold. Six fusion variants were run; the one that works is the proposal's own rule, but treat the 0.80 as optimistic until it is confirmed on the Test set.

## Checks done
- Era confound (Dataset 1 is 2014, Dataset 2 is 2016): Isolation Forest AUC for the labeled attack hours is 0.80 against 2014 validation hours and 0.80 against 2016 clean hours; 2016 clean hours score like 2014 ones. No inflation from the era for this detector.
- XGBoost clean hours are scored by one held-out-attack model each (round-robin), the same way attack hours are, so averaging does not make them look calmer. Averaging them instead changed the results by less than 0.01.

## Not yet done
- Transformer scores: the exporter is written but untested (no checkpoint on this machine); the ensemble with all three detectors needs it.
- Precision depends on the 219 : 1,705 positive-to-negative mix, which is arbitrary.
