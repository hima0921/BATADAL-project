# Label-sensitivity experiment

## 1. Question and files

How much do the results change depending on how we treat the -999 rows in Dataset 2? Some work treats -999 as normal (for example a 2026 hybrid XGBoost / Isolation Forest / LSTM paper on BATADAL). But our audit (`docs/audit.md`) found about 273 real attack hours hidden in the -999 rows, including all of attacks 2 and 7.

Ridge never learns from labels, so the label rule can't change it. To test this, we need a model that learns from labels: a classifier trained on Dataset 2.

All steps are in `notebooks/Label_Experiment.ipynb`. Every number below comes from the output of that notebook, except the Ridge numbers, which come from `docs/ridge.md`.

- Data used: `data/processed/dataset2_eval.csv` (Dataset 2 with `ATT_FLAG`, and our `label` and `attack_id` from the 7 attack windows). 36 sensor features.
- Dataset 1 and the Test set are not used.

## 2. Summary

| Step | Result |
|---|---|
| 1. Label rules and blocks | 3 rules that only differ in the 273 hidden hours. 7 blocks, one per attack |
| 2. Logistic regression | Weak (ROC-AUC 0.524 / 0.535 / 0.599), but rule 1 < rule 2 < rule 3 |
| 3. Random forest, 1 seed | ROC-AUC 0.700 / 0.704 / 0.757 |
| 4. Random forest, 3 seeds | ROC-AUC 0.696 / 0.705 / 0.756, PR-AUC 0.269 / 0.272 / 0.452. Std 0.01 or less |
| 5. Save | 3 files in `models/Label_Experiment/` |

## 3. Design

**The three label rules** (training labels only)

| Rule | What it does | Attack hours | Normal hours | Left out |
|---|---|---|---|---|
| 1. -999 as normal | `ATT_FLAG = 1` is attack, everything else is normal | 219 | 3,958 | 0 |
| 2. Confirmed only | `ATT_FLAG = 1` is attack, hours outside the 7 windows are normal | 219 | 3,685 | 273 |
| 3. Rebuilt from the 7 windows | every hour inside a window is attack (our `label`) | 492 | 3,685 | 0 |

- The rules only differ in what they do with the 273 hidden attack hours: normal (rule 1), left out (rule 2) or attack (rule 3).
- Rule 2 uses the Dataset 2 hours outside the windows as normal, not Dataset 1, because Dataset 1 is a different simulation year (2014 vs 2016). This is the second option in `docs/audit.md`, open item 2.

**The 7 blocks**

Dataset 2 is cut into 7 blocks, one per attack. Each cut is halfway between the end of one attack and the start of the next.

| Block | Start | End | Hours | Attack hours | Attack |
|---|---|---|---|---|---|
| 1 | 2016-07-04 00:00 | 2016-09-21 05:00 | 1,902 | 50 | 1 |
| 2 | 2016-09-21 06:00 | 2016-10-03 09:00 | 292 | 24 | 2 |
| 3 | 2016-10-03 10:00 | 2016-10-20 19:00 | 418 | 60 | 3 |
| 4 | 2016-10-20 20:00 | 2016-11-14 16:00 | 597 | 94 | 4 |
| 5 | 2016-11-14 17:00 | 2016-12-02 17:00 | 433 | 60 | 5 |
| 6 | 2016-12-02 18:00 | 2016-12-12 09:00 | 232 | 94 | 6 |
| 7 | 2016-12-12 10:00 | 2016-12-25 00:00 | 303 | 110 | 7 |

The blocks are very uneven. Block 1 is 1,902 hours with 50 attack hours. Block 6 is 232 hours with 94.

**Leave one attack out**
- 7 rounds. In each round we train on 6 blocks and test on the 7th, so the model is always tested on an attack it never trained on.
- The 24 hours on each side of the test block are left out of training.
- Hours with no label (rule 2's hidden hours) are left out of training.

**Same answer key**
- The test block is always graded with the rule 3 labels. So only the training labels change, and all three rules are graded the same way.
- ROC-AUC and PR-AUC are computed once over all 4,177 hours, not per block, because the blocks are so uneven.

**Models**
- Logistic regression, `class_weight="balanced"`.
- Random forest: 300 trees, `min_samples_leaf=5`, `class_weight="balanced"`, seeds 0, 1 and 2.

## 4. Results

**Overall**

| Model | Rule 1 | Rule 2 | Rule 3 |
|---|---|---|---|
| Logistic regression, ROC-AUC | 0.524 | 0.535 | 0.599 |
| Logistic regression, PR-AUC | 0.247 | 0.255 | 0.336 |
| Random forest, ROC-AUC (mean ± std, 3 seeds) | 0.696 ± 0.004 | 0.705 ± 0.001 | 0.756 ± 0.003 |
| Random forest, PR-AUC (mean ± std, 3 seeds) | 0.269 ± 0.005 | 0.272 ± 0.002 | 0.452 ± 0.010 |

- Logistic regression is weak, but the order is rule 1 < rule 2 < rule 3. At the 0.5 cutoff it flags 10% to 15% of normal hours, so its per-attack shares only show the pattern.
- The random forest is stronger and flags about 1% of normal hours at 0.5. The std is 0.01 or less, so the result doesn't depend on the seed.

**Per attack: attacks 1 and 6** (share of attack hours above 0.5)

| Attack | Model | Rule 1 | Rule 2 | Rule 3 |
|---|---|---|---|---|
| 1 | Random forest (3 seeds, smallest to biggest) | 0.00 | 0.00 | 0.38 to 0.42 |
| 1 | Logistic regression | 0.12 | 0.12 | 0.70 |
| 6 | Random forest (3 seeds, smallest to biggest) | 0.00 to 0.01 | 0.00 | 0.68 to 0.70 |
| 6 | Logistic regression | 0.14 | 0.14 | 0.27 |

The other attacks change much less. With the random forest under rule 3, attack 3 is at 0.10 to 0.12, attack 4 at 0.04 to 0.10, attack 5 at 0.27 to 0.28 and attack 7 at 0.02 to 0.04.

## 5. Main findings

1. **The label rule matters most when the unlabeled hours are the only examples of a kind of attack.**
   - Attacks 1 and 2 are similar (both are about tank 7), and attack 2 has no labels. So when we test on attack 1, rules 1 and 2 have no tank 7 attack to learn from. Attack 1 goes from 0.00 to 0.38 to 0.42.
   - Attacks 6 and 7 are similar (both slow down pump 7), and attack 7 has no labels. Attack 6 goes from 0.00 to 0.01 to 0.68 to 0.70. (Attack 5 also slows pump 7, but most of its hours are unlabeled too.)
   - Attacks 3, 4 and 5 each have a similar attack that is at least partly labeled under every rule (attack 3 has attack 4, attack 4 has attack 3, and attack 5 has attack 6), so the classifier gets some examples of that kind of attack either way. They change much less: attack 3 goes from 0.02 to about 0.11 and attack 4 from about 0.04 to 0.06, both still low, and attack 5 stays about 0.25 to 0.27.
2. **Rules 1 and 2 give almost the same results** (random forest ROC-AUC 0.696 and 0.705, PR-AUC 0.269 and 0.272). So the loss comes from the hidden hours missing as attack examples, not from calling them normal.
3. **PR-AUC shows the difference much more than ROC-AUC:** 0.269 to 0.452, compared with 0.696 to 0.756. PR-AUC looks at the top of the ranking, and attacks are rare.
4. **Even the best classifier is below Ridge.** The random forest with rule 3 gets ROC-AUC 0.756. Ridge gets 0.864 (`docs/ridge.md`, section 11) and never saw a label. With only 7 attacks, learning normal behavior works better than learning from labeled attacks.
5. **Attacks 3 and 7 stay hard** under every rule and seed. Ridge also only catches about 17% and 24% of their hours (10 of 60 and 26 of 110, `docs/ridge.md`, section 7).

## 6. Cautions

- These are simple classifiers, and there are only 7 attacks.
- The 0.5 cutoff is not a calibrated threshold. The per-attack shares only compare the rules with each other. They are not detection results.
- Rules 2 and 3 assume every hour outside the 7 windows is normal. This depends on the list of attacks being complete.
- Ridge's ROC-AUC comes from a different setup (trained on Dataset 1, no leave one attack out), so the comparison with Ridge is rough.

## 7. Saved files

Folder: `models/Label_Experiment/`

| File | What it is |
|---|---|
| `label_experiment_scores.csv` | One row per Dataset 2 hour (4,177 rows, 14 columns): `DATETIME`, `ATT_FLAG`, `label`, `attack_id`, `block`, the training labels of each rule (`train_label_rule1` to `3`), the logistic regression scores (`lr_score_rule1` to `3`) and the random forest scores with seed 0 (`rf_seed0_score_rule1` to `3`) |
| `label_experiment_summary.csv` | ROC-AUC and PR-AUC for each model and rule (6 rows). Mean and std over 3 seeds for the random forest. The logistic regression has 1 run, so its std is empty |
| `label_experiment_per_attack.csv` | Share of hours above 0.5 for each model, rule and attack, including 0 for outside the attacks (48 rows), with the mean, smallest and biggest over the seeds |

## 8. What it means for the project

- **Dataset 2's labels are risky to train on**, because the hidden attack hours are the only examples of some kinds of attack (like attacks 2 and 7). A model trained on `ATT_FLAG = 1` never sees them as attacks. How we treat the -999 rows changes a trained classifier's results a lot, mostly through the attacks that have no labels.
- **It supports training our detectors on Dataset 1 only**, which is all normal, so no attack labels are needed. Ridge, which does this, beats every classifier here.
- **It supports scoring with the rebuilt labels (rule 3).** Under rule 1, the 273 hidden attack hours would count as normal, and correct alarms on them would count as false alarms.
