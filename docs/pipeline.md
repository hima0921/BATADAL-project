# Data pipeline: split and scaling

## 1. Purpose and files

This is the shared data preparation for the project (Milestone 1 in the proposal). It rounds the data to 2 decimals, drops the 7 columns that never change, splits Dataset 1 by time, scales the continuous columns, and saves the results so everyone's models use the same data.

All steps are in `notebooks/Pipeline.ipynb`. Every number below comes from the output of that notebook. It comes after the data audit (`docs/audit.md`) and the exploration (`docs/exploration.md`).

- Data used: Dataset 1 (`Training dataset 1.csv`) and Dataset 2 (`Training dataset 2.csv`).
- The Test set is not used. It stays untouched until all choices and thresholds are frozen.
- Nothing in the raw data files is changed. The results are saved as new files in `data/processed/`.

## 2. Summary

| Step | Result |
|---|---|
| 0b. Round and drop | All data rounded to 2 decimals. 7 never-changing columns dropped. Dataset 1 is 8,761 × 38, Dataset 2 is 4,177 × 38 |
| 1. Separate the columns | 28 continuous, 8 on/off |
| 2. Split Dataset 1 by time | Training 5,256 rows, Validation 1,728, Calibration 1,729, with two 24-hour gaps |
| 3. Fit the scaler | MinMaxScaler fitted on the 28 continuous Training columns only |
| 4. Scale the other data | Validation, Calibration and Dataset 2 scaled with the same scaler |
| 5. On/off columns | Not scaled. Still only 0 and 1 in all four tables |
| 6. Dataset 2 labels | Not done yet (see section 11) |
| 7. Save the data | 4 scaled tables and the scaler saved in `data/processed/` |

## 3. Step 0b: Round to 2 decimals and drop the 7 never-changing columns

**What we did**
- Rounded every column except `DATETIME` to 2 decimals, in Dataset 1 and Dataset 2.
- Dropped `S_PU1`, `F_PU3`, `S_PU3`, `F_PU5`, `S_PU5`, `F_PU9` and `S_PU9` from both.
- Did both before the split and the scaling, so every later step uses the rounded data without these columns.

**Why**
- Rounding: Dataset 1 has up to 9 decimals, but Dataset 2 and the Test set only have 2 (`docs/audit.md`). Rounding makes Training look like the data the models will be scored on.
- Dropping: these 7 columns never change in Dataset 1 or Dataset 2 (`docs/audit.md`). A column that never changes gives the models nothing to learn from.

**Result**
- Dataset 1: 8,761 rows × 38 columns. Dataset 2: 4,177 rows × 38 columns. That is `DATETIME`, 36 features and `ATT_FLAG`.
- After rounding, `P_J280` has only 3 different values in all of Dataset 1.

**Caution**
- Pump 3 runs in the Test set, and `S_PU1` is 0 in a few Test rows (`docs/audit.md`). Since we drop these columns, no model will see that.

## 4. Step 1: Separate continuous and on/off columns

- `cont_cols`: the 28 columns that start with `L_` (tank levels), `F_` (flows) or `P_` (pressures).
- `onoff_cols`: the 8 columns that start with `S_` (pump and valve status).
- The only columns left over are `DATETIME` and `ATT_FLAG`, so no sensor column was missed.
- The lists are built from the column names, so the dropped columns are not in them: `F_PU3`, `F_PU5` and `F_PU9` are gone from `cont_cols`, and `S_PU1`, `S_PU3`, `S_PU5` and `S_PU9` are gone from `onoff_cols`.
- The same lists work for Dataset 2 and the Test set, because the audit showed all three files have the same column names (after removing the spaces in Dataset 2).

## 5. Step 2: Split Dataset 1 by time (60 / 20 / 20)

**What we did**
- Cut Dataset 1 by row position at 60% and 80% (row 5,256 and row 7,008), in time order. No shuffling.
- Left a 24-hour gap (24 rows) before Validation and before Calibration. These rows are not used in any block.
- Each gap is taken from the start of the next block, so Training keeps its full size.

**Why**
- Time order: neighbouring hours look almost the same. A random split would put near-copies of Training hours into Validation and Calibration, and the models would look better than they are.
- The gaps: the GRU and Transformer look at the last 12 or 24 hours. Without a gap, a window at the start of a block would reach back into the block before it.
- The exploration found no drift in Dataset 1, so there was no reason to change 60/20/20.

**Result**

| Block | Rows | First hour | Last hour | Used for |
|---|---|---|---|---|
| Training | 5,256 | 2014-01-06 00:00 | 2014-08-12 23:00 | Learning normal behavior |
| Gap | 24 | 2014-08-13 00:00 | 2014-08-13 23:00 | Not used |
| Validation | 1,728 | 2014-08-14 00:00 | 2014-10-24 23:00 | Tuning the models |
| Gap | 24 | 2014-10-25 00:00 | 2014-10-25 23:00 | Not used |
| Calibration | 1,729 | 2014-10-26 00:00 | 2015-01-06 00:00 | Setting alarm thresholds |
| Total used | 8,713 of 8,761 | | | |

Each block covers a different season. The exploration checked these same periods and found no drift.

## 6. Step 3: Fit the MinMaxScaler on Training only

**What we did**
- Fitted `MinMaxScaler` (scikit-learn) on the 28 continuous columns of the Training block only. It saves each column's min and max from Training.
- Each value becomes (value − Training min) ÷ (Training max − Training min).

**Why Training only**
- If the scaler saw Validation, Calibration or Dataset 2, it would learn their min and max. That leaks information from data the models should not see while learning.

**Result**
- All 28 columns go from 0 to 1 in Training. Since the never-changing columns were dropped in step 0b, every column has a range to scale.
- `F_PU6` and `F_PU11` reach 1, but their max comes from very few hours (14 and 4 hours when the pump was on).

**Caution about `P_J280`**
- `P_J280` has only 3 different values after rounding, so in Training its scaled value can only be 0, 0.5 or 1.
- One 0.01 change in the raw reading is a jump of 0.5 after scaling. So small changes in `P_J280` will look big to the models.

## 7. Step 4: Scale the other data with the same scaler

**What we did**
- Used the same fitted scaler (transform only, no new fit) on Validation, Calibration and all of Dataset 2.
- The Test set is not scaled yet. Only the scaler is saved, so the Test set can be scaled with the exact same rule at the end.

**Result**

| Table | Scaled range (all continuous columns) | Notes |
|---|---|---|
| Validation | about −0.015 to 1.02 | Stays close to 0–1, as expected with no drift |
| Calibration | about −0.003 to 1.03 | Same |
| Dataset 2 | about −1.28 to 2.86 | Goes much further out (table below) |

- In Validation and Calibration, `P_J415` (max 0.72 in Validation) and `P_J317` (max about 0.65 in Validation and 0.66 in Calibration) never get close to 1. Training had a rare high peak for these two sensors, and that peak set the max. This is not an error.
- Pump 6 is 0 in Validation and reaches 0.97 in Calibration. Pump 11 is 0 in both. This matches the exploration.
- `P_J280` goes from exactly 0 to 1 in both Validation and Calibration, because of its 3 rounded values.

Dataset 2 columns that go furthest outside 0–1:

| Column | Scaled min | Scaled max | Possible reason |
|---|---|---|---|
| `F_PU1` | −1.277 | 0.996 | Pump 1 flow drops far below Training (raw min 38.92) |
| `P_J14` | 0.010 | 2.860 | Pressure spike |
| `P_J302` | −0.042 | 2.662 | Pressure spike (moves with `P_J307`, a near-duplicate pair) |
| `P_J307` | −0.042 | 2.642 | Same as `P_J302` |
| `P_J269` | 0.004 | 2.060 | Pressure spike |
| `P_J280` | 0.000 | 1.500 | Only one 0.01 step above the Training max (3 rounded values, see step 3) |
| `F_PU6` | 0.000 | 1.360 | Rare pump runs at a higher flow than in Training |
| `P_J306` | −0.009 | 1.235 | |
| `L_T1` | 0.000 | 1.200 | Tank 1 higher than in Training (attacks 3 and 4 overflow tank 1) |
| `F_PU8` | 0.000 | 1.169 | |
| `L_T7` | −0.157 | 1.000 | Tank 7 lower than in Training (attacks 1 and 2 cause low levels in tank 7) |
| `F_PU11` | 0.000 | 1.124 | Rare pump runs at a higher flow than in Training |
| `F_PU7` | −0.017 | 1.002 | The one negative flow (−0.88) from the audit |

These are only the min and max. We have not checked yet whether they fall inside the attack windows, so the reasons are guesses.

Before rounding, `P_J280` in Dataset 2 went from −0.197 to 1.492. Rounding removed the values below 0, but now each step is big (step 3).

## 8. Step 5: On/off columns stay as 0 and 1

- The 8 `S_` columns are not scaled, because they are already 0 and 1.
- Only `cont_cols` were replaced in steps 3 and 4, so the `S_` columns were never changed.
- A check confirmed they are only 0 and 1 in Training, Validation, Calibration and Dataset 2. (Dataset 2 stores them as `1.00` and `0.00`, which still count as 1 and 0.)

## 9. Step 7: Saved files

Folder: `data/processed/`

| File | What it is | Rows |
|---|---|---|
| `train_scaled.csv` | Training block, scaled | 5,256 |
| `val_scaled.csv` | Validation block, scaled | 1,728 |
| `cal_scaled.csv` | Calibration block, scaled | 1,729 |
| `dataset2_scaled.csv` | All of Dataset 2, scaled | 4,177 |
| `scaler.joblib` | The fitted scaler (Training min and max of the 28 continuous columns), for scaling the Test set later | |

- Each CSV has 38 columns: `DATETIME`, the 36 features (28 continuous and 8 on/off), and `ATT_FLAG` (the raw label from the file).
- The 7 never-changing columns are not in the files.
- Dates are saved as `2014-01-06 00:00:00`.

**How to load**

```python
import pandas as pd
import joblib

train = pd.read_csv("../data/processed/train_scaled.csv", parse_dates=["DATETIME"])
val = pd.read_csv("../data/processed/val_scaled.csv", parse_dates=["DATETIME"])
cal = pd.read_csv("../data/processed/cal_scaled.csv", parse_dates=["DATETIME"])
d2 = pd.read_csv("../data/processed/dataset2_scaled.csv", parse_dates=["DATETIME"])

scaler = joblib.load("../data/processed/scaler.joblib")  # only for the Test set, at the end
```

The paths assume the code runs from the `notebooks` folder.

When the Test set is scaled at the end, it must first be rounded to 2 decimals and have the same 7 columns dropped, because the scaler expects the same 28 continuous columns.

## 10. Decisions made

1. **Round all data to 2 decimals (step 0b).** Dataset 2 and the Test set only have 2 decimals, so Training should look the same. Side effect: `P_J280` has only 3 values, so its scaled steps are 0.5 (step 3).
2. **Drop the 7 never-changing columns (step 0b).** `S_PU1`, `F_PU3`, `S_PU3`, `F_PU5`, `S_PU5`, `F_PU9` and `S_PU9` never change in Dataset 1 or Dataset 2, so models can't learn from them. Side effect: no model sees pump 3 running or `S_PU1` at 0 in the Test set.
3. **Keep the 0–1 range, and keep the `S_` columns as 0 and 1.**

## 11. Still open

1. **Dataset 2 labels (step 6).** Not done yet. The plan is to save them in a separate file (for example `data/processed/dataset2_labels.csv`), so the scaled files above do not change. How to treat the −999 rows is still for us to decide. One option is to save all three label rules from the proposal as separate columns, since the label-sensitivity experiment compares them.
2. **Range: 0–1 vs −1 to 1.** Our files use 0 to 1. The Transformer training script scales to −1 to 1 (`feature_range = (-1.0, 1.0)` in `models/Transformer/train_batadal_recon_model.py`), because its model ends in `tanh`. With that range, its loader also turns the `S_` columns into −1 and 1. We need to agree on one range.
3. **Windows for the GRU and Transformer.** The Transformer training script already has a 12/24-hour window config. We should check the GRU uses the same window code.
4. **One shared source of data.** The data loader in the repo (`models/Transformer/batadal_dataset.py`) reads the raw CSVs and does its own split and scaling. It should use these saved files, or the same rounding, drop, split and scaling, so there is only one version of the data.
