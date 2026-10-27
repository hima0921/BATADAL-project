# Exploration: Dataset 1 and Dataset 2

## 1. Purpose

This is optional extra exploration, done after the data audit (`docs/audit.md`). The goal is to understand the data before the pipeline is built.

- Notebook: `notebooks/Explore.ipynb`. Every number below comes from its outputs.
- Data used: Dataset 1 (`Training dataset 1.csv`) and Dataset 2 (`Training dataset 2.csv`) only.
- The Test set is not used. It stays untouched until all choices and thresholds are frozen.
- This is exploratory data analysis, not feature engineering. Nothing in the data is changed, but the findings feed into the pipeline choices.

| Name | File (in `data/`) | Role |
|---|---|---|
| Dataset 1 | `Training dataset 1.csv` | Normal training data (no attacks) |
| Dataset 2 | `Training dataset 2.csv` | Development data (partly labeled) |
| Attack windows | `attack_windows_dataset2.csv` | The 7 real attack windows for Dataset 2 (saved by the notebook) |

## 2. Summary

| Step | Main result |
|---|---|
| 1. Drift within Dataset 1 | No drift. Every sensor's average shifts less than 0.1 SD, and every regular sensor's spread ratio is 0.94 to 1.01. No reason to change the 60/20/20 split. |
| 2. Rare pumps 6 and 11 | They run much more often in Dataset 2, and 93% (pump 6) and 97% (pump 11) of those hours are inside attack windows. Keep them. |
| 3. Near-duplicate sensors | 24 pairs above 0.95 correlation in block A: flow/status pairs, near-identical pressures, and pumps linked to pressures. Keep all 42 features for Ridge. |
| 4. First normal week | Tank 1 has a cycle of about 2 days, longer than the 12-hour and 24-hour windows. `P_J280` has a very small range, so rounding may matter. |
| 5. Attack 1 | Shows up mainly as rapid on/off switching of pumps 10 and 11, a change in pattern more than in averages. |

## 3. Step 0: Setup and attack windows

**What was done**
- Both training files are loaded from `../data/`. The spaces are stripped from the column names, and `DATETIME` is turned into real dates (format day/month/year hour).
- The `attacks` table holds the 7 real attack windows for Dataset 2 from the BATADAL site. Hours are counted including both the start and end hour.
- The table is saved as `data/attack_windows_dataset2.csv`.

**Attack windows (from the saved file)**

| Attack | Real start | Real end | Real hours |
|---|---|---|---|
| 1 | 2016-09-13 23:00 | 2016-09-16 00:00 | 50 |
| 2 | 2016-09-26 11:00 | 2016-09-27 10:00 | 24 |
| 3 | 2016-10-09 09:00 | 2016-10-11 20:00 | 60 |
| 4 | 2016-10-29 19:00 | 2016-11-02 16:00 | 94 |
| 5 | 2016-11-26 17:00 | 2016-11-29 04:00 | 60 |
| 6 | 2016-12-06 07:00 | 2016-12-10 04:00 | 94 |
| 7 | 2016-12-14 15:00 | 2016-12-19 04:00 | 110 |
| Total | | | 492 |

The file itself only labels 219 hours as attacks (found in the audit).

## 4. Step 1: Drift check within Dataset 1

**Why**

Drift means "normal" behavior changes over time. The detectors learn normal behavior from the first 60% of Dataset 1, and alarm thresholds are set on the last 20%. If normal behavior changed over the year, the models could raise false alarms and the thresholds could be set wrong. The 60/20/20 split is not fixed yet, so this check also shows whether that split is reasonable.

**What was done**

1a. **Three time blocks.** Dataset 1 is cut by row position at 60% and 80%. No gaps and no scaling. This split is only for exploration, not the final pipeline split. (0.6 and 0.8 are cut points, not block sizes. The blocks are 60%, 20%, and 20%.)

1b. **Average per block.** The average of each of the 43 sensor columns (31 continuous, 12 on/off) in each block. For on/off (`S_`) columns, the average is the share of hours the pump or valve is on.

1c. **Standardized shift.** For each sensor: (block average − block A average) ÷ block A standard deviation (SD). This is an effect size (Glass's delta: like Cohen's d, but divided by the SD of the reference group only). Block A's SD is used because block A is the training block. Dividing by the SD puts all sensors on the same scale, since they use different units. Constant columns give NaN because their SD is 0.
- Rough guide (own rule of thumb, similar to Cohen's small/medium): under 0.2 = no drift, 0.2 to 0.5 = some drift, over 0.5 = clear drift.

1d. **Columns that never change.** Checked with `nunique() == 1` in blocks A, B, C and in Dataset 2.

1e. **Spread.** Two blocks can have the same average, but one can swing much more. For each sensor: block SD ÷ block A SD. A ratio around 1 means the same swing.
- Rough guide (own rule of thumb): 0.8 to 1.25 = about the same.

**Results**

Time blocks:

| Block | Start | End | Rows | Season |
|---|---|---|---|---|
| A | 2014-01-06 00:00 | 2014-08-12 23:00 | 5,256 | Winter to summer |
| B | 2014-08-13 00:00 | 2014-10-24 23:00 | 1,752 | Late summer to fall |
| C | 2014-10-25 00:00 | 2015-01-06 00:00 | 1,753 | Late fall to early winter |
| Total | | | 8,761 | |

The blocks follow each other with no gaps and no overlap.

Averages (1b):
- Most sensors barely change across the blocks.
- The biggest movements are small: `L_T7` 3.29 → 3.31 → 3.35, `L_T1` 2.67 → 2.68 → 2.71, `F_PU2` 69.38 → 69.08 → 70.10.
- The pumps fall into three groups by share of hours on: pumps 3, 5, and 9 are never on. Pumps 6 and 11 are almost never on. All other pumps and the valve are on 42% of the time or more (pump 4 lowest at 42%, pump 1 at 100%).

Standardized shift (1c):
- Every sensor shifts by less than 0.1 SD.
- Largest: `L_T7`, 0.03 in block B and 0.08 in block C. This is a slow, small rise over the year.
- `L_T1` and `L_T6`: 0.03 in block C.
- NaN for 7 columns: `S_PU1`, `F_PU3`, `S_PU3`, `F_PU5`, `S_PU5`, `F_PU9`, `S_PU9`.

Columns that never change (1d):
- Constant everywhere in Datasets 1 and 2: `S_PU1`, `F_PU3`, `S_PU3`, `F_PU5`, `S_PU5`, `F_PU9`, `S_PU9`.
- `S_PU1` is 1 (pump 1 on) in all 8,761 hours of Dataset 1 and all 4,177 hours of Dataset 2. This matches `docs/audit.md`, which already lists `S_PU1` as never changing, so it is not a new finding.
- Pump 6 (`F_PU6`, `S_PU6`) is constant only in block B. Pump 11 (`F_PU11`, `S_PU11`) is constant in blocks B and C. Both pumps run in Dataset 2.
- There are two kinds of "doesn't change":
  1. The average doesn't change between blocks. This is good. The sensor still moves hour to hour.
  2. The value never changes at all. This gives the model nothing to learn from.

Spread (1e):
- Every regular sensor has a ratio between 0.94 and 1.01 in blocks B and C. The lowest is `L_T4`, 0.94 in block C.
- Only the rare pumps differ: the ratio is 0 where they don't run, and pump 6 is about 0.65 in block C, where it ran only 2 hours.

**What it means for the project**
- Within Dataset 1 there is no drift in averages or in spread.
- This gives no reason to change the 60/20/20 time split.
- A possible reason for the stability: BATADAL comes from a computer simulation of the C-Town network, not a real city.

**Cautions**
- The cutoffs (0.2 / 0.5 SD, 0.8 to 1.25) are rules of thumb, not formal tests.
- Each block covers a different season, so the check can't separate a season effect from drift. Here neither showed up.
- In the notebook, the spread check (1e) comes after step 2, not right after 1d.

## 5. Step 2: Rare pumps 6 and 11

**Why**

The drift tables showed that pumps 6 and 11 run in some blocks but not in others. This step checks if they are a false-alarm risk or an attack signal.

Why "rare": in Dataset 1 the pumps fall into three clear groups: never on (3, 5, 9), on less than 0.2% of hours (6 and 11), and on 42% or more (all others). Nothing in between.

**What was done**
- Counted the hours each pump is on in blocks A, B, C and in Dataset 2.
- Marked the Dataset 2 hours inside the 7 real attack windows (`in_attack`). The check printed 492 hours, which matches the attacks table.
- Counted the hours each pump is on inside and outside the attack windows.

**Results**

| Pump | Block A | Block B | Block C | Dataset 1 total | Dataset 2 total | Dataset 2 vs Dataset 1 |
|---|---|---|---|---|---|---|
| 6 | 14 | 0 | 2 | 16 of 8,761 (about 0.2%) | 87 of 4,177 (about 2%) | roughly 11 times more often |
| 11 | 4 | 0 | 0 | 4 of 8,761 (under 0.1%) | 32 of 4,177 (about 0.8%) | roughly 17 times more often |

- Pump 11 never runs in blocks B and C (the Validation and Calibration blocks).

Inside vs outside the attack windows (Dataset 2):

| Pump | On inside attacks | On outside attacks | Share inside | Share of outside hours on | Share of Dataset 1 hours on |
|---|---|---|---|---|---|
| 6 | 81 | 6 | 93% | about 0.16% | 0.18% |
| 11 | 31 | 1 | 97% | about 0.03% | 0.05% |

There are about 3,685 hours outside the attack windows (4,177 − 492).

**What it means for the project**
- The extra running in Dataset 2 comes from the attacks, not from normal operation changing between years.
- Keep pumps 6 and 11. A model that notices a rare pump turning on should catch some attacks.
- Small risk: the few normal hours when they run may cause false alarms. In Dataset 1 that is 16 hours for pump 6 and 4 hours for pump 11. In Dataset 2 (outside the attack windows) it is 6 and 1 hours.
- This is different from the 7 constant columns, which carry no information in training.

**Cautions**
- The hours outside the attack windows are −999 rows, so they are unlabeled, not confirmed normal. This step assumes they are normal.
- Any result for pumps 6 and 11 rests on very few "on" hours in block A (14 and 4), so it is less reliable.

## 6. Step 3: Near-duplicate sensors

**Why**

Ridge predicts each continuous sensor from the other 42 features. If a sensor has a near-duplicate, Ridge mostly copies it. That decides which attacks Ridge can and can't catch. The autoencoder may lean on the same pairs. Knowing the pairs also helps explain later why some attacks are caught and others are missed.

**What was done**
- Correlation (−1 to 1) between every pair of the 36 sensors that change in block A. The 7 constant columns are skipped, because correlation can't be calculated for a column that never moves.
- Block A only, since it is the training data. This gives a 36 × 36 table with 630 pairs.
- Test value: `F_PU2` vs `S_PU2` = 0.999.
- Cutoff: |correlation| above 0.95 (own choice; other cutoffs were not used, as they were not useful here).

**Results: 24 pairs, of three kinds**

1. A pump's flow and its on/off status:

| Pair | Correlation |
|---|---|
| `F_PU6` / `S_PU6` | 1.000 |
| `F_PU8` / `S_PU8` | 0.999 |
| `F_PU2` / `S_PU2` | 0.999 |
| `F_PU4` / `S_PU4` | 0.999 |
| `F_PU7` / `S_PU7` | 0.998 |
| `F_PU11` / `S_PU11` | 0.998 |
| `F_PU10` / `S_PU10` | 0.997 |
| `F_V2` / `S_V2` | 0.956 |

2. Pressure sensors that are basically the same reading:

| Pair | Correlation |
|---|---|
| `P_J300` / `P_J289` | 1.000 |
| `P_J302` / `P_J307` | 1.000 |
| `P_J289` / `P_J422` | 0.997 |
| `P_J300` / `P_J422` | 0.996 |

So `P_J300`, `P_J289`, and `P_J422` act as one group, and `P_J302` with `P_J307` as another.

3. Pumps with a pressure sensor:

| Pair | Correlation |
|---|---|
| `F_PU1` / `P_J269` | −1.000 |
| `F_PU2` / `P_J280` | −0.993 |
| `F_PU8` / `P_J306` | 0.991 |
| `S_PU8` / `P_J306` | 0.989 |
| `S_PU2` / `P_J280` | −0.986 |
| `S_PU2` / `P_J269` | 0.963 |
| `S_PU4` / `P_J256` | 0.959 |
| `S_PU7` / `P_J415` | 0.958 |
| `F_PU4` / `P_J256` | 0.956 |
| `F_PU2` / `P_J269` | 0.950 |
| `F_PU7` / `P_J415` | 0.950 |

Also `F_PU1` / `S_PU2` (−0.962): pump 1's flow drops when pump 2 turns on.

**What it means for the project**
- Ridge catches attacks that break a pair (for example, flow changes while the status says off), but it can miss attacks that change both sensors together.
- Plan: keep all 42 features for the main Ridge model, as in the proposal.

Reasons for not cutting the near-duplicates:
1. Ridge handles correlated inputs fine, so the math doesn't need it.
2. The pairs are useful. An attack that fakes one sensor but not its partner creates a big gap, which is often the easiest kind to catch.
3. The information is spread across several sensors (flow, status, nearby pressure), so cutting one wouldn't remove the duplicate anyway. For example, leaving out `S_PU2` when predicting `F_PU2` still leaves `P_J280`.
4. Keeping all features matches the proposal and keeps Ridge comparable to the autoencoder.

Instead, the pairs can be used to explain missed attacks. A leave-out version of Ridge could be an optional extra comparison. It is not required.

**Cautions**
- "Linked" means correlated, not a confirmed physical connection. This was not checked against the C-Town network map.
- Only pairs above 0.95 are listed. Weaker links exist, and Ridge uses them too.
- Correlation only catches straight-line relationships.
- Pairs with pumps 6 and 11 rest on very few hours.
- In the printed table in the notebook, the numbers on the left are the original row labels, not ranks. `.reset_index(drop=True)` would renumber them.

## 7. Step 4: Plot of the first normal week of Dataset 1

**Why**

To see how the sensors move hour by hour and to check the correlation results by eye. The GRU and Transformer use 12-hour and 24-hour windows, so it helps to see the daily pattern.

**What was done**

Plotted `L_T1`, `F_PU1`, `F_PU2`, and `P_J280` for the first 168 rows (7 × 24 hours) of Dataset 1, 2014-01-06 to 2014-01-12.

**Results**
- Pump 2 is on most of the time (about 95 to 100) and switches off for about 8 to 19 hours at a time, roughly every 2 days.
- `P_J280` is an exact mirror of pump 2: low when pump 2 is on, high when it is off (the −0.993 correlation).
- Pump 1's flow jumps from about 90–100 to about 113–121 whenever pump 2 is off, so pump 1 takes over the load.
- Tank 1 (`L_T1`) rises and falls in a cycle of about 2 days, not 24 hours. Pump 2 switches off near tank 1's peaks.

**What it means for the project**
- The 12-hour and 24-hour windows are shorter than tank 1's 2-day cycle, so a window sees only part of a cycle. This is something we should discuss.

**Cautions**
- This is only one week.
- Rounding (check pending): `P_J280` moves only between about 2.974 and 2.991 (a range of 0.017). Dataset 1 has up to 9 decimals, but Dataset 2 and the Test set have at most 2. So in those files `P_J280` can only be 2.97, 2.98, or 2.99. After scaling to 0–1, each rounding step is a big jump, which could cause false alarms. Other small-range sensors may have the same problem.

## 8. Step 5: Attack 1 (13 to 16 Sep 2016)

**Why**

To see what an attack looks like in the sensors, compared with normal behavior.

**What was done**

5a. Took the Dataset 2 rows inside attack 1's real window. For each changing sensor, compared the average during the attack with block A, in block A SDs (same method as the drift check), and listed the top 8.

5b. Plotted `F_PU10`, `F_PU11`, `P_J317`, and `P_J302` from 3 days before to 2 days after the attack, with the attack shaded. Pump 11 was included because it is at the top of the change list. Its rareness is the point, since it connects to the rare pump finding in step 2.

**Results**

5a. The window has 50 hours, which matches the attacks table. Top 8 changes:

| Sensor | Change (block A SDs) | Note |
|---|---|---|
| `F_PU11` | 22.98 | Huge partly because pump 11's normal SD is tiny |
| `S_PU11` | 21.00 | Same reason |
| `F_PU10` | −0.90 | Pump 10 runs less |
| `P_J317` | 0.76 | Rises |
| `S_PU10` | −0.60 | Pump 10 runs less |
| `P_J307` | −0.29 | Drops together with `P_J302` (near-duplicate pair) |
| `P_J302` | −0.28 | Drops together with `P_J307` |
| `L_T1` | 0.19 | Barely moves |

All of these are far bigger than the drift shifts (under 0.1).

5b. The plot:
- Pump 11 is off for days before the attack, switches on at the attack start, and switches off right after it ends.
- During the attack, pumps 10 and 11 switch on and off together many times, each at a lower flow (about 25 instead of pump 10's usual 31). Near the end, both stay off for several hours.
- `P_J317` swings much wider than normal: about 53 to 92 during the attack, compared with about 55 to 78 before it.
- `P_J302` becomes more jagged during the attack and dips to about 14, but otherwise stays in a similar range.

**What it means for the project**
- The attack mainly shows up as rapid on/off switching, which is a change in pattern over time.
- Averages understate this. `P_J317` shifted only 0.76 SD on average but swings much more.
- Detectors that look at recent history (GRU, Transformer) should suit this kind of attack.
- The published attack window lines up well with the data, which supports using the windows to rebuild labels.

**Cautions**
- The pump 11 numbers in 5a exaggerate its ranking, because its normal SD is tiny. The plot shows real Dataset 2 values, so this caution affects the shift number but not the plot.
- This compares 2016 attack hours with 2014 normal data. Dataset 1 vs Dataset 2 has not been compared yet.
- Averages can hide attacks that change a sensor's pattern without changing its average.
- This is one attack out of 7.

## 9. Decisions we need to make

1. **Drop or keep the 7 never-changing columns** (`S_PU1`, and the flow and status of pumps 3, 5, and 9). Dropping is simpler. Keeping them lets a model notice a sudden change, but may cause false alarms. Pump 3 does run in the Test set (from the audit).
2. **Pump 3 and scaling.** If pump 3 is kept, its scaled value is 0 in training but goes above 1 in the Test set, because the scaler is fit on training data only.
3. **How to treat the −999 rows in Dataset 2.** These rows are unlabeled, not normal. Dataset 2 has no rows labeled 0.
4. **The split.** The split can still change, and any change should be based on Dataset 1 only. The drift results support keeping 60/20/20.
5. **Rounding for small-range sensors** like `P_J280` (2 decimals in Dataset 2 and the Test set).
6. **Tank 1's 2-day cycle** vs the 12-hour and 24-hour windows.

## 10. Open items / to do later

- **Compare Dataset 1 (block A) with the normal hours of Dataset 2** (same average and spread checks). This waits for the −999 decision, because that decides which Dataset 2 hours count as normal. It doesn't affect training (detectors train on Dataset 1 only), but it would help predict false alarms on Dataset 2.
- **Check which sensors have ranges small enough** for 2-decimal rounding to matter.
- **`S_PU1` in `docs/audit.md`.** `docs/audit.md` already lists `S_PU1` as never changing (section 3, and section 4 item 1), so no change is needed there.

## 11. General cautions

- All cutoffs (0.2 / 0.5 SD for drift, 0.8 to 1.25 for spread, 0.95 for correlation) are rules of thumb, not formal tests.
- Results involving pumps 6 and 11 rest on very few hours.
- "Linked" sensors are correlated, not confirmed to be physically connected.
- Correlation only catches straight-line relationships.
- Hours outside the attack windows in Dataset 2 are assumed normal, but they are unlabeled (−999).
- BATADAL is a simulation, which may explain why normal behavior is so stable.
- Dataset 1 has up to 9 decimals. Dataset 2 and the Test set have at most 2.
