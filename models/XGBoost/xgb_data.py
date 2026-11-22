"""
Data and feature engineering for the XGBoost detector.

Reads the shared files written by notebooks/Pipeline.ipynb (data/processed/),
so the XGBoost detector sees the same blocks, the same 36 features and the same
scaling as the Transformer. Trees do not need scaling, but nothing is changed
here: the values are already rounded, scaled with the Training-only scaler and
free of the 7 never-changing columns.

Label policy (project decision):
  Dataset 1  every hour is normal                       -> negatives
  Dataset 2  ATT_FLAG == 1 is a confirmed attack hour   -> positives
             ATT_FLAG == -999 is unlabeled              -> DROPPED, never a
             training or hour-level evaluation label. (The rows that fall inside
             a real attack window are still read, but only to measure how long
             an alarm takes to appear after the real start of an attack.)

Features follow Nemade et al. (2026), section 4.1, with two changes:
  * every rolling window is trailing (uses only hours up to t), so the score
    for hour t is causal like the Transformer's;
  * features are built inside each contiguous block, never across the 24-hour
    gaps between blocks, and the first WARMUP rows of each block are dropped.
    WARMUP = 23 = the first hour a 24-hour window can end on, so the scored
    hours are exactly the hours the 24-hour Transformer scores.
"""
import os

import numpy as np
import pandas as pd

LABEL_COLUMNS = ["label", "attack_id"]
NOT_FEATURES = ["DATETIME", "ATT_FLAG"] + LABEL_COLUMNS

PROCESSED_FILES = {
    "train": "train_scaled.csv",
    "val": "val_scaled.csv",
    "cal": "cal_scaled.csv",
    "d2": "dataset2_eval.csv"}

LAGS = (1, 2, 3)
DIFFS = (1, 3)
ROLL_WINDOWS = (3, 6, 24)
WARMUP = max(ROLL_WINDOWS) - 1  # 23


def build_features(df, time_features=True):
    """
    One feature row per hour of ONE contiguous block. Rows before WARMUP hold
    NaN (a rolling window or lag that reaches before the block); the caller
    drops them.
    """
    sensors = [c for c in df.columns if c not in NOT_FEATURES]
    X = df[sensors].astype(np.float64)

    parts = [X]
    for lag in LAGS:
        parts.append(X.shift(lag).add_suffix(f"_lag{lag}"))
    for step in DIFFS:
        parts.append((X - X.shift(step)).add_suffix(f"_diff{step}"))

    # Rolling statistics for tank levels and junction pressures only (paper 4.1).
    rolled = [c for c in sensors if c.startswith(("L_", "P_"))]
    R = X[rolled]
    for w in ROLL_WINDOWS:
        roll = R.rolling(w, min_periods=w)
        mean = roll.mean()
        parts.append(mean.add_suffix(f"_rmean{w}"))
        parts.append(roll.std().add_suffix(f"_rstd{w}"))
        parts.append((R - mean).add_suffix(f"_rdev{w}"))

    if time_features:
        t = pd.to_datetime(df["DATETIME"], format="%Y-%m-%d %H:%M:%S")
        hour, day = t.dt.hour, t.dt.dayofweek
        parts.append(pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * day / 7),
            "dow_cos": np.cos(2 * np.pi * day / 7)}, index=df.index))

    return pd.concat(parts, axis=1)


def load_block(path, name, time_features=True):
    """Read one processed file; returns (features, meta), warm-up rows removed."""
    df = pd.read_csv(path)
    when = pd.to_datetime(df["DATETIME"], format="%Y-%m-%d %H:%M:%S")
    if not (when.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path}: rows are not strictly hourly")

    flags = set(df["ATT_FLAG"].unique().tolist())
    if name == "d2":
        if not flags <= {-999, 1}:
            raise ValueError(f"{path}: unexpected ATT_FLAG values {sorted(flags)}")
        if any(c not in df.columns for c in LABEL_COLUMNS):
            raise ValueError(f"{path}: no label/attack_id columns; pass dataset2_eval.csv")
    elif flags != {0}:
        raise ValueError(f"{path}: Dataset 1 blocks must be all ATT_FLAG = 0")

    feats = build_features(df, time_features)
    meta = pd.DataFrame({"datetime": when, "att_flag": df["ATT_FLAG"], "row": np.arange(len(df))})
    if name == "d2":
        meta["label"] = df["label"].to_numpy()
        meta["attack_id"] = df["attack_id"].to_numpy()

    keep = np.arange(len(df)) >= WARMUP
    feats, meta = feats[keep].reset_index(drop=True), meta[keep].reset_index(drop=True)
    if feats.isna().any().any():
        raise ValueError(f"{path}: NaN left in features after the warm-up rows")
    return feats, meta


def load_blocks(processed_dir, time_features=True):
    """{'train','val','cal','d2'} -> (features, meta). Same column order in all."""
    blocks = {name: load_block(os.path.join(processed_dir, f), name, time_features)
              for name, f in PROCESSED_FILES.items()}
    columns = list(blocks["train"][0].columns)
    for name, (feats, _) in blocks.items():
        if list(feats.columns) != columns:
            raise ValueError(f"{name}: feature columns differ from the train block")
    return blocks
