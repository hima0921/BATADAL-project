"""
BATADAL data loader pipeline (PyTorch).

Reads the shared, already-prepared data that notebooks/Pipeline.ipynb saves in
data/processed/ (see docs/pipeline.md), so every model uses one version of the
data. Rounding, dropping the 7 never-changing columns, the 60/20/20 split, the
scaling and the Dataset 2 labels are all done by that notebook, not here.

    from batadal_dataset import build_dataloaders
    loaders, scaler, info = build_dataloaders(
        "data/processed", "data/processed/dataset2_eval.csv", window=24)
    for x, target, label, row in loaders["train"]:
        ...
"""
import os

import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import DataLoader, Dataset

# (proposal, exploration step 4): the Transformer read 12-hour and
# 24-hour windows. The pipeline's 24-hour gaps were sized for these two.
ALLOWED_WINDOWS = (12, 24)

# Only the Test set uses this. It has no label column at all, and 0 would read
# as "normal". Dataset 1 and Dataset 2 never get this value.
NO_LABEL = -1

# (pipeline step 7 and notebook step 8): the files Pipeline.ipynb writes.
PROCESSED_FILES = {
    "train": "train_scaled.csv",
    "val": "val_scaled.csv",
    "calib": "cal_scaled.csv"}
DATASET2_FILE = "dataset2_eval.csv"
SCALER_FILE = "scaler.joblib"

# (pipeline step 0b, audit issue 1): the 7 columns that never change in
# Dataset 1 or Dataset 2. The pipeline drops them, so they are not in the
# processed files. Listed here to report them and to drop them from the raw
# Test set at the end.
DROPPED_COLUMNS = ["S_PU1", "F_PU3", "S_PU3", "F_PU5", "S_PU5", "F_PU9", "S_PU9"]

# (notebook step 8): the two label columns of dataset2_eval.csv. They are the
# answer key, so they must never be read as features.
LABEL_COLUMNS = ["label", "attack_id"]
NOT_FEATURES = ["DATETIME", "ATT_FLAG"] + LABEL_COLUMNS


# Loading and Validation.
def check_table(path, when, features, X):
    """Checks shared by the processed files and the raw Test set."""
    # (audit "Time"): rows must be strictly hourly inside a file. The audit
    # found no gaps or repeats; this fails loudly if a file breaks that,
    # because a window across a gap is not 12 or 24 real hours.
    if not (when.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path}: rows are not strictly hourly")

    # (audit "Empty cells", "On/off columns"): nothing is imputed, because
    # there is nothing to impute. Assert it instead of assuming it.
    if np.isnan(X).any():
        raise ValueError(f"{path}: contains empty cells")
    binary = np.array([c.startswith("S_") for c in features])
    if not np.isin(X[:, binary], (0, 1)).all():
        raise ValueError(f"{path}: an S_ column has values other than 0 and 1")


def check_labels(path, flags, labels, attack_id):
    """
    Checks on the label columns of dataset2_eval.csv (notebook step 8).

    The labels are read from the file, not rebuilt here, so these checks are
    what stops a damaged or hand-edited file from being scored against.
    """
    if not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path}: label has values other than 0 and 1")
    if attack_id.min() < 0:
        raise ValueError(f"{path}: attack_id has negative values")
    # An hour is an attack hour exactly when it belongs to one of the attacks.
    if not ((labels == 1) == (attack_id > 0)).all():
        raise ValueError(f"{path}: label and attack_id disagree")
    # (audit "Attack periods"): every labeled run in the file sits inside a
    # real attack window, so an hour the file marks 1 can never be label 0.
    if not (labels[flags == 1] == 1).all():
        raise ValueError(f"{path}: an hour with ATT_FLAG = 1 has label 0")


def load_processed_csv(path, with_labels=False):
    """
    Read one file written by Pipeline.ipynb. Returns
    (datetimes, feature_names, X, raw_flags, labels, attack_id). The values are
    already rounded, scaled and without the 7 dropped columns, so nothing is
    changed here.

    with_labels=False  a Dataset 1 block. labels and attack_id come back None.
    with_labels=True   dataset2_eval.csv, which carries label and attack_id.
    """
    # (pipeline step 7): the processed files store DATETIME as text like
    # "2014-01-06 00:00:00". It is parsed after the column check below, so a
    # raw CSV passed by mistake gets a clear error, not a date-format one.
    df = pd.read_csv(path)

    present = [c for c in DROPPED_COLUMNS if c in df.columns]
    if len(present) > 0 or "ATT_FLAG" not in df.columns:
        raise ValueError(
            f"{path}: this does not look like a processed file from Pipeline.ipynb "
            f"(it should have ATT_FLAG and none of the 7 dropped columns). "
            f"Pass the files in data/processed/, not the raw CSVs.")

    has_labels = [c for c in LABEL_COLUMNS if c in df.columns]
    if with_labels and has_labels != LABEL_COLUMNS:
        raise ValueError(
            f"{path}: no label and attack_id columns. Pass {DATASET2_FILE} "
            f"(notebook step 8), not dataset2_scaled.csv.")
    if not with_labels and len(has_labels) > 0:
        raise ValueError(
            f"{path}: a Dataset 1 block should not have the columns {has_labels}")

    when = pd.to_datetime(df["DATETIME"], format="%Y-%m-%d %H:%M:%S")

    # The label columns are left out of the features here. This is the line
    # that keeps the answer key away from the models.
    features = [c for c in df.columns if c not in NOT_FEATURES]

    # (audit "Data types"): cast everything to float32 so dtype never depends
    # on which file was read. The rounding was already done by the pipeline,
    # in float64, before these values were saved.
    X = df[features].to_numpy(np.float32)

    # Validation step occurs here, in case values are changed in the csv.
    check_table(path, when, features, X)

    flags = df["ATT_FLAG"].to_numpy()
    labels = attack_id = None
    if with_labels:
        labels = df["label"].to_numpy(np.int64)
        attack_id = df["attack_id"].to_numpy(np.int64)
        check_labels(path, flags, labels, attack_id)
    return when.to_numpy(), features, X, flags, labels, attack_id


def load_raw_test_csv(path, scaler):
    """
    Read the RAW Test set and prepare it exactly as the pipeline prepared the
    other files (pipeline step 7, "How to load"): strip the column names,
    round to 2 decimals, drop the 7 columns, scale with the saved scaler.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    when = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")

    missing = [c for c in scaler.names if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"{path}: missing feature columns {missing}")

    # float64 for the rounding itself, so 2.975 does not round by float32 noise.
    X = np.round(df[scaler.names].to_numpy(np.float64), 2)
    X = scaler.transform(X)
    check_table(path, when, scaler.names, X)
    return when.to_numpy(), X


# Scaling.
class ProcessedScaler:
    """
    The scaler Pipeline.ipynb fitted and saved (pipeline step 3), in the form
    the training and evaluation scripts expect.

    (proposal Milestone 1, pipeline step 3): it is a MinMaxScaler fitted on the
    continuous Training columns only, to the range 0 to 1. Nothing is fitted
    here. The on/off columns were never scaled, so they get min 0 and span 1.
    """
    def __init__(self, scaler_path, feature_names):
        fitted = joblib.load(scaler_path)
        self.names = list(feature_names)
        self.binary = np.array([c.startswith("S_") for c in self.names])

        continuous = [c for c in self.names if not c.startswith("S_")]
        fitted_on = list(getattr(fitted, "feature_names_in_", continuous))
        if fitted_on != continuous:
            raise ValueError(
                f"{scaler_path}: the scaler was fitted on different columns than "
                f"the continuous columns of the processed files")

        self.min_ = np.zeros(len(self.names), dtype=np.float64)
        self.span_ = np.ones(len(self.names), dtype=np.float64)
        self.min_[~self.binary] = fitted.data_min_
        self.span_[~self.binary] = fitted.data_range_

        self.lo, self.hi = fitted.feature_range
        self.constant_policy = "drop"
        self.constant_ = list(DROPPED_COLUMNS)
        # Every column in the processed files is kept: the 7 constant columns
        # are already gone.
        self.keep_ = np.ones(len(self.names), dtype=bool)
        self.out_names_ = list(self.names)

    def transform(self, X):
        # (audit "Value ranges", pipeline step 4): no clipping. Dataset 2 runs
        # from about -1.28 to 2.86 after scaling; a value outside the range the
        # model learned is evidence of an anomaly, and clipping would erase it.
        Z = self.lo + (X - self.min_) / self.span_ * (self.hi - self.lo)
        return Z.astype(np.float32)


# Dataset.
class BATADAL_Dataset(Dataset):
    """
    Sliding windows over one contiguous, already-scaled block of hours.

    Returns (input_tensor, target_tensor, label, row):
      label  label of hour t;  row  position of hour t in the source file.
    """
    def __init__(
            self,
            X,
            labels,
            window=24,
            stride=1,
            row_offset=0):
        if window not in ALLOWED_WINDOWS:
            raise ValueError(f"window={window}: use one of {ALLOWED_WINDOWS}")
        first = window - 1
        if len(X) <= first:
            raise ValueError(f"block has {len(X)} rows, too few for window {window}")

        # Nothing is padded and there is no start token: every window is full
        # length, and no window crosses from one block or file into another.
        # The first 11 or 23 hours of each block therefore get no score. In
        # Dataset 2 that costs nothing (attack 1 starts on 2016-09-13, the file
        # on 2016-07-04). For the Test set it has not been checked, because
        # the Test set stays untouched.
        self.X = torch.from_numpy(np.ascontiguousarray(X))
        self.labels = torch.from_numpy(labels)
        self.window = window
        self.row_offset = row_offset

        # windows are sliced on demand from one tensor. Stacking all
        # of them up front would copy every hour W times.
        self.ends = np.arange(first, len(X), stride)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, index):
        t = int(self.ends[index])

        input_tensor = self.X[t - self.window + 1:t + 1]
        target_tensor = input_tensor

        # (exploration step 5, audit issue 4): a window is labeled by
        # its LAST hour, and that hour's row index is returned. The score for
        # hour t then uses only hours up to t, so alarms are causal and the
        # detection delay (hours from real start to first alarm) is meaningful.
        return input_tensor, target_tensor, self.labels[t], self.row_offset + t


# Dataset Loader.
def build_dataloaders(
        processed_dir,
        dataset2_path=None,
        test_path=None,
        unlock_test=False,
        window=24,
        batch_size=128,
        train_stride=1,
        feature_range=(0.0, 1.0),
        constant_policy="drop",
        seed=0,
        num_workers=0):
    """
    Returns (loaders, scaler, info). loaders has train/val/calib and, when the
    paths are given, dev (Dataset 2) and test.

    processed_dir   the folder Pipeline.ipynb writes (data/processed/).
    dataset2_path   dataset2_eval.csv: Dataset 2, scaled, with label and
                    attack_id (notebook step 8).
    test_path       the RAW Test set CSV; it is prepared here with the saved
                    scaler, and only with unlock_test=True.
    window          12 or 24 hours.
    """
    processed_dir = str(processed_dir)
    if not os.path.isdir(processed_dir):
        raise ValueError(
            f"{processed_dir}: expected the folder of processed files written by "
            f"Pipeline.ipynb (data/processed/), not a single CSV file")

    if window not in ALLOWED_WINDOWS:
        raise ValueError(f"window={window}: use one of {ALLOWED_WINDOWS}")

    # (pipeline decisions 1-3): rounding, the dropped columns and the 0 to 1
    # range are fixed by the shared pipeline and cannot be changed from here.
    # These two arguments are kept so a different request fails loudly instead
    # of silently getting the pipeline's data.
    if tuple(feature_range) != (0.0, 1.0):
        raise ValueError(
            f"feature_range={tuple(feature_range)}: the processed files are scaled "
            f"to 0 to 1 (pipeline decision 3). Change it in Pipeline.ipynb.")
    if constant_policy != "drop":
        raise ValueError(
            f"constant_policy={constant_policy!r}: the processed files have the 7 "
            f"never-changing columns dropped (pipeline decision 2).")

    # (pipeline step 2): the three blocks of Dataset 1, already split by time
    # 60/20/20 with a 24-hour gap after each cut.
    tables = {}
    for name, file_name in PROCESSED_FILES.items():
        tables[name] = load_processed_csv(os.path.join(processed_dir, file_name))
    names = tables["train"][1]
    for name, table in tables.items():
        if table[1] != names:
            raise ValueError(f"{PROCESSED_FILES[name]}: feature columns differ from the Training file")
        # (audit "Labels"): Dataset 1 must be entirely ATT_FLAG = 0. This
        # stops a partly labeled file (-999 or 1) being used as "normal" data
        # by mistake.
        if not (table[3] == 0).all():
            raise ValueError(f"{PROCESSED_FILES[name]}: Dataset 1 blocks must be all ATT_FLAG = 0")

    # (pipeline step 3): check the Training file really is the scaled one.
    # After fitting, every continuous Training column runs from 0 to 1.
    continuous = np.array([not c.startswith("S_") for c in names])
    train_X = tables["train"][2]
    if not (np.allclose(train_X[:, continuous].min(axis=0), 0.0, atol=1e-6)
            and np.allclose(train_X[:, continuous].max(axis=0), 1.0, atol=1e-6)):
        raise ValueError(f"{PROCESSED_FILES['train']}: continuous columns do not run "
                         f"from 0 to 1, so this is not the scaled Training file")

    scaler = ProcessedScaler(os.path.join(processed_dir, SCALER_FILE), names)

    # Row positions in Dataset 1, from the dates, so `row` means the same as
    # it would in the raw file (hours since the first Training hour).
    first_hour = tables["train"][0][0]
    blocks = {}
    for name, table in tables.items():
        start = int((table[0][0] - first_hour) / np.timedelta64(1, "h"))
        blocks[name] = (start, start + len(table[2]))

    # (proposal Milestone 1, pipeline step 2): the gap between blocks must
    # cover a whole window, so the first validation or calibration windows are
    # not scored right next to hours the model trained on. The pipeline's
    # 24-hour gaps cover both allowed windows; this catches a changed split.
    for before, after in (("train", "val"), ("val", "calib")):
        gap_hours = blocks[after][0] - blocks[before][1]
        if gap_hours < window:
            raise ValueError(
                f"the gap between the {before} and {after} blocks is {gap_hours} hours, "
                f"shorter than the window ({window})")

    datasets = {}
    for name, table in tables.items():
        # All ATT_FLAG = 0 was checked above, so every Dataset 1 hour is 0.
        labels = np.zeros(len(table[2]), dtype=np.int64)
        datasets[name] = BATADAL_Dataset(
            table[2], labels, window,
            train_stride if name == "train" else 1, row_offset=blocks[name][0])

    info = {"feature_names": scaler.out_names_, "in_dim": len(scaler.out_names_),
            "constant_in_train": scaler.constant_, "blocks": blocks,
            "datetime_blocks": {name: table[0] for name, table in tables.items()}}

    # (pipeline step 4, notebook step 8): Dataset 2 was scaled with the
    # Training scaler and is never used to fit anything. It is only for
    # checking models.
    #   label      1 inside the 7 real attack windows, 0 otherwise. The 0s are
    #              ASSUMED normal: they rest on the site's list of 7 attacks
    #              being complete (audit issue 2). They are not confirmed.
    #   attack_id  1 to 7, or 0. Its first hour is the attack's REAL start, so
    #              detection delay is measured from there (audit issue 4).
    #   att_flag   the raw file label (-999 or 1), passed through untouched for
    #              the label-sensitivity experiment. From att_flag and label:
    #                unlabeled as normal    att_flag == 1
    #                confirmed labels only  keep the hours where att_flag != -999
    #                real attack windows    label
    if dataset2_path is not None:
        when, names_k, X, flags, labels, attack_id = load_processed_csv(
            str(dataset2_path), with_labels=True)
        if names_k != names:
            raise ValueError(f"{dataset2_path}: feature columns differ from the Training file")
        # Evaluation always uses stride 1: one score per hour.
        datasets["dev"] = BATADAL_Dataset(X, labels, window, 1)
        info["dev"] = {"datetime": when, "labels": labels, "attack_id": attack_id,
                       "att_flag": flags}

    # (audit issue 6, pipeline step 7): the Test set stays untouched until
    # every model choice and threshold is frozen, so loading it takes a
    # deliberate flag. The pipeline does not save a scaled Test file; the raw
    # file is prepared here with the saved scaler. It has no ATT_FLAG and its
    # attack list was not audited, so every hour gets NO_LABEL.
    if test_path is not None:
        if not unlock_test:
            raise PermissionError("Test set is locked. Pass unlock_test=True only "
                                  "after all choices and thresholds are frozen.")
        when, X = load_raw_test_csv(str(test_path), scaler)
        labels = np.full(len(X), NO_LABEL, dtype=np.int64)
        datasets["test"] = BATADAL_Dataset(X, labels, window, 1)
        info["test"] = {"datetime": when, "labels": labels}

    # (proposal "three random seeds"): only the training loader
    # shuffles, with its own seeded generator, so each seed gives a
    # reproducible batch order. Everything else stays in time order, which the
    # score-to-hour mapping and the episode grouping of alarms rely on.
    gen = torch.Generator().manual_seed(seed)
    loaders = {
        name: DataLoader(ds, batch_size=batch_size, shuffle=(name == "train"),
                         generator=gen if name == "train" else None,
                         num_workers=num_workers)
        for name, ds in datasets.items()}
    return loaders, scaler, info

if __name__ == "__main__":
    import sys
    # python batadal_dataset.py data/processed [window]
    processed = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    loaders, scaler, info = build_dataloaders(
        processed, os.path.join(processed, DATASET2_FILE), window=window)
    print(f"features: {info['in_dim']} | dropped by the pipeline: "
          f"{', '.join(info['constant_in_train'])}")
    for name, loader in loaders.items():
        x, target, label, row = next(iter(loader))
        # (input_tensor, target_tensor, label, row)
        print(f"{name:5s} | windows {len(loader.dataset):5d} | input {tuple(x.shape)} "
              f"| target {tuple(target.shape)}")
