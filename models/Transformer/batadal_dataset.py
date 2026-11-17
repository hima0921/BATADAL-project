"""
BATADAL data loader pipeline (PyTorch).

Reads the shared, already-prepared data that notebooks/Pipeline.ipynb saves in
data/processed/ (see docs/pipeline.md), so every model uses one version of the
data. Rounding, dropping the 7 never-changing columns, the 60/20/20 split and
the scaling are done by that notebook, not here.

    from batadal_dataset import build_dataloaders
    loaders, scaler, info = build_dataloaders(
        "data/processed", "data/processed/dataset2_scaled.csv", window=24)
    for x, target, label, row in loaders["train"]:
        ...
"""
import os

import numpy as np
import pandas as pd
import joblib

import torch
from torch.utils.data import DataLoader, Dataset

UNKNOWN = -1          # Label for "not known" (file value -999, or no label column)

# (pipeline step 7): the files Pipeline.ipynb writes to data/processed/.
PROCESSED_FILES = {
    "train": "train_scaled.csv",
    "val": "val_scaled.csv",
    "calib": "cal_scaled.csv"}
DATASET2_FILE = "dataset2_scaled.csv"
SCALER_FILE = "scaler.joblib"

# (pipeline step 0b, audit issue 1): the 7 columns that never change in
# Dataset 1 or Dataset 2. The pipeline drops them, so they are not in the
# processed files. Listed here to report them and to drop them from the raw
# Test set at the end.
DROPPED_COLUMNS = ["S_PU1", "F_PU3", "S_PU3", "F_PU5", "S_PU5", "F_PU9", "S_PU9"]

# (audit s3 "Attack periods", issue 4): the 7 REAL attack windows of
# Dataset 2 from the BATADAL site, not the file labels. The file labels cover
# only 219 of 492 attack hours and start up to 24 h late. Both ends inclusive,
# which reproduces the site's hour counts (50, 24, 60, 94, 60, 94, 110).
ATTACK_WINDOWS_DATASET2 = [
    (1, "2016-09-13 23:00", "2016-09-16 00:00"),
    (2, "2016-09-26 11:00", "2016-09-27 10:00"),
    (3, "2016-10-09 09:00", "2016-10-11 20:00"),
    (4, "2016-10-29 19:00", "2016-11-02 16:00"),
    (5, "2016-11-26 17:00", "2016-11-29 04:00"),
    (6, "2016-12-06 07:00", "2016-12-10 04:00"),
    (7, "2016-12-14 15:00", "2016-12-19 04:00"),
]


# Loading and Validation.
def check_table(path, when, features, X):
    """Checks shared by the processed files and the raw Test set."""
    # (audit "Time"): rows must be strictly hourly inside a file. The audit
    # found no gaps or repeats; this fails loudly if a file breaks that,
    # because a window across a gap is not 24 real hours.
    if not (when.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path}: rows are not strictly hourly")

    # (audit "Empty cells", "On/off columns"): nothing is imputed, because
    # there is nothing to impute. Assert it instead of assuming it.
    if np.isnan(X).any():
        raise ValueError(f"{path}: contains empty cells")
    binary = np.array([c.startswith("S_") for c in features])
    if not np.isin(X[:, binary], (0, 1)).all():
        raise ValueError(f"{path}: an S_ column has values other than 0 and 1")


def load_processed_csv(path):
    """
    Read one file written by Pipeline.ipynb. Returns
    (datetimes, feature_names, X, raw_flags). The values are already rounded,
    scaled and without the 7 dropped columns, so nothing is changed here.
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
    when = pd.to_datetime(df["DATETIME"], format="%Y-%m-%d %H:%M:%S")

    features = [c for c in df.columns if c not in ("DATETIME", "ATT_FLAG")]

    # (audit "Data types"): cast everything to float32 so dtype never depends
    # on which file was read. The rounding was already done by the pipeline,
    # in float64, before these values were saved.
    X = df[features].to_numpy(np.float32)
    check_table(path, when, features, X)
    return when.to_numpy(), features, X, df["ATT_FLAG"].to_numpy()


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


# Labels.
def make_labels(when, flags, label_rule="file", attack_windows=None):
    """
    Per-hour labels (1 attack, 0 normal, UNKNOWN) and per-hour attack ids.

    (audit issues 2 and 3, proposal label-sensitivity experiment, pipeline
    open item 1): -999 means "not known", never "normal", and Dataset 2 has no
    0 rows at all. The processed files keep the raw ATT_FLAG, so the rule is
    applied here, as an explicit argument, one per proposal label rule:
      "file"                -999 -> UNKNOWN. Only confirmed labels (rule 2).
      "schedule"            1 inside the real attack windows, else 0 (rule 3).
                            The 0s are ASSUMED normal: they rest on the site's
                            list being complete.
      "unlabeled_as_normal" -999 -> 0. The deliberately flawed control (rule 1):
                            it turns about 273 real attack hours into "normal".
    """
    n = len(when)
    attack_id = np.zeros(n, dtype=np.int64)
    for aid, start, end in attack_windows or []:
        inside = (when >= np.datetime64(start)) & (when <= np.datetime64(end))
        attack_id[inside] = aid

    if label_rule == "schedule":
        if not attack_windows:
            raise ValueError("label_rule='schedule' needs attack_windows")
        labels = (attack_id > 0).astype(np.int64)
    elif flags is None:                       # Test set: no ATT_FLAG column
        labels = np.full(n, UNKNOWN, dtype=np.int64)
    elif label_rule == "file":
        labels = np.where(flags == 1, 1, np.where(flags == 0, 0, UNKNOWN))
    elif label_rule == "unlabeled_as_normal":
        labels = (flags == 1).astype(np.int64)
    else:
        raise ValueError(f"unknown label_rule {label_rule!r}")

    # (audit issue 4): attack_id comes from the REAL windows under
    # every rule, so detection delay can be measured from the real start. From
    # the labeled start it would look up to 24 h shorter than it is (attack 4).
    return labels.astype(np.int64), attack_id


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
        # (pipeline open item 3, exploration step 4): window is an argument,
        # not a constant. 12 and 24 are the proposal's values.
        first = window - 1
        if len(X) <= first:
            raise ValueError(f"block has {len(X)} rows, too few for window {window}")

        # Nothing is padded and there is no start token: every window is full
        # length. The first W-1 hours of a file therefore get no score, which
        # costs nothing here (audit: every row of Dataset 2 before 2016-09-14
        # is unlabeled, and attack 1 starts in mid-September).
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
        label_rule="schedule",
        attack_windows_dataset2=ATTACK_WINDOWS_DATASET2,
        seed=0,
        num_workers=0):
    """
    Returns (loaders, scaler, info). loaders has train/val/calib and, when the
    paths are given, dev (Dataset 2) and test.

    processed_dir   the folder Pipeline.ipynb writes (data/processed/).
    dataset2_path   the processed Dataset 2 file (dataset2_scaled.csv).
    test_path       the RAW Test set CSV; it is prepared here with the saved
                    scaler, and only with unlock_test=True.
    """
    processed_dir = str(processed_dir)
    if not os.path.isdir(processed_dir):
        raise ValueError(
            f"{processed_dir}: expected the folder of processed files written by "
            f"Pipeline.ipynb (data/processed/), not a single CSV file")

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
    for name, (_, names_k, _, flags) in tables.items():
        if names_k != names:
            raise ValueError(f"{PROCESSED_FILES[name]}: feature columns differ from the Training file")
        # (audit "Labels"): Dataset 1 must be entirely ATT_FLAG = 0. This
        # stops a partly labeled file (-999 or 1) being used as "normal" data
        # by mistake.
        if not (flags == 0).all():
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
    for name, (when, _, X, _) in tables.items():
        start = int((when[0] - first_hour) / np.timedelta64(1, "h"))
        blocks[name] = (start, start + len(X))

    # (proposal Milestone 1, pipeline step 2): the gap between blocks must
    # cover a whole window. With a shorter gap, the first validation or
    # calibration windows would be scored right next to hours the model
    # trained on.
    for before, after in (("train", "val"), ("val", "calib")):
        gap_hours = blocks[after][0] - blocks[before][1]
        if gap_hours < window:
            raise ValueError(
                f"the gap between the {before} and {after} blocks is {gap_hours} hours, "
                f"shorter than the window ({window})")

    datasets = {}
    for name, (when, _, X, flags) in tables.items():
        labels, _ = make_labels(when, flags)
        datasets[name] = BATADAL_Dataset(
            X, labels, window,
            train_stride if name == "train" else 1, row_offset=blocks[name][0])

    info = {"feature_names": scaler.out_names_, "in_dim": len(scaler.out_names_),
            "constant_in_train": scaler.constant_, "blocks": blocks,
            "datetime_blocks": {name: table[0] for name, table in tables.items()}}

    # (pipeline step 4, exploration open items): Dataset 2 was scaled with the
    # Training scaler and is never used to fit anything. It is a development
    # set for label experiments and for checking that errors rise during the
    # 7 attacks.
    if dataset2_path is not None:
        when, names_k, X, flags = load_processed_csv(str(dataset2_path))
        if names_k != names:
            raise ValueError(f"{dataset2_path}: feature columns differ from the Training file")
        labels, attack_id = make_labels(when, flags, label_rule, attack_windows_dataset2)
        # Evaluation always uses stride 1: one score per hour.
        datasets["dev"] = BATADAL_Dataset(X, labels, window, 1)
        info["dev"] = {"datetime": when, "labels": labels, "attack_id": attack_id}

    # (audit issue 6, pipeline step 7): the Test set stays untouched until
    # every model choice and threshold is frozen, so loading it takes a
    # deliberate flag. The pipeline does not save a scaled Test file; the raw
    # file is prepared here with the saved scaler. It has no ATT_FLAG and its
    # attack list was not audited, so its labels are UNKNOWN.
    if test_path is not None:
        if not unlock_test:
            raise PermissionError("Test set is locked. Pass unlock_test=True only "
                                  "after all choices and thresholds are frozen.")
        when, X = load_raw_test_csv(str(test_path), scaler)
        labels, attack_id = make_labels(when, None, "file", None)
        datasets["test"] = BATADAL_Dataset(X, labels, window, 1)
        info["test"] = {"datetime": when, "labels": labels, "attack_id": attack_id}

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
    # python batadal_dataset.py data/processed data/processed/dataset2_scaled.csv
    loaders, scaler, info = build_dataloaders(
        sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"features: {info['in_dim']} | dropped by the pipeline: "
          f"{', '.join(info['constant_in_train'])}")
    for name, loader in loaders.items():
        x, target, label, row = next(iter(loader))
        print(f"{name:5s} | windows {len(loader.dataset):5d} | input {tuple(x.shape)} "
              f"| target {tuple(target.shape)}")
