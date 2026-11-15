"""
BATADAL data loader pipeline (PyTorch).

    from batadal_dataset import build_dataloaders
    loaders, scaler, info = build_dataloaders(
        "data/Training dataset 1.csv", "data/Training dataset 2.csv", window=24)
    for x, target, label, row in loaders["train"]:
        ...
"""
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

UNKNOWN = -1          # Label for "not known" (file value -999, or no label column)
N_CONTINUOUS = 31
N_BINARY = 12

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
def load_batadal_csv(path, round_decimals=2):
    """Read one BATADAL file. Returns (datetimes, feature_names, X, raw_flags)."""
    df = pd.read_csv(path)

    # (audit "Column names"): Dataset 2 has a leading space on every
    # header except DATETIME. Strip, so all three files share one schema.
    df.columns = df.columns.str.strip()

    # (audit "Time"): DATETIME is day/month/year hour. Parse it with an
    # explicit format, then require strictly hourly rows. The audit found no
    # gaps or repeats; this check makes the windowing fail loudly if a future
    # file breaks that, because a window across a gap is not 24 real hours.
    when = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
    if not (when.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path}: rows are not strictly hourly")

    features = [c for c in df.columns if c not in ("DATETIME", "ATT_FLAG")]

    # (audit "Data types"): the same column is int in one file and
    # float in another (1 vs 1.00). Cast everything to float32 so dtype never
    # depends on which file was read.
    X = df[features].to_numpy(np.float32)

    # (audit "Empty cells", "On/off columns"): no imputation code,
    # because there is nothing to impute. Assert it instead of assuming it.
    if np.isnan(X).any():
        raise ValueError(f"{path}: contains empty cells")
    binary = np.array([c.startswith("S_") for c in features])
    if not np.isin(X[:, binary], (0, 1)).all():
        raise ValueError(f"{path}: an S_ column has values other than 0 and 1")
    if (binary.sum(), (~binary).sum()) != (N_BINARY, N_CONTINUOUS):
        raise ValueError(f"{path}: expected 31 continuous + 12 on/off features")

    # (audit issue 5, exploration step 4 "Rounding"): Dataset 1 has up
    # to 9 decimals, Dataset 2 and the Test set have 2. Round everything to 2 so
    # the model trains on the precision it will be scored on. Otherwise a sensor
    # like P_J280 (range 0.017, so only 3 possible values at 2 decimals) shows
    # rounding steps the model never saw, which become false alarms.
    # float64 for the rounding itself, so 2.975 does not round by float32 noise.
    if round_decimals is not None:
        X = np.round(X.astype(np.float64), round_decimals).astype(np.float32)

    # (audit "Value ranges"): the L_T6 ceiling at 5.5 and the single
    # negative flow (F_PU7 = -0.88, inside a labeled attack) are left as they
    # are. Neither is treated as an error, so nothing is clipped or repaired.

    flags = df["ATT_FLAG"].to_numpy() if "ATT_FLAG" in df.columns else None
    return when.to_numpy(), features, X, flags


# Labels.
def make_labels(when, flags, label_rule="file", attack_windows=None):
    """
    Per-hour labels (1 attack, 0 normal, UNKNOWN) and per-hour attack ids.

    (audit issues 2 and 3, proposal label-sensitivity experiment):
    -999 means "not known", never "normal", and Dataset 2 has no 0 rows at all.
    So the rule is an explicit argument, one per proposal label rule:
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
class TrainOnlyScaler:
    """Min-max scaling whose statistics come from the training block only."""
    def __init__(
            self,
            feature_names,
            feature_range=(0.0, 1.0),
            constant_policy="drop"):
        if constant_policy not in ("drop", "keep"):
            raise ValueError(f"unknown constant_policy {constant_policy!r}")
        self.names = list(feature_names)
        self.lo, self.hi = feature_range
        self.constant_policy = constant_policy
        self.binary = np.array([c.startswith("S_") for c in self.names])

    def fit(self, X_train):
        # (proposal Milestone 1, audit issue 6): min and max come from
        # the training block only. Validation, calibration, Dataset 2 and the
        # Test set never touch them, so no future information leaks in.
        self.min_ = X_train.min(axis=0)
        span = X_train.max(axis=0) - self.min_
        constant = span == 0

        # (audit issue 1, exploration decisions 1-2): 7 columns never
        # change in training, so they have no range to scale by.
        #   "drop": (default) remove them. Simple, but the model can then
        #           never see pump 3, 5 or 9 switch on.
        #   "keep": leave them in. Pumps 6 and 11 are almost never on
        #           in Dataset 1 and 93-97% of their Dataset 2 running hours are
        #           inside attacks (exploration step 2), so "a pump that never
        #           runs starts running" is exactly the signal worth keeping.
        #           A constant flow borrows the median range of the flows that
        #           do move, so if it ever runs it lands near the same scale as
        #           its siblings instead of at a raw value like 119.
        for j in np.flatnonzero(constant & ~self.binary):
            family = self.names[j].split("_")[0]          # "F", "L" or "P"
            same = np.array([c.split("_")[0] == family for c in self.names])
            donors = span[same & ~constant & ~self.binary]
            span[j] = np.median(donors) if len(donors) else 1.0

        # (proposal Milestone 1): on/off columns are already 0/1, so they are
        # not scaled. Only the continuous features are.
        span[self.binary] = 1.0
        self.min_[self.binary] = 0.0
        self.span_ = span
        self.constant_ = [n for n, c in zip(self.names, constant) if c]
        self.keep_ = ~constant if self.constant_policy == "drop" \
            else np.ones(len(self.names), dtype=bool)
        self.out_names_ = [n for n, k in zip(self.names, self.keep_) if k]
        return self

    def transform(self, X):
        # (audit "Value ranges"): no clipping. L_T1 and P_J415 exceed
        # the training maximum in later files; a value outside the range the
        # model learned is evidence of an anomaly, and clipping would erase it.
        Z = self.lo + (X - self.min_) / self.span_ * (self.hi - self.lo)
        return Z[:, self.keep_].astype(np.float32)


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
        # (exploration step 4, decision 6): window is an argument, not
        # a constant. 12 and 24 are the proposal's values, but tank 1 cycles
        # about every 2 days, so 48 should be tried without touching this file.
        first = window - 1
        if len(X) <= first:
            raise ValueError(f"block has {len(X)} rows, too few for window {window}")

        # Unlike the template, nothing is padded and there is no start token:
        # every window is full length. The first W-1 hours of a file therefore
        # get no score, which costs nothing here (audit: every row of Dataset 2
        # before 2016-09-14 is unlabeled, and attack 1 starts in mid-September).
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
        dataset1_path,
        dataset2_path=None,
        test_path=None,
        unlock_test=False,
        window=24,
        batch_size=128,
        train_stride=1,
        split=(0.6, 0.2, 0.2),
        gap_hours=24,
        feature_range=(0.0, 1.0),
        constant_policy="drop",
        round_decimals=2,
        label_rule="schedule",
        attack_windows_dataset2=ATTACK_WINDOWS_DATASET2,
        seed=0,
        num_workers=0):
    """Returns (loaders, scaler, info). loaders has train/val/calib and,
    when the paths are given, dev (Dataset 2) and test."""
    # (proposal Milestone 1): the gap between blocks must cover a whole window.
    # With a shorter gap, the first validation or calibration windows would be
    # scored right next to hours the model trained on.
    if gap_hours < window:
        raise ValueError(f"gap_hours ({gap_hours}) must be at least window ({window})")

    when1, names, X1, flags1 = load_batadal_csv(dataset1_path, round_decimals)

    # (audit "Labels"): Dataset 1 must be entirely ATT_FLAG = 0. This
    # stops a partly labeled file (-999 or 1) being used as "normal" training
    # data by mistake.
    if flags1 is None or not (flags1 == 0).all():
        raise ValueError(f"{dataset1_path}: training data must be all ATT_FLAG = 0")

    # (proposal Milestone 1, exploration step 1): chronological
    # 60/20/20 split into train / validation / calibration. The drift check
    # found shifts under 0.1 SD and spread ratios of 0.94-1.01 across exactly
    # these blocks, so there is no reason to move the cut points. No shuffling
    # across blocks: that would leak neighbouring hours between them.
    n = len(X1)
    a, b = int(n * split[0]), int(n * (split[0] + split[1]))

    # (proposal): a 24 h gap after each cut, and windows are built
    # inside each block, so no window straddles two blocks and no validation
    # window overlaps hours the model trained on.
    blocks = {"train": (0, a), "val": (a + gap_hours, b), "calib": (b + gap_hours, n)}

    scaler = TrainOnlyScaler(names, feature_range, constant_policy).fit(X1[:a])

    labels1, _ = make_labels(when1, flags1)
    datasets = {}
    for name, (s, e) in blocks.items():
        datasets[name] = BATADAL_Dataset(
            scaler.transform(X1[s:e]), labels1[s:e], window,
            train_stride if name == "train" else 1, row_offset=s)

    def add_file(name, path, rule, attack_windows):
        when, names_k, X, flags = load_batadal_csv(path, round_decimals)
        if names_k != names:
            raise ValueError(f"{path}: feature columns differ from Dataset 1")
        labels, attack_id = make_labels(when, flags, rule, attack_windows)
        # Evaluation always uses stride 1: one score per hour.
        datasets[name] = BATADAL_Dataset(scaler.transform(X), labels, window, 1)
        info[name] = {"datetime": when, "labels": labels, "attack_id": attack_id}

    info = {"feature_names": scaler.out_names_, "in_dim": len(scaler.out_names_),
            "constant_in_train": scaler.constant_, "blocks": blocks,
            "datetime_dataset1": when1}

    # (exploration open items): Dataset 2 is never used to fit the
    # scaler or the model. It is a development set for label experiments and
    # for checking that errors rise during the 7 attacks.
    if dataset2_path is not None:
        add_file("dev", dataset2_path, label_rule, attack_windows_dataset2)

    # (audit issue 6): the Test set stays untouched until every model
    # choice and threshold is frozen, so loading it takes a deliberate flag.
    # It has no ATT_FLAG and its attack list was not audited: labels are UNKNOWN
    # unless you pass the site's windows yourself at final evaluation.
    if test_path is not None:
        if not unlock_test:
            raise PermissionError("Test set is locked. Pass unlock_test=True only "
                                  "after all choices and thresholds are frozen.")
        add_file("test", test_path, "file", None)

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
    # python batadal_dataset.py "data/Training dataset 1.csv" "data/Training dataset 2.csv"
    loaders, scaler, info = build_dataloaders(
        sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"features: {info['in_dim']} | constant in training: "
          f"{', '.join(info['constant_in_train']) or 'none'}")
    for name, loader in loaders.items():
        x, target, label, row = next(iter(loader))
        print(f"{name:5s} | windows {len(loader.dataset):5d} | input {tuple(x.shape)} "
              f"| target {tuple(target.shape)}")