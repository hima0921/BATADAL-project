"""
Freeze Ridge's alarm calibration, so the Test set can be scored later with fixed thresholds.

It uses Vincent's shared code (models/Transformer/evaluate_recon_model.py) with the same
settings as notebooks/Ridge_shared_eval.ipynb:
    calibrate(calib_errors, val_errors, top_k=3, max_alarm_fraction=0.01,
              max_episodes_per_week=1.0, merge_gap_hours=2)
on the squared per-sensor Ridge errors of the Calibration and Validation blocks
(Dataset 1 only, from models/Ridge/ridge_errors.csv). Nothing is fitted on Dataset 2
or the Test set.

It saves calibration.pt and calibration.json with save_calibration(), reloads the saved
calibration.pt, runs run_evaluation(..., grace_hours=3) on Dataset 2 into a temporary folder,
and checks that the result matches models/Ridge/evaluation/metrics.json and per_attack.csv
(the saved shared-evaluation results). It stops with an error if anything differs.
models/Ridge/evaluation/ itself is never written to.

Run from the repo root:
    python models/Ridge/freeze_calibration.py                      # writes models/Ridge/calibration/
    python models/Ridge/freeze_calibration.py --out-dir <folder>   # e.g. to re-check without overwriting
"""
import argparse
import json
import os
import pathlib
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Windows: torch and numpy both ship OpenMP

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models" / "Transformer"))
from evaluate_recon_model import (  # noqa: E402
    calibrate, save_calibration, load_calibration, run_evaluation)

ERRORS = ROOT / "models" / "Ridge" / "ridge_errors.csv"
DATASET2 = ROOT / "data" / "processed" / "dataset2_eval.csv"
SAVED_EVAL = ROOT / "models" / "Ridge" / "evaluation"

# The settings of notebooks/Ridge_shared_eval.ipynb (the team's shared settings).
TOP_K = 3
MAX_ALARM_FRACTION = 0.01
MAX_EPISODES_PER_WEEK = 1.0
MERGE_GAP_HOURS = 2
GRACE_HOURS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=pathlib.Path, default=ROOT / "models" / "Ridge" / "calibration")
    args = parser.parse_args()

    errs = pd.read_csv(ERRORS, parse_dates=["DATETIME"])
    sensors = [c for c in errs.columns if c not in ["DATETIME", "split"]]
    val_errors = errs.loc[errs["split"] == "validation", sensors].to_numpy() ** 2
    calib_errors = errs.loc[errs["split"] == "calibration", sensors].to_numpy() ** 2
    dev = errs[errs["split"] == "dataset2"].reset_index(drop=True)
    dev_errors = dev[sensors].to_numpy() ** 2

    d2 = pd.read_csv(DATASET2, parse_dates=["DATETIME"])
    if not (d2["DATETIME"].values == dev["DATETIME"].values).all():
        raise ValueError("ridge_errors.csv and dataset2_eval.csv cover different hours")

    # 1. Calibrate on Dataset 1 only and save.
    calibration = calibrate(calib_errors, val_errors, top_k=TOP_K, max_alarm_fraction=MAX_ALARM_FRACTION,
                            max_episodes_per_week=MAX_EPISODES_PER_WEEK, merge_gap_hours=MERGE_GAP_HOURS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_calibration(calibration, str(args.out_dir), extra={
        "model": "Ridge (models/Ridge/ridge_model.joblib)",
        "errors": "squared per-sensor errors from models/Ridge/ridge_errors.csv",
        "data": "Dataset 1 only (cal_scaled.csv, val_scaled.csv)",
        "calibration_rows": int(len(calib_errors)),
        "validation_rows": int(len(val_errors)),
        "feature_names": sensors})
    print("saved", args.out_dir / "calibration.pt", "and calibration.json")

    # 2. Reload the saved file and check it reproduces the saved Dataset 2 results.
    frozen, _ = load_calibration(str(args.out_dir / "calibration.pt"))
    for score_type, threshold in calibration["thresholds"].items():
        if frozen["thresholds"][score_type] != threshold:
            raise SystemExit(f"{score_type}: saved threshold differs from the fresh one")
    with tempfile.TemporaryDirectory() as tmp:
        run_evaluation(frozen, dev_errors, d2["DATETIME"].to_numpy(), d2["label"].to_numpy(),
                       d2["attack_id"].to_numpy(), sensors, out_dir=tmp, grace_hours=GRACE_HOURS,
                       log=lambda *a, **k: None)
        new = json.load(open(os.path.join(tmp, "metrics.json")))
        new_attacks = pd.read_csv(os.path.join(tmp, "per_attack.csv"))
    old = json.load(open(SAVED_EVAL / "metrics.json"))
    old_attacks = pd.read_csv(SAVED_EVAL / "per_attack.csv")

    differ = [k for k in old if not (old[k] == new.get(k) or
              (isinstance(old[k], float) and abs(old[k] - new[k]) < 1e-12))]
    if differ or not old_attacks.equals(new_attacks):
        raise SystemExit(f"results differ from {SAVED_EVAL}: {differ or 'per_attack.csv'}")

    k = "topk_percentile_"
    print(f"thresholds: { {s: round(t, 6) for s, t in frozen['thresholds'].items()} }")
    print(f"matches {SAVED_EVAL.relative_to(ROOT)} ({len(old)} metrics, per_attack.csv): "
          f"S {new[k + 's_score']:.3f}, {new[k + 'attacks_detected']}/7 attacks, "
          f"mean delay {new[k + 'mean_delay_hours']:.1f} h, "
          f"{new[k + 'false_alarm_episodes_per_week']:.2f} false-alarm episodes/week")


if __name__ == "__main__":
    main()
