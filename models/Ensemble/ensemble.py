"""
Combines the per-hour scores of the detectors and evaluates the result.

    python ensemble.py \
        --xgb-dir ../XGBoost/runs/xgb \
        --teammate-iforest-zip ../Isolation_Forest/iforest_ensemble_outputs.zip \
        --transformer-scores transformer_hourly_scores.csv \
        --out-dir runs/ensemble

--transformer-scores is optional (made by export_transformer_scores.py). Without
it the ensemble is XGBoost + Isolation Forest only.

Everything an ensemble could be tuned on is kept away from Dataset 2:
  1. each score is turned into a common scale using the CALIBRATION block only
       percentile : share of calibration scores it is >= to (0 to 1, saturates at 1)
       ratio      : log(1 + score / calibration 99th percentile)  (no ceiling)
  2. detectors are fused by a fixed rule, not fitted weights:
       mean       equal-weight mean
       paper      0.4 XGBoost / 0.1 Isolation Forest / 0.5 reconstruction model,
                  the paper's weights, renormalised over the detectors present
       max        the largest normalised score
  3. the alarm threshold is the lowest one inside the false-alarm budget
     (<= 1 episode/week, <= 1% of hours) on the calibration block. An episode merges
    alarm runs <= 2 quiet hours apart (xgb_metrics.MERGE_GAP_HOURS), the same for all detectors.

No precision/F1 (they mix Dataset 2 and validation hours); the S-score is "approximate".
The Dataset 2 hours outside every attack window are scored as an era check
(d2_clean_alarm_fraction and d2_clean_episodes_per_week; hours outside every window,
minus 3 grace hours after each attack; the 1% / 1-per-week budget applies).

Evaluation matches train_xgboost.py: positives are the ATT_FLAG == 1 hours of
Dataset 2 (-999 dropped); false alarms are counted on the validation block;
detection delay is measured over the whole real attack window. XGBoost's
Dataset 2 scores are out-of-fold (the model that scored an attack never trained
on it). Dataset 1 hours are scored by the mean of its fold models.
"""
import os
import sys
import json
import pathlib
import argparse
import itertools

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "XGBoost"))
from xgb_metrics import roc_auc, pr_auc, clean_block_stats, find_threshold, HOURS_PER_WEEK, normal_hours_mask, normal_hours_stats  # noqa: E402

DETECTORS = {"xgb": "xgb_score", "iforest": "if_score"}
PAPER_WEIGHTS = {"xgb": 0.4, "iforest": 0.1, "transformer": 0.5}


def read_teammate_iforest(zip_path, seed):
    """
    Scores of the team's Isolation Forest (models/Isolation_Forest/iforest_ensemble_outputs.zip):
    36 raw sensors, 200 trees, fitted on the Dataset 1 train block. Read as plain CSV from the
    zip (the .joblib models in it are never loaded). Keyed by DATETIME.
    """
    import zipfile
    frames = []
    with zipfile.ZipFile(zip_path) as z:
        for source, name in (("val", "val"), ("cal", "cal"), ("d2", "dataset2")):
            with z.open(f"scores/iforest_scores_{name}.csv") as f:
                df = pd.read_csv(f)
            frames.append(pd.DataFrame({"source": source, "datetime": pd.to_datetime(df["DATETIME"]),
                                        "if_score": df[f"score_seed{seed}"]}))
    return pd.concat(frames, ignore_index=True)


def load_scores(xgb_path, forest, transformer_path=None, transformer_column=None):
    """One frame keyed by (source, row) holding only the hours every detector has."""
    xgb = pd.read_csv(xgb_path)[["source", "datetime", "row", "att_flag", "attack_id", "xgb_score"]]
    xgb["datetime"] = pd.to_datetime(xgb["datetime"])
    keys = ["source", "row"] if "row" in forest.columns else ["source", "datetime"]
    frame = xgb.merge(forest, on=keys, how="inner", validate="1:1")
    if transformer_path is not None:
        tr = pd.read_csv(transformer_path)[["source", "row", transformer_column]]
        frame = frame.merge(tr.rename(columns={transformer_column: "tr_score"}),
                            on=["source", "row"], how="inner", validate="1:1")
    return frame.sort_values(["source", "row"]).reset_index(drop=True)


def normalise(frame, column, how):
    """Scale one detector's scores using the calibration block only."""
    ref = frame.loc[frame.source == "cal", column].to_numpy()
    values = frame[column].to_numpy()
    if how == "percentile":
        return np.searchsorted(np.sort(ref), values, side="right") / len(ref)
    return np.log1p(values / max(np.percentile(ref, 99), 1e-12))


def evaluate(frame, column, max_alarm_fraction=0.01, max_episodes_per_week=1.0):
    cal = frame[frame.source == "cal"][column].to_numpy()
    val = frame[frame.source == "val"][column].to_numpy()
    threshold = find_threshold(cal, max_alarm_fraction, max_episodes_per_week)
    val_stats = clean_block_stats(val, threshold)
    cal_stats = clean_block_stats(cal, threshold)

    d2 = frame[frame.source == "d2"]
    per_attack, ttd, tp_total, pos_total = [], [], 0, 0
    for a, part in d2[d2.attack_id > 0].groupby("attack_id"):
        part = part.sort_values("row")
        alarms = (part[column] > threshold).to_numpy()
        labeled = (part.att_flag == 1).to_numpy()
        first = np.flatnonzero(alarms)
        # Delay from the real start of the attack; the window rows are consecutive hours,
        # but the first scored hour may be later than the real start if hours are missing.
        delay = int(part.row.iloc[first[0]] - part.row.iloc[0]) if len(first) else None
        hours = int(len(part))
        tp = int((alarms & labeled).sum())
        tp_total += tp
        pos_total += int(labeled.sum())
        ttd.append((delay if delay is not None else hours) / hours)
        per_attack.append({"attack": int(a), "detected": delay is not None, "delay": delay,
                           "labeled": int(labeled.sum()), "labeled_alarmed": tp,
                           "window_hours": hours, "window_alarmed": int(alarms.sum())})

    recall = tp_total / pos_total if pos_total else None
    tnr = 1.0 - val_stats["alarm_fraction"]
    pos = d2[d2.att_flag == 1][column].to_numpy()
    rank_scores = np.concatenate([pos, val])
    rank_labels = np.concatenate([np.ones(len(pos)), np.zeros(len(val))])
    # Schedule labels: every hour of a real attack window is positive.
    win = d2[d2.attack_id > 0][column].to_numpy()
    sch_scores = np.concatenate([win, val])
    sch_labels = np.concatenate([np.ones(len(win)), np.zeros(len(val))])
    # Era check: Dataset 2 hours outside every attack window should alarm at about the budget.
    d2s = d2.sort_values("row")
    normal = normal_hours_stats((d2s[column] > threshold).to_numpy(), normal_hours_mask(d2s.attack_id.to_numpy()),
                                d2s.row.to_numpy())
    s_ttd = 1.0 - float(np.mean(ttd))
    s_clf = (recall + tnr) / 2.0
    detected = [r for r in per_attack if r["detected"]]
    # No precision or F1: the true positives (Dataset 2) and false positives (validation block)
    # come from different series, so they would only reflect the chosen mix. Use alarm hours
    # per clean week here and compute precision/F1 on the Test set.
    return {
        "detected": len(detected), "attacks": len(per_attack),
        "mean_delay": float(np.mean([r["delay"] for r in detected])) if detected else None,
        "recall": recall, "tnr": tnr,
        "val_alarm_hours_per_week": val_stats["alarm_hours"] / (len(val) / HOURS_PER_WEEK),
        "val_episodes_per_week": val_stats["episodes_per_week"],
        "cal_alarm_fraction": cal_stats["alarm_fraction"],
        "d2_clean_alarm_fraction": normal["alarm_fraction"],
        "d2_clean_episodes_per_week": normal["episodes_per_week"],
        "recall_schedule": float(sum(r["window_alarmed"] for r in per_attack) / max(sum(r["window_hours"] for r in per_attack), 1)),
        "roc_auc": roc_auc(rank_scores, rank_labels), "pr_auc": pr_auc(rank_scores, rank_labels),
        "roc_auc_schedule": roc_auc(sch_scores, sch_labels),
        "s_score_approx": 0.5 * s_ttd + 0.5 * s_clf, "threshold": threshold}, per_attack


def run(frame, names, norm, per_attack_rows=None):
    """All single detectors and every combination, under one normalisation."""
    columns = {**DETECTORS, **({"transformer": "tr_score"} if "tr_score" in frame else {})}
    scaled = pd.DataFrame({n: normalise(frame, c, norm) for n, c in columns.items()})
    results = []
    for size in range(1, len(columns) + 1):
        for combo in itertools.combinations(columns, size):
            fusions = {"single": scaled[list(combo)[0]]} if size == 1 else {
                "mean": scaled[list(combo)].mean(axis=1),
                "paper": sum(PAPER_WEIGHTS[n] * scaled[n] for n in combo) / sum(PAPER_WEIGHTS[n] for n in combo),
                "max": scaled[list(combo)].max(axis=1)}
            for fusion, score in fusions.items():
                view = frame[["source", "row", "att_flag", "attack_id"]].copy()
                view["score"] = score.to_numpy()
                metrics, per = evaluate(view, "score")
                if per_attack_rows is not None:
                    per_attack_rows.extend({"norm": norm, "detectors": "+".join(combo), "fusion": fusion, **r} for r in per)
                results.append({"norm": norm, "detectors": "+".join(combo), "fusion": fusion, **metrics})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-dir", type=pathlib.Path, required=True)
    parser.add_argument("--iforest-dir", type=pathlib.Path, default=None,
                        help="Scores from isolation_forest.py (this folder's own Isolation Forest).")
    parser.add_argument("--teammate-iforest-zip", type=pathlib.Path, default=None,
                        help="models/Isolation_Forest/iforest_ensemble_outputs.zip (the team's Isolation Forest).")
    parser.add_argument("--transformer-scores", type=pathlib.Path, default=None)
    parser.add_argument("--transformer-column", default="tr_topk_ratio",
                        choices=["tr_topk_percentile", "tr_mean", "tr_topk_ratio"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, per_rows = [], []
    for seed in args.seeds:
        if args.teammate_iforest_zip is not None:
            forest = read_teammate_iforest(args.teammate_iforest_zip, seed)
        elif args.iforest_dir is not None:
            forest = pd.read_csv(args.iforest_dir / f"seed{seed}" / "iforest_hourly_scores.csv")
        else:
            parser.error("give --teammate-iforest-zip or --iforest-dir")
        frame = load_scores(args.xgb_dir / f"seed{seed}" / "xgb_hourly_scores.csv", forest,
                            args.transformer_scores, args.transformer_column)
        for norm in ("percentile", "ratio"):
            seed_per = []
            for result in run(frame, DETECTORS, norm, seed_per):
                rows.append({"seed": seed, **result})
            per_rows.extend({"seed": seed, **r} for r in seed_per)
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "all_runs.csv", index=False)
    # Per attack, for every seed / scale / detector set / fusion rule.
    pd.DataFrame(per_rows).to_csv(args.out_dir / "per_attack.csv", index=False)
    keys = ["norm", "detectors", "fusion"]
    metrics = ["detected", "mean_delay", "recall", "tnr", "val_alarm_hours_per_week", "d2_clean_alarm_fraction", "d2_clean_episodes_per_week",
               "recall_schedule", "roc_auc", "pr_auc", "roc_auc_schedule", "s_score_approx"]
    summary = table.groupby(keys, sort=False)[metrics].mean().round(3)
    summary.to_csv(args.out_dir / "summary.csv")
    pd.set_option("display.width", 200)
    print(summary.to_string())


if __name__ == "__main__":
    main()
