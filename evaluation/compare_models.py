"""Compare every detector on Dataset 2 with one shared metric implementation.

Each model keeps its own hourly score and alarm (threshold set on its own
calibration data). This script only re-counts those alarms the same way for
every model, so the numbers in the table are directly comparable.

Standard library only. Run from the repo root:

    python evaluation/compare_models.py                       # shared settings
    python evaluation/compare_models.py --merge-gap 0 --grace 0
"""

import argparse
import csv
import io
import json
import statistics
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/processed/dataset2_eval.csv"
WINDOWS = ROOT / "data/attack_windows_dataset2.csv"
TRANSFORMER = ROOT / "models/Transformer/Notebook/inference_results/hourly_scores.csv"
RIDGE = ROOT / "models/Ridge/ridge_scores.csv"
IFOREST_ZIP = ROOT / "models/Isolation_Forest/iforest_results.zip"
OUT = ROOT / "evaluation/results"


def hour_key(text):
    """'2016-07-04 11', '2016-07-04 11:00:00' -> datetime at the hour."""
    return datetime.strptime(text.strip()[:13], "%Y-%m-%d %H")


# ---------------------------------------------------------------- loading

def load_labels():
    labels = {}
    with open(LABELS, newline="") as f:
        for row in csv.DictReader(f):
            labels[hour_key(row["DATETIME"])] = int(float(row["label"]))
    return labels


def load_windows():
    windows = []
    with open(WINDOWS, newline="") as f:
        for row in csv.DictReader(f):
            windows.append({
                "id": int(row["attack_id"]),
                "start": hour_key(row["real_start"]),
                "end": hour_key(row["real_end"]),
                "hours": int(float(row["real_hours"])),
            })
    return windows


def read_scores(rows, time_col, score_col, alarm_col):
    out = {}
    for row in rows:
        out[hour_key(row[time_col])] = (float(row[score_col]), int(float(row[alarm_col])))
    return out


def load_models():
    """Returns {model name: [run, ...]}; each run is {hour: (score, alarm)}."""
    models = {}

    with open(TRANSFORMER, newline="") as f:
        models["Transformer AE"] = [read_scores(
            csv.DictReader(f), "datetime", "topk_percentile_score", "topk_percentile_alarm")]

    with open(RIDGE, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "dataset2"]
    models["Ridge"] = [read_scores(rows, "DATETIME", "score", "alarm")]

    runs = []
    with zipfile.ZipFile(IFOREST_ZIP) as z:
        for seed in ("seed0", "seed1", "seed2"):
            with z.open(f"{seed}/hourly_scores.csv") as f:
                reader = csv.DictReader(io.TextIOWrapper(f, newline=""))
                runs.append(read_scores(reader, "DATETIME", "score", "alarm"))
    models["Isolation Forest"] = runs

    return models


# ---------------------------------------------------------------- metrics

def roc_auc(scores, labels):
    """Mann-Whitney U with average ranks for ties."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    rank_sum = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def average_precision(scores, labels):
    """Same definition as sklearn.metrics.average_precision_score."""
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    pos = sum(labels)
    tp = fp = 0
    ap = prev_recall = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        recall = tp / pos
        ap += (recall - prev_recall) * tp / (tp + fp)
        prev_recall = recall
        i = j
    return ap


def count_episodes(alarms, merge_gap):
    """Runs of alarm hours; runs separated by <= merge_gap quiet hours count once."""
    episodes = 0
    quiet = None  # quiet hours since the last alarm (None = no alarm yet)
    for a in alarms:
        if a:
            if quiet is None or quiet > merge_gap:
                episodes += 1
            quiet = 0
        elif quiet is not None:
            quiet += 1
    return episodes


def evaluate(run, hours, labels, windows, merge_gap, grace):
    grace_hours = set()
    for w in windows:
        for k in range(1, grace + 1):
            grace_hours.add(w["end"] + timedelta(hours=k))

    tp = fp = fn = tn = 0
    scores, ys, clean_alarms = [], [], []
    for h in hours:
        score, alarm = run[h]
        y = labels[h]
        if y == 0 and h in grace_hours:
            clean_alarms.append(0)   # left out, and breaks no episode
            continue
        scores.append(score)
        ys.append(y)
        if y == 1:
            tp += alarm
            fn += 1 - alarm
            clean_alarms.append(0)
        else:
            fp += alarm
            tn += 1 - alarm
            clean_alarms.append(alarm)

    per_attack = []
    for w in windows:
        h, first = w["start"], None
        while h <= w["end"]:
            if h in run and run[h][1] == 1:
                first = h
                break
            h += timedelta(hours=1)
        delay = None if first is None else (first - w["start"]).total_seconds() / 3600
        per_attack.append({"attack": w["id"], "hours": w["hours"],
                           "detected": first is not None, "delay_hours": delay})

    detected = [a for a in per_attack if a["detected"]]
    ttd_ratios = [a["delay_hours"] / a["hours"] if a["detected"] else 1.0 for a in per_attack]
    clean = fp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    tnr = tn / clean
    episodes = count_episodes(clean_alarms, merge_gap)
    s_ttd = 1 - statistics.mean(ttd_ratios)
    s_clf = (recall + tnr) / 2

    return {
        "scored_hours": len(hours),
        "attacks_detected": len(detected),
        "missed_attacks": ",".join(str(a["attack"]) for a in per_attack if not a["detected"]),
        "mean_delay_hours": statistics.mean(a["delay_hours"] for a in detected) if detected else None,
        "median_delay_hours": statistics.median(a["delay_hours"] for a in detected) if detected else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tnr": tnr,
        "roc_auc": roc_auc(scores, ys),
        "pr_auc": average_precision(scores, ys),
        "false_alarm_episodes": episodes,
        "episodes_per_week": episodes / (clean / 168),
        "s_ttd": s_ttd,
        "s_clf": s_clf,
        "s_score": 0.5 * s_ttd + 0.5 * s_clf,
    }, per_attack


# ---------------------------------------------------------------- main

COLUMNS = ["attacks_detected", "missed_attacks", "mean_delay_hours", "median_delay_hours",
           "precision", "recall", "f1", "tnr", "roc_auc", "pr_auc",
           "false_alarm_episodes", "episodes_per_week", "s_ttd", "s_clf", "s_score"]


def summarise(results):
    """Mean over runs (seeds); also the std when there is more than one run."""
    row, spread = {}, {}
    for c in COLUMNS:
        vals = [r[c] for r in results]
        if c == "missed_attacks":
            row[c] = " | ".join(v or "-" for v in vals) if len(set(vals)) > 1 else (vals[0] or "")
        elif any(v is None for v in vals):
            row[c] = None
        else:
            row[c] = statistics.mean(vals)
            if len(vals) > 1:
                spread[c] = statistics.stdev(vals)
    return row, spread


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--merge-gap", type=int, default=2, help="quiet hours that still join two alarm runs")
    ap.add_argument("--grace", type=int, default=3, help="hours after each attack left out of scoring")
    ap.add_argument("--all-hours", action="store_true",
                    help="score each model on all its own hours instead of the hours every model scored")
    args = ap.parse_args()

    labels = load_labels()
    windows = load_windows()
    models = load_models()

    common = set(labels)
    for runs in models.values():
        for run in runs:
            common &= set(run)

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"gap{args.merge_gap}_grace{args.grace}" + ("_allhours" if args.all_hours else "")
    summary_rows, attack_rows = [], []

    for name, runs in models.items():
        results = []
        for i, run in enumerate(runs):
            hours = sorted(set(run) & set(labels)) if args.all_hours else sorted(common)
            metrics, per_attack = evaluate(run, hours, labels, windows, args.merge_gap, args.grace)
            results.append(metrics)
            for a in per_attack:
                attack_rows.append({"model": name, "run": i, **a})
        row, spread = summarise(results)
        summary_rows.append({"model": name, "runs": len(runs), "scored_hours": results[0]["scored_hours"],
                             **row, **{f"{k}_std": v for k, v in spread.items()}})

    std_cols = sorted({k for r in summary_rows for k in r if k.endswith("_std")})
    fields = ["model", "runs", "scored_hours"] + COLUMNS + std_cols
    with open(OUT / f"comparison_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)
    with open(OUT / f"per_attack_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "run", "attack", "hours", "detected", "delay_hours"])
        w.writeheader()
        w.writerows(attack_rows)

    fmt = lambda v, d=3: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    print(f"merge gap {args.merge_gap} h, grace {args.grace} h, "
          f"{'own hours per model' if args.all_hours else f'{len(common)} common hours'}\n")
    head = ["model", "att", "delay", "P", "R", "F1", "ROC", "PR", "FA/wk", "S_TTD", "S_CLF", "S"]
    print("  ".join(f"{h:>8}" if i else f"{h:<17}" for i, h in enumerate(head)))
    for r in summary_rows:
        vals = [fmt(r["attacks_detected"], 1), fmt(r["mean_delay_hours"], 1), fmt(r["precision"]),
                fmt(r["recall"]), fmt(r["f1"]), fmt(r["roc_auc"]), fmt(r["pr_auc"]),
                fmt(r["episodes_per_week"], 2), fmt(r["s_ttd"]), fmt(r["s_clf"]), fmt(r["s_score"])]
        print(f"{r['model']:<17}  " + "  ".join(f"{v:>8}" for v in vals))
    print(f"\nwrote {OUT.relative_to(ROOT)}/comparison_{tag}.csv and per_attack_{tag}.csv")
    print(json.dumps({"settings": vars(args)}))


if __name__ == "__main__":
    main()
