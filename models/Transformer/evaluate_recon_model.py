import os
import csv
import json
import pathlib
import logging
import argparse

import numpy as np

import torch

import matplotlib
matplotlib.use("Agg")  # Write image files without needing a display.
import matplotlib.pyplot as plt

from model import Transformer_Recon_Model

from batadal_dataset import build_dataloaders, DATASET2_FILE

"""
Scores, calibrates and evaluates a FROZEN Transformer reconstruction model.

    python evaluate_recon_model.py \
        --dataset-path data/processed \
        --model-checkpoint runs/recon_24h/best_model.pt \
        --out-dir runs/recon_24h/evaluation

Dataset 2 is read from data/processed/dataset2_eval.csv (Pipeline.ipynb step
8), which carries the labels: `label` is 1 inside the 7 real attack windows and
0 elsewhere, and `attack_id` names the attack. Pass --dataset2-path only if the
file is somewhere else.

Steps, following the proposal:
1. Scoring.     Squared reconstruction error per hour and per sensor, taken at
                the LAST hour of each window, on the calibration block, the
                validation block and Dataset 2.
2. Calibration. On the calibration block of Dataset 1 only (clean data the
                model never trained on): per-sensor references, an hourly
                score, and the lowest alarm threshold inside the false-alarm
                budget (at most 1 alarm episode per week and 1% of hours).
                Pass --calibration to use a frozen calibration.pt from
                calibrate_recon_model.py instead of recalibrating.
3. Evaluation.  On Dataset 2: attacks detected, detection delay from the real
                attack start, false alarms, precision, recall, F1, ROC-AUC,
                PR-AUC and the BATADAL S-score. No point adjustment.
4. Plots.       Score timeline, per-attack sensor contributions, the largest
                false alarms, and detection delay.

The Test set is never read here.
"""

HOURS_PER_WEEK = 24 * 7  # 168

# Three hourly scores are calibrated and evaluated side by side.
#   topk_percentile - the proposal's score: each sensor's error as a percentile
#                     of its calibration errors, mean of the top-k sensors.
#   mean            - the proposal's comparison: plain average error.
#   topk_ratio      - each sensor's error divided by its own 99th-percentile
#                     calibration error, mean of the top-k sensors. Like the
#                     percentile score, but it has no ceiling at 1.
SCORE_TYPES = ["topk_percentile", "mean", "topk_ratio"]

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SHADE = "#ecebe7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.:
        raise argparse.ArgumentTypeError("%r not in range > 0."%(x,))
    return x


def load_model(checkpoint_path):
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        logging.error(f"Could not load checkpoint {checkpoint_path}: {e}")
        return False, None
    return True, checkpoint_dict


def scaler_matches(scaler, classifier_dict):
    """
    True when the data's scaler agrees with the one stored in the checkpoint.
    Checkpoints trained before the shared data pipeline stored values for all
    43 raw columns plus a keep mask; those are compared on the kept columns.
    """
    saved_min = np.asarray(classifier_dict["scaler_min"], dtype=np.float64)
    saved_span = np.asarray(classifier_dict["scaler_span"], dtype=np.float64)
    if len(saved_min) != len(scaler.min_):
        keep = np.asarray(classifier_dict["scaler_keep"], dtype=bool)
        saved_min, saved_span = saved_min[keep], saved_span[keep]
    if len(saved_min) != len(scaler.min_):
        return False
    return bool(np.allclose(scaler.min_, saved_min, rtol=1e-4, atol=1e-6)
                and np.allclose(scaler.span_, saved_span, rtol=1e-4, atol=1e-6))


# ----------------------------- 1. Scoring -----------------------------------
@torch.no_grad()
def sensor_errors(model, dataloader, device):
    """
    Squared reconstruction error of the LAST hour of every window, per sensor.
    Returns (errors (Hours,Dim_in), rows (Hours,)). Using only the last hour
    keeps the score causal: hour t is scored from hours up to t.
    """
    model.eval()

    errors_list, rows_list = [], []
    for windows, targets, _, rows in dataloader:
        # The loader returns one row per window: the position of its last
        # hour. A loader that returns one row per hour of the window would
        # break the hour-to-score mapping below, so stop here instead.
        if rows.dim() != 1:
            raise ValueError("Expected one row index per window (the last-hour loader).")
        windows = windows.to(device)  # (N,Seq_window,Dim_in)
        targets = targets.to(device)  # (N,Seq_window,Dim_in)

        recon_windows = model(windows)

        errors = (recon_windows - targets) ** 2  # (N,Seq_window,Dim_in)
        errors_list.append(errors[:, -1, :].cpu())  # (N,Dim_in)
        rows_list.append(rows)
    return torch.cat(errors_list).numpy().astype(np.float64), torch.cat(rows_list).numpy()


# ----------------------------- 2. Calibration -------------------------------
def to_percentiles(errors, sorted_calib_errors):
    """Share of calibration errors that each error is >= to, per sensor (0 to 1)."""
    percentiles = np.empty_like(errors, dtype=np.float64)
    for j in range(errors.shape[1]):
        reference = sorted_calib_errors[:, j]
        percentiles[:, j] = np.searchsorted(reference, errors[:, j], side="right") / len(reference)
    return percentiles


def to_ratios(errors, calib_q99):
    """Each sensor's error divided by its own 99th-percentile calibration error."""
    return errors / np.maximum(calib_q99, 1e-12)


def hourly_scores(errors, reference, score_type, top_k):
    """One anomaly score per hour."""
    if score_type == "mean":
        return errors.mean(axis=1)
    if score_type == "topk_percentile":
        per_sensor = to_percentiles(errors, reference["sorted_calib_errors"])
    elif score_type == "topk_ratio":
        per_sensor = to_ratios(errors, reference["calib_q99"])
    else:
        raise ValueError(f"unknown score_type {score_type!r}")
    return np.sort(per_sensor, axis=1)[:, -top_k:].mean(axis=1)


def episode_spans(alarms, merge_gap_hours):
    """
    Alarm episodes as (first hour, last hour) index pairs. Back-to-back alarm
    hours are one episode, and two runs separated by at most merge_gap_hours
    quiet hours are merged into one.
    """
    alarm_hours = np.flatnonzero(alarms)
    if len(alarm_hours) == 0:
        return []
    breaks = np.flatnonzero(np.diff(alarm_hours) > merge_gap_hours + 1)
    starts = np.concatenate([[alarm_hours[0]], alarm_hours[breaks + 1]])
    ends = np.concatenate([alarm_hours[breaks], [alarm_hours[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def clean_block_stats(scores, threshold, merge_gap_hours):
    """False-alarm figures on a block of clean (attack-free) hours."""
    alarms = scores > threshold
    episodes = len(episode_spans(alarms, merge_gap_hours))
    return {
        "hours": int(len(scores)),
        "alarm_hours": int(alarms.sum()),
        "alarm_fraction": float(alarms.mean()),
        "episodes": episodes,
        "episodes_per_week": float(episodes / (len(scores) / HOURS_PER_WEEK))}


def find_threshold(scores, max_alarm_fraction, max_episodes_per_week, merge_gap_hours):
    """
    Lowest threshold whose alarms (score > threshold) stay inside BOTH budgets
    on the calibration scores. Candidates are tried from the highest score down
    and the search stops at the first one that breaks a budget, so every
    threshold above the chosen one is inside the budget as well.
    """
    candidates = np.unique(scores)[::-1]  # Highest first: zero alarms.
    threshold = candidates[0]
    for candidate in candidates:
        stats = clean_block_stats(scores, candidate, merge_gap_hours)
        if stats["alarm_fraction"] > max_alarm_fraction:
            break
        if stats["episodes_per_week"] > max_episodes_per_week:
            break
        threshold = candidate
    return float(threshold)


def calibrate(calib_errors, val_errors, top_k=3, max_alarm_fraction=0.01,
              max_episodes_per_week=1.0, merge_gap_hours=0):
    """
    Everything calibration produces, from per-sensor errors on clean data:
    the per-sensor references, and for each hourly score its alarm threshold
    and the false-alarm figures on the calibration and validation blocks.
    """
    # Per-sensor references, from the calibration block only.
    reference = {
        "sorted_calib_errors": np.sort(calib_errors, axis=0),
        "calib_q99": np.percentile(calib_errors, 99, axis=0)}

    calibration = {
        "reference": reference,
        "top_k": top_k,
        "max_alarm_fraction": max_alarm_fraction,
        "max_episodes_per_week": max_episodes_per_week,
        "merge_gap_hours": merge_gap_hours,
        "thresholds": {}, "calib_stats": {}, "val_stats": {}, "val_in_budget": {},
        "calib_scores": {}, "val_scores": {}}
    for score_type in SCORE_TYPES:
        calib_scores = hourly_scores(calib_errors, reference, score_type, top_k)
        val_scores = hourly_scores(val_errors, reference, score_type, top_k)
        threshold = find_threshold(
            scores=calib_scores,
            max_alarm_fraction=max_alarm_fraction,
            max_episodes_per_week=max_episodes_per_week,
            merge_gap_hours=merge_gap_hours)
        val_stats = clean_block_stats(val_scores, threshold, merge_gap_hours)

        calibration["thresholds"][score_type] = threshold
        calibration["calib_stats"][score_type] = clean_block_stats(calib_scores, threshold, merge_gap_hours)
        calibration["val_stats"][score_type] = val_stats
        # Transfer check. Slightly optimistic: validation chose the checkpoint.
        calibration["val_in_budget"][score_type] = bool(
            val_stats["alarm_fraction"] <= max_alarm_fraction
            and val_stats["episodes_per_week"] <= max_episodes_per_week)
        calibration["calib_scores"][score_type] = calib_scores
        calibration["val_scores"][score_type] = val_scores
    return calibration


def save_calibration(calibration, out_dir, extra):
    """calibration.pt (everything needed to score new data) and a flat calibration.json."""
    saved = {
        "top_k": calibration["top_k"],
        "max_alarm_fraction": calibration["max_alarm_fraction"],
        "max_episodes_per_week": calibration["max_episodes_per_week"],
        "merge_gap_hours": calibration["merge_gap_hours"],
        "thresholds": calibration["thresholds"],
        "calib_stats": calibration["calib_stats"],
        "val_stats": calibration["val_stats"],
        "val_in_budget": calibration["val_in_budget"],
        "sorted_calib_errors": torch.from_numpy(calibration["reference"]["sorted_calib_errors"]),
        "calib_q99": torch.from_numpy(calibration["reference"]["calib_q99"])}
    saved.update(extra)
    torch.save(saved, os.path.join(out_dir, "calibration.pt"))

    flat = {key: value for key, value in extra.items() if not isinstance(value, (list, dict))}
    for key in ("top_k", "max_alarm_fraction", "max_episodes_per_week", "merge_gap_hours"):
        flat[key] = calibration[key]
    for score_type in SCORE_TYPES:
        flat[f"{score_type}_threshold"] = calibration["thresholds"][score_type]
        flat[f"{score_type}_calib_alarm_fraction"] = calibration["calib_stats"][score_type]["alarm_fraction"]
        flat[f"{score_type}_calib_episodes_per_week"] = calibration["calib_stats"][score_type]["episodes_per_week"]
        flat[f"{score_type}_val_alarm_fraction"] = calibration["val_stats"][score_type]["alarm_fraction"]
        flat[f"{score_type}_val_episodes_per_week"] = calibration["val_stats"][score_type]["episodes_per_week"]
        flat[f"{score_type}_val_in_budget"] = calibration["val_in_budget"][score_type]
    with open(os.path.join(out_dir, "calibration.json"), "w") as summary_f:
        json.dump(flat, summary_f, indent=2)


def load_calibration(calibration_path):
    """Read calibration.pt back into the form calibrate() returns."""
    saved = torch.load(calibration_path, map_location="cpu")
    calibration = {key: saved[key] for key in (
        "top_k", "max_alarm_fraction", "max_episodes_per_week", "merge_gap_hours",
        "thresholds", "calib_stats", "val_stats", "val_in_budget")}
    calibration["reference"] = {
        "sorted_calib_errors": saved["sorted_calib_errors"].numpy(),
        "calib_q99": saved["calib_q99"].numpy()}
    calibration["calib_scores"], calibration["val_scores"] = {}, {}
    return calibration, saved


# ----------------------------- 3. Evaluation --------------------------------
def roc_auc(scores, labels):
    """Area under the ROC curve (ties share their average rank)."""
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    average_rank = (last_rank - (counts - 1) / 2.0)[inverse]
    return float((average_rank[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(scores, labels):
    """Area under the precision-recall curve (average precision)."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    # One point per distinct score value, so tied hours are taken together.
    last_of_value = np.flatnonzero(np.append(np.diff(sorted_scores) != 0, True))
    true_positives = np.cumsum(sorted_labels == 1)[last_of_value]
    precision = true_positives / (last_of_value + 1)
    recall = true_positives / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def evaluate_scores(scores, threshold, labels, attack_id, attacks, ignore, merge_gap_hours):
    """
    All proposal metrics for one hourly score on Dataset 2.

    scores, labels, attack_id, ignore are aligned, one entry per scored hour.
    attacks is a list of {"id", "start", "hours"} with start as an index into
    those arrays. Hours with label 1 are attack hours, label 0 clean hours,
    anything else (unknown) is left out of the hour-by-hour figures. `ignore`
    marks the grace hours after an attack, also left out.
    """
    alarms = scores > threshold
    is_attack = (labels == 1) & ~ignore
    is_clean = (labels == 0) & ~ignore

    # Attacks detected and detection delay, measured from the REAL start.
    per_attack, ttd_ratios = [], []
    for attack in attacks:
        inside = np.flatnonzero((attack_id == attack["id"]) & alarms)
        detected = len(inside) > 0
        delay = int(inside[0] - attack["start"]) if detected else None
        per_attack.append({"id": attack["id"], "detected": detected, "delay_hours": delay})
        # BATADAL: an attack that is not detected counts its full duration.
        ttd_ratios.append((delay if detected else attack["hours"]) / attack["hours"])

    # Hour-by-hour confusion counts. No point adjustment: one detected hour
    # credits one hour, never the whole attack.
    tp = int((alarms & is_attack).sum())
    fn = int((~alarms & is_attack).sum())
    fp = int((alarms & is_clean).sum())
    tn = int((~alarms & is_clean).sum())

    def ratio(a, b):
        return float(a / b) if b > 0 else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    tnr = ratio(tn, tn + fp)

    # False alarms: episodes among the clean hours only.
    clean_hours = fp + tn
    false_episodes = len(episode_spans(alarms & is_clean, merge_gap_hours))
    episodes_per_week = ratio(false_episodes, clean_hours / HOURS_PER_WEEK)
    clean_alarm_fraction = ratio(fp, clean_hours)

    # BATADAL ranking score: S = 0.5 * S_TTD + 0.5 * S_CLF.
    s_ttd = float(1.0 - np.mean(ttd_ratios)) if len(ttd_ratios) > 0 else None
    s_clf = (recall + tnr) / 2.0 if recall is not None and tnr is not None else None
    s_score = 0.5 * s_ttd + 0.5 * s_clf if s_ttd is not None and s_clf is not None else None

    known = is_attack | is_clean
    delays = [a["delay_hours"] for a in per_attack if a["detected"]]
    metrics = {
        "threshold": float(threshold),
        "attacks_detected": int(sum(a["detected"] for a in per_attack)),
        "attacks_total": len(per_attack),
        "missed_attacks": ",".join(str(a["id"]) for a in per_attack if not a["detected"]),
        "mean_delay_hours": float(np.mean(delays)) if len(delays) > 0 else None,
        "median_delay_hours": float(np.median(delays)) if len(delays) > 0 else None,
        "false_alarm_episodes": false_episodes,
        "false_alarm_episodes_per_week": episodes_per_week,
        "clean_hours_under_alarm": clean_alarm_fraction,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative_rate": tnr,
        "roc_auc": roc_auc(scores[known], labels[known]),
        "pr_auc": pr_auc(scores[known], labels[known]),
        "s_ttd": s_ttd,
        "s_clf": s_clf,
        "s_score": s_score}
    return metrics, per_attack, alarms


# ----------------------------- 4. Plots -------------------------------------
def style_axis(axis):
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)


def shade_attacks(axis, when, attacks, label_attacks=True):
    for attack in attacks:
        start = when[attack["start"]]
        end = when[min(attack["start"] + attack["hours"] - 1, len(when) - 1)]
        axis.axvspan(start, end, color=SHADE, zorder=0)
        if label_attacks:
            axis.annotate(
                str(attack["id"]), (start, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(2, -11), fontsize=8, color=MUTED)


def plot_calibration(calib_scores, val_scores, threshold, score_type, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharey=True)
    for axis, scores, name in zip(axes, (calib_scores, val_scores), ("calibration", "validation")):
        axis.plot(scores, color=SERIES[0], linewidth=0.7, label=f"{name} score")
        axis.axhline(threshold, color=INK, linestyle="--", linewidth=1, label="alarm threshold")
        axis.set_ylabel("anomaly score", color=INK, fontsize=10)
        axis.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=MUTED)
        style_axis(axis)
    axes[-1].set_xlabel("hour in block", color=INK, fontsize=10)
    fig.suptitle(f"Clean data against the threshold ({score_type})",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, f"calibration_{score_type}.png"), dpi=150)
    plt.close(fig)


def plot_timeline(when, scores, threshold, alarms, attack_id, attacks, score_type, out_dir):
    """Dataset 2 hourly score, the threshold, the attack windows and the alarms."""
    fig, axis = plt.subplots(figsize=(13, 4))
    shade_attacks(axis, when, attacks)
    axis.plot(when, scores, color=SERIES[0], linewidth=0.8, label="hourly score", zorder=2)
    axis.axhline(threshold, color=INK, linestyle="--", linewidth=1, label="alarm threshold", zorder=3)

    true_alarms = alarms & (attack_id > 0)
    false_alarms = alarms & (attack_id == 0)
    # Shape as well as colour separates the two kinds of alarm.
    axis.scatter(when[true_alarms], scores[true_alarms], s=14, color=SERIES[0],
                 edgecolors="white", linewidths=0.4, label="alarm inside an attack", zorder=4)
    axis.scatter(when[false_alarms], scores[false_alarms], s=30, color=SERIES[1], marker="^",
                 edgecolors="white", linewidths=0.4, label="alarm outside any attack", zorder=5)

    if score_type != "topk_percentile":
        axis.set_yscale("log")
    axis.set_ylabel("anomaly score", color=INK, fontsize=10)
    axis.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False,
                labelcolor=MUTED, ncol=4)
    style_axis(axis)
    fig.suptitle(f"Dataset 2: {score_type} score (shaded: the 7 real attack windows, numbered)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, f"timeline_{score_type}.png"), dpi=150)
    plt.close(fig)


def plot_event(when, scores, threshold, ratios, feature_names, attacks,
               first, last, title, file_name, out_dir, score_type, top_sensors, pad_hours=24):
    """
    Close-up of one attack or false alarm: the score on top, and below it the
    sensor-level contributions (each sensor's error divided by its own normal
    99th-percentile error), for the sensors with the highest error in the event.
    """
    lo, hi = max(0, first - pad_hours), min(len(scores) - 1, last + pad_hours)
    span = slice(lo, hi + 1)

    order = np.argsort(ratios[first:last + 1].mean(axis=0))[::-1][:top_sensors]
    panel = ratios[span][:, order].T  # (Sensors,Hours)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2]})

    x = np.arange(lo, hi + 1)
    top.axvspan(first - 0.5, last + 0.5, color=SHADE, zorder=0)
    top.plot(x, scores[span], color=SERIES[0], linewidth=1.5, label=f"{score_type} score")
    top.axhline(threshold, color=INK, linestyle="--", linewidth=1, label="alarm threshold")
    if score_type != "topk_percentile":
        top.set_yscale("log")
    top.set_ylabel("anomaly score", color=INK, fontsize=10)
    top.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False,
               labelcolor=MUTED, ncol=2)
    style_axis(top)

    # One hue, light to dark. Values are capped so one extreme sensor does not
    # wash out the rest; the cap is stated on the colour bar.
    cap = max(float(np.percentile(panel, 99)), 1.0)
    image = bottom.imshow(
        panel, aspect="auto", cmap="Blues", vmin=0.0, vmax=cap,
        extent=(lo - 0.5, hi + 0.5, len(order) - 0.5, -0.5), interpolation="nearest")
    bottom.axvline(first - 0.5, color=INK, linewidth=0.8)
    bottom.axvline(last + 0.5, color=INK, linewidth=0.8)
    bottom.set_yticks(np.arange(len(order)))
    bottom.set_yticklabels([feature_names[j] for j in order], fontsize=8, color=MUTED)
    ticks = np.arange(lo, hi + 1, max(1, (hi - lo) // 8))
    bottom.set_xticks(ticks)
    bottom.set_xticklabels(
        [np.datetime_as_string(when[t], unit="h").replace("T", " ") for t in ticks],
        fontsize=8, color=MUTED, rotation=20, ha="right")
    bottom.tick_params(length=0)
    for side in bottom.spines.values():
        side.set_visible(False)
    bar = fig.colorbar(image, ax=[top, bottom], fraction=0.025, pad=0.015)
    bar.set_label(f"sensor error / its normal 99th percentile (capped at {cap:.3g})",
                  fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.outline.set_visible(False)

    fig.suptitle(title, x=0.01, y=0.97, ha="left", fontsize=11, color=INK)
    fig.savefig(os.path.join(out_dir, file_name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_sensors(ratios, attack_id, attacks, feature_names, out_dir, top_n=5):
    """The sensors with the highest mean error during each attack."""
    fig, axes = plt.subplots(1, len(attacks), figsize=(2.9 * len(attacks), 3.0), squeeze=False)
    for axis, attack in zip(axes[0], attacks):
        mean_ratio = ratios[attack_id == attack["id"]].mean(axis=0)
        order = np.argsort(mean_ratio)[::-1][:top_n][::-1]
        axis.barh(np.arange(len(order)), mean_ratio[order], height=0.6, color=SERIES[0])
        axis.set_yticks(np.arange(len(order)))
        axis.set_yticklabels([feature_names[j] for j in order], fontsize=8, color=MUTED)
        axis.set_title(f"attack {attack['id']}", loc="left", fontsize=10, color=INK)
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.suptitle(
        "Sensors with the highest error in each attack (mean error / normal 99th percentile; "
        "scales differ between panels)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out_dir, "top_sensors_per_attack.png"), dpi=150)
    plt.close(fig)


def plot_delays(per_attack_by_score, attacks, out_dir):
    """Detection delay per attack for each score; missed attacks are written out."""
    fig, axis = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / len(per_attack_by_score)
    positions = np.arange(len(attacks))
    longest = max([a["delay_hours"] or 0 for rows in per_attack_by_score.values() for a in rows] + [1])

    for index, (score_type, rows) in enumerate(per_attack_by_score.items()):
        offset = (index - (len(per_attack_by_score) - 1) / 2.0) * width
        delays = [a["delay_hours"] if a["detected"] else 0 for a in rows]
        axis.bar(positions + offset, delays, width=width * 0.9, color=SERIES[index], label=score_type)
        for position, attack_row in zip(positions, rows):
            text = str(attack_row["delay_hours"]) if attack_row["detected"] else "missed"
            height = attack_row["delay_hours"] if attack_row["detected"] else 0
            axis.annotate(
                text, (position + offset, height), textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7, color=MUTED,
                rotation=0 if attack_row["detected"] else 90)

    axis.set_xticks(positions)
    axis.set_xticklabels([f"attack {a['id']}\n({a['hours']} h)" for a in attacks])
    axis.set_ylim(0, longest * 1.25)
    axis.set_ylabel("hours from real start to first alarm", color=INK, fontsize=10)
    axis.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), fontsize=8, frameon=False,
                labelcolor=MUTED, ncol=len(per_attack_by_score))
    style_axis(axis)
    fig.suptitle("Detection delay per attack", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "detection_delay.png"), dpi=150)
    plt.close(fig)


# ----------------------------- Everything after the model -------------------
def run_evaluation(calibration, dev_errors, dev_when, dev_labels, dev_attack_id,
                   feature_names, out_dir, grace_hours=0, top_sensors=12, log=print):
    """
    Evaluation on Dataset 2, plots and result files, from per-sensor errors and
    a calibration (from calibrate() or load_calibration()). Takes plain arrays,
    so it works for any detector that can produce them. dev_* arrays are
    aligned with dev_errors, one entry per scored hour.
    """
    reference = calibration["reference"]
    top_k = calibration["top_k"]
    max_alarm_fraction = calibration["max_alarm_fraction"]
    max_episodes_per_week = calibration["max_episodes_per_week"]
    merge_gap_hours = calibration["merge_gap_hours"]

    # The attacks present among the scored hours, from the real windows.
    attacks = []
    for aid in sorted(int(a) for a in np.unique(dev_attack_id) if a > 0):
        hours = np.flatnonzero(dev_attack_id == aid)
        attacks.append({"id": aid, "start": int(hours[0]), "hours": int(len(hours))})

    # Grace hours after each attack are left out of the hour-by-hour figures.
    ignore = np.zeros(len(dev_labels), dtype=bool)
    for attack in attacks:
        end = attack["start"] + attack["hours"]
        ignore[end:end + grace_hours] = True
    ignore &= dev_attack_id == 0

    dev_ratios = to_ratios(dev_errors, reference["calib_q99"])

    all_metrics, per_attack_by_score, hourly = {}, {}, {}
    for score_type in SCORE_TYPES:
        dev_scores = hourly_scores(dev_errors, reference, score_type, top_k)

        # The threshold is fixed by calibration and is not adjusted here.
        threshold = calibration["thresholds"][score_type]
        calib_stats = calibration["calib_stats"][score_type]
        val_stats = calibration["val_stats"][score_type]

        metrics, per_attack, alarms = evaluate_scores(
            scores=dev_scores,
            threshold=threshold,
            labels=dev_labels,
            attack_id=dev_attack_id,
            attacks=attacks,
            ignore=ignore,
            merge_gap_hours=merge_gap_hours)

        metrics["calib_alarm_fraction"] = calib_stats["alarm_fraction"]
        metrics["calib_episodes_per_week"] = calib_stats["episodes_per_week"]
        metrics["val_alarm_fraction"] = val_stats["alarm_fraction"]
        metrics["val_episodes_per_week"] = val_stats["episodes_per_week"]
        metrics["val_in_budget"] = calibration["val_in_budget"][score_type]
        week, fraction = metrics["false_alarm_episodes_per_week"], metrics["clean_hours_under_alarm"]
        metrics["dev_in_budget"] = None if week is None else bool(
            week <= max_episodes_per_week and fraction <= max_alarm_fraction)
        # A threshold at the highest calibration score can never be crossed by
        # a score that has a ceiling (the percentile score tops out at 1).
        metrics["threshold_at_ceiling"] = bool(
            score_type == "topk_percentile" and threshold >= 1.0)

        all_metrics[score_type] = metrics
        per_attack_by_score[score_type] = per_attack
        hourly[score_type] = (dev_scores, alarms)

        if score_type in calibration["calib_scores"]:
            plot_calibration(calibration["calib_scores"][score_type],
                             calibration["val_scores"][score_type],
                             threshold, score_type, out_dir)
        plot_timeline(dev_when, dev_scores, threshold, alarms, dev_attack_id, attacks,
                      score_type, out_dir)

    # Sensor-level contributions: one close-up per attack, shown with the
    # unbounded per-sensor score.
    close_up = "topk_ratio"
    scores, alarms = hourly[close_up]
    threshold = all_metrics[close_up]["threshold"]
    for attack, row in zip(attacks, per_attack_by_score[close_up]):
        outcome = f"detected after {row['delay_hours']} h" if row["detected"] else "MISSED"
        plot_event(
            dev_when, scores, threshold, dev_ratios, feature_names, attacks,
            first=attack["start"], last=attack["start"] + attack["hours"] - 1,
            title=f"Attack {attack['id']} ({attack['hours']} h, {outcome}): score and sensor contributions",
            file_name=f"attack_{attack['id']}.png", out_dir=out_dir,
            score_type=close_up, top_sensors=top_sensors)
    if len(attacks) > 0:
        plot_top_sensors(dev_ratios, dev_attack_id, attacks, feature_names, out_dir)
        plot_delays(per_attack_by_score, attacks, out_dir)

    # The three largest false alarms, to see which sensors caused them.
    is_clean = (dev_labels == 0) & ~ignore
    false_spans = episode_spans(alarms & is_clean, merge_gap_hours)
    false_spans = sorted(false_spans, key=lambda s: -scores[s[0]:s[1] + 1].max())[:3]
    for index, (first, last) in enumerate(false_spans, start=1):
        plot_event(
            dev_when, scores, threshold, dev_ratios, feature_names, attacks,
            first=first, last=last,
            title=f"False alarm {index} ({last - first + 1} h, outside every attack window): "
                  f"score and sensor contributions",
            file_name=f"false_alarm_{index}.png", out_dir=out_dir,
            score_type=close_up, top_sensors=top_sensors)

    # ---- Result files
    # Flat JSON, one "<score>_<metric>" key each, so run_ablations.py can collect it.
    flat = {
        "top_k": top_k,
        "merge_gap_hours": merge_gap_hours,
        "grace_hours": grace_hours,
        "max_alarm_fraction": max_alarm_fraction,
        "max_episodes_per_week": max_episodes_per_week,
        "scored_hours": int(len(dev_labels)),
        "attack_hours": int((dev_labels == 1).sum()),
        "clean_hours": int(is_clean.sum())}
    for score_type, metrics in all_metrics.items():
        for key, value in metrics.items():
            flat[f"{score_type}_{key}"] = value
    with open(os.path.join(out_dir, "metrics.json"), "w") as metrics_f:
        json.dump(flat, metrics_f, indent=2)

    with open(os.path.join(out_dir, "per_attack.csv"), "w", newline="") as attack_f:
        writer = csv.writer(attack_f)
        header = ["attack", "real_start", "hours"]
        for score_type in SCORE_TYPES:
            header += [f"{score_type}_detected", f"{score_type}_delay_hours"]
        writer.writerow(header)
        for index, attack in enumerate(attacks):
            line = [attack["id"],
                    np.datetime_as_string(dev_when[attack["start"]], unit="h").replace("T", " "),
                    attack["hours"]]
            for score_type in SCORE_TYPES:
                row = per_attack_by_score[score_type][index]
                line += [int(row["detected"]), "" if row["delay_hours"] is None else row["delay_hours"]]
            writer.writerow(line)

    with open(os.path.join(out_dir, "hourly_scores.csv"), "w", newline="") as hourly_f:
        writer = csv.writer(hourly_f)
        header = ["datetime", "label", "attack_id"]
        for score_type in SCORE_TYPES:
            header += [f"{score_type}_score", f"{score_type}_alarm"]
        writer.writerow(header)
        for t in range(len(dev_labels)):
            line = [np.datetime_as_string(dev_when[t], unit="h").replace("T", " "),
                    int(dev_labels[t]), int(dev_attack_id[t])]
            for score_type in SCORE_TYPES:
                line += [f"{hourly[score_type][0][t]:.6g}", int(hourly[score_type][1][t])]
            writer.writerow(line)

    # ---- Results table
    def show(value, digits=3):
        if value is None:
            return "n/a"
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)

    rows = [
        ("Attacks detected", lambda m: f"{m['attacks_detected']} of {m['attacks_total']}"),
        ("Missed attacks (Attack ID:)", lambda m: m["missed_attacks"] or "none"),
        ("Mean delay (h, detected only)", lambda m: show(m["mean_delay_hours"], 1)),
        ("False-alarm episodes per week", lambda m: show(m["false_alarm_episodes_per_week"], 2)),
        ("Clean hours under alarm", lambda m: "n/a" if m["clean_hours_under_alarm"] is None
            else f"{m['clean_hours_under_alarm']:.2%}"),
        ("Inside budget on Dataset 2", lambda m: show(m["dev_in_budget"])),
        ("Inside budget on validation", lambda m: show(m["val_in_budget"])),
        ("Precision", lambda m: show(m["precision"])),
        ("Recall", lambda m: show(m["recall"])),
        ("F1", lambda m: show(m["f1"])),
        ("ROC-AUC", lambda m: show(m["roc_auc"])),
        ("PR-AUC", lambda m: show(m["pr_auc"])),
        ("S-score", lambda m: show(m["s_score"])),
        ("  S_TTD", lambda m: show(m["s_ttd"])),
        ("  S_CLF", lambda m: show(m["s_clf"])),
        ("Threshold", lambda m: f"{m['threshold']:.5g}")]
    log("#" * 100)
    log(f"Results on Dataset 2 ({flat['scored_hours']:,} scored hours: "
        f"{flat['attack_hours']:,} attack, {flat['clean_hours']:,} clean)")
    log(f"{'':<32}" + "".join(f"{score_type:<20}" for score_type in SCORE_TYPES))
    for name, cell in rows:
        log(f"{name:<32}" + "".join(f"{cell(all_metrics[s]):<20}" for s in SCORE_TYPES))
    log("#" * 100)
    for score_type, metrics in all_metrics.items():
        if metrics["threshold_at_ceiling"]:
            log(f"NOTE: the {score_type} threshold sits at the score's ceiling of 1, "
                f"so that score cannot raise an alarm.")
        if metrics["dev_in_budget"] is False:
            log(f"NOTE: {score_type} is OUTSIDE the false-alarm budget on Dataset 2.")
    return flat


def main():
    project_name = "BATADAL Recon Model Evaluation"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset-path",
        help="Folder of processed files from Pipeline.ipynb (data/processed).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--dataset2-path",
        help="Processed Dataset 2 with labels. Default: dataset2_eval.csv in --dataset-path.",
        type=pathlib.Path,
        default=None)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to the frozen model checkpoint (e.g. best_model.pt).",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--calibration",
        help="calibration.pt from calibrate_recon_model.py. If given, its frozen "
             "thresholds and settings are used and nothing is recalibrated.",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "--batch-size",
        help="Batch size for scoring.",
        type=int,
        default=512)
    parser.add_argument(
        "--top-k",
        help="Number of highest sensor scores averaged in the top-k scores.",
        type=int,
        default=3)
    parser.add_argument(
        "--max-alarm-fraction",
        help="Budget: largest share of normal hours allowed under alarm.",
        type=restricted_float,
        default=0.01)
    parser.add_argument(
        "--max-episodes-per-week",
        help="Budget: most false alarm episodes allowed per normal week.",
        type=restricted_float,
        default=1.0)
    parser.add_argument(
        "--merge-gap-hours",
        help="Alarm runs separated by at most this many quiet hours are one episode.",
        type=int,
        default=0)
    parser.add_argument(
        "--grace-hours",
        help="Hours after each attack left out of the false-alarm and hourly figures.",
        type=int,
        default=0)
    parser.add_argument(
        "--top-sensors",
        help="Sensors shown in each close-up plot.",
        type=int,
        default=12)
    parser.add_argument(
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    dataset_path = args["dataset_path"]  # Dataset 1 path.
    dataset2_path = args["dataset2_path"]  # Dataset 2 path.
    if dataset2_path is None:
        dataset2_path = pathlib.Path(dataset_path) / DATASET2_FILE
    model_checkpoint = args["model_checkpoint"]  # Filepath to frozen model.
    batch_size = args["batch_size"]  # Batch size for scoring.
    out_dir = args["out_dir"]  # Destination path for results.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    # Log file path.
    log_path = os.path.join(
        out_dir,
        f"{project_name}.log")

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()],
        level=logging.INFO,
        force=True)

    # Load the frozen model checkpoint.
    classifier_status, classifier_dict = load_model(model_checkpoint)
    if not classifier_status:
        raise Exception(f"An error occured while loading model checkpoint!")

    # Same window the model was trained with (stored in the checkpoint). The
    # loader fixes the 0 to 1 range and the dropped columns; passing the
    # checkpoint's values makes an older checkpoint fail loudly if they differ.
    # The loader reads the labels from dataset2_eval.csv and rejects a file
    # without them.
    window = classifier_dict["window"]
    dataloaders, scaler, dataset_info = build_dataloaders(
        str(dataset_path),
        str(dataset2_path),
        window=window,
        batch_size=batch_size,
        feature_range=tuple(classifier_dict.get("feature_range", (0.0, 1.0))),
        constant_policy=classifier_dict.get("constant_policy", "drop"))

    feature_names = dataset_info["feature_names"]
    if classifier_dict["feature_names"] != feature_names:
        raise Exception("Checkpoint was trained on a different set of features.")
    if not scaler_matches(scaler, classifier_dict):
        raise Exception("Scaler differs from the checkpoint's: is this the same processed data?")

    # Recon Model.
    model = Transformer_Recon_Model(
        in_dim=classifier_dict["in_dim"],
        embedding_dim=classifier_dict["embedding_dim"],
        hidden_dim=classifier_dict["hidden_dim"],
        num_heads=classifier_dict["num_heads"],
        num_latent_tokens=classifier_dict["num_latent_tokens"],
        num_window=window,
        num_enc_blocks=classifier_dict["num_enc_blocks"],
        num_dec_blocks=classifier_dict["num_dec_blocks"],
        latent_dim=classifier_dict.get("latent_dim", 16),
        dropout=classifier_dict["dropout"],
        activation_type=classifier_dict["activation_type"])
    model.load_state_dict(classifier_dict["model"])
    model = model.to(device)

    logging.info(f"{project_name}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("#" * 100)
    logging.info(f"Model Checkpoint: {model_checkpoint}")
    logging.info(f"Checkpoint Global Steps: {classifier_dict['curr_global_steps']:,}")
    logging.info(f"Window (hours): {window:,}")
    logging.info(f"Features: {len(feature_names):,}")
    logging.info(f"Dataset 2: {dataset2_path} (labels: the 7 real attack windows)")
    logging.info(f"Grace hours after an attack: {args['grace_hours']:,}")

    # 1-2. Scoring and calibration on the clean blocks of Dataset 1, unless a
    # frozen calibration is given.
    if args["calibration"] is not None:
        calibration, saved = load_calibration(args["calibration"])
        if saved["feature_names"] != feature_names or saved["window"] != window:
            raise Exception("Calibration was made for a different model or feature set.")
        logging.info(f"Using frozen calibration: {args['calibration']}")
    else:
        calib_errors, _ = sensor_errors(
            model=model,
            dataloader=dataloaders["calib"],
            device=device)
        val_errors, _ = sensor_errors(
            model=model,
            dataloader=dataloaders["val"],
            device=device)
        calibration = calibrate(
            calib_errors=calib_errors,
            val_errors=val_errors,
            top_k=args["top_k"],
            max_alarm_fraction=args["max_alarm_fraction"],
            max_episodes_per_week=args["max_episodes_per_week"],
            merge_gap_hours=args["merge_gap_hours"])
        logging.info("Calibrated on the calibration block of Dataset 1.")
    logging.info(f"Budget: at most {calibration['max_alarm_fraction']:.2%} of clean hours under alarm")
    logging.info(f"Budget: at most {calibration['max_episodes_per_week']:g} alarm episodes per week")
    logging.info(f"Episode rule: merge runs separated by at most {calibration['merge_gap_hours']:,} quiet hours")

    # 3. Scoring Dataset 2.
    dev_errors, dev_rows = sensor_errors(
        model=model,
        dataloader=dataloaders["dev"],
        device=device)

    # Dataset 2's dates, labels and attack ids for exactly the scored hours.
    dev_when = dataset_info["dev"]["datetime"][dev_rows]
    dev_labels = dataset_info["dev"]["labels"][dev_rows]
    dev_attack_id = dataset_info["dev"]["attack_id"][dev_rows]

    # Per-sensor errors, saved so the scores can be fused with other detectors.
    np.savez_compressed(
        os.path.join(out_dir, "sensor_errors.npz"),
        sorted_calib_errors=calibration["reference"]["sorted_calib_errors"],
        calib_q99=calibration["reference"]["calib_q99"],
        dev_errors=dev_errors,
        dev_rows=dev_rows,
        feature_names=np.array(feature_names))

    # 4. Evaluation, plots and result files.
    run_evaluation(
        calibration=calibration,
        dev_errors=dev_errors,
        dev_when=dev_when,
        dev_labels=dev_labels,
        dev_attack_id=dev_attack_id,
        feature_names=feature_names,
        out_dir=out_dir,
        grace_hours=args["grace_hours"],
        top_sensors=args["top_sensors"],
        log=logging.info)

    logging.info(f"Saved results and plots to {out_dir}")

if __name__ == "__main__":
    main()
