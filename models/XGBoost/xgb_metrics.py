"""
Threshold-free and episode metrics, in plain NumPy.

These mirror the functions of the same names in
models/Transformer/evaluate_recon_model.py (same definitions, same tie
handling), copied so this folder does not need PyTorch. When the ensemble is
built they should move into one shared module.
"""
import numpy as np

HOURS_PER_WEEK = 24 * 7
# Fixed before any test-set scoring and used for every detector: alarm runs separated by at most
# this many quiet hours count as ONE episode (a flickering alarm is one event, not many).
MERGE_GAP_HOURS = 3


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
    """Average precision; one point per distinct score so ties are taken together."""
    n_pos = int((labels == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    last_of_value = np.flatnonzero(np.append(np.diff(sorted_scores) != 0, True))
    true_positives = np.cumsum(sorted_labels == 1)[last_of_value]
    precision = true_positives / (last_of_value + 1)
    recall = true_positives / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def episode_spans(alarms, merge_gap_hours=MERGE_GAP_HOURS):
    """Alarm episodes as (first, last) index pairs; runs <= merge_gap quiet hours apart merge."""
    alarm_hours = np.flatnonzero(alarms)
    if len(alarm_hours) == 0:
        return []
    breaks = np.flatnonzero(np.diff(alarm_hours) > merge_gap_hours + 1)
    starts = np.concatenate([[alarm_hours[0]], alarm_hours[breaks + 1]])
    ends = np.concatenate([alarm_hours[breaks], [alarm_hours[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def clean_block_stats(scores, threshold, merge_gap_hours=MERGE_GAP_HOURS):
    """False-alarm figures on a block of attack-free hours."""
    alarms = scores > threshold
    episodes = len(episode_spans(alarms, merge_gap_hours))
    return {
        "hours": int(len(scores)),
        "alarm_hours": int(alarms.sum()),
        "alarm_fraction": float(alarms.mean()),
        "episodes": episodes,
        "episodes_per_week": float(episodes / (len(scores) / HOURS_PER_WEEK))}


def find_threshold(scores, max_alarm_fraction=0.01, max_episodes_per_week=1.0, merge_gap_hours=MERGE_GAP_HOURS):
    """
    Lowest threshold whose alarms (score > threshold) stay inside BOTH budgets
    on clean calibration scores. Candidates run from the highest score down and
    the search stops at the first one that breaks a budget.
    """
    candidates = np.unique(scores)[::-1]
    threshold = candidates[0]
    for candidate in candidates:
        stats = clean_block_stats(scores, candidate, merge_gap_hours)
        if stats["alarm_fraction"] > max_alarm_fraction:
            break
        if stats["episodes_per_week"] > max_episodes_per_week:
            break
        threshold = candidate
    return float(threshold)
