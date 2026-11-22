"""
Supervised XGBoost detector for BATADAL, after Nemade et al. (2026).

    python train_xgboost.py \
        --dataset-path ../../data/processed \
        --out-dir runs/xgb \
        --seeds 0 1 2

Labels (project decision): Dataset 1 is all normal; Dataset 2 contributes only
its confirmed attack hours (ATT_FLAG == 1, 219 hours in 5 of the 7 attacks).
ATT_FLAG == -999 rows are dropped: they are never a training label and never an
hour-level evaluation label.

Protocol (this is where it differs from the paper, which scored the same series
it trained on):

  negatives for training   Dataset 1 TRAIN block
  threshold                CALIBRATION block, lowest threshold inside the
                           false-alarm budget (<= 1 episode/week, <= 1% hours)
  false-alarm evaluation   VALIDATION block (never trained on, never used to
                           set the threshold)
  positives                leave-one-attack-out over the 5 attacks that have
                           labeled hours (1, 3, 4, 5, 6): the model that scores
                           attack k never saw attack k.
  attacks 2 and 7          have no labeled hour at all, so no model ever
                           trains on them. They are scored by the model trained
                           on all 219 positives and are a true hold-out for
                           detection and delay.

Detection delay is measured from the real start of each attack, which comes
from attack_id in dataset2_eval.csv. The -999 hours inside a real window are
scored only for this (did an alarm appear, and when); they are not counted in
precision, recall or any hour-level figure.

Outputs per seed (<out-dir>/seed<k>/): metrics.json, per_attack.csv,
folds.csv, xgb_hourly_scores.csv (for the ensemble), feature_importance.csv,
final_model.json, plots. <out-dir>/summary.json averages over seeds.
"""
import os
import json
import pathlib
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgb_data import load_blocks
from xgb_metrics import (
    HOURS_PER_WEEK, roc_auc, pr_auc, episode_spans, clean_block_stats, find_threshold)

# Nemade et al. (2026), section 4.2.
PAPER_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8)


def fit_xgb(neg_X, pos_X, params, seed, n_jobs):
    """One classifier. scale_pos_weight = negatives / positives of THIS training set."""
    X = np.vstack([neg_X, pos_X])
    y = np.concatenate([np.zeros(len(neg_X)), np.ones(len(pos_X))])
    model = xgb.XGBClassifier(
        scale_pos_weight=len(neg_X) / len(pos_X),
        eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=n_jobs, **params)
    model.fit(X, y)
    return model


def proba(model, X):
    return model.predict_proba(X)[:, 1].astype(np.float64)


def ratio(a, b):
    return float(a / b) if b > 0 else None


def run_seed(blocks, args, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    params = {**PAPER_PARAMS, "n_estimators": args.n_estimators, "max_depth": args.max_depth,
              "learning_rate": args.learning_rate}

    train_F = blocks["train"][0]
    val_F, cal_F = blocks["val"][0], blocks["cal"][0]
    d2_F, d2_meta = blocks["d2"]
    columns = list(train_F.columns)

    # ---- Label policy: positives are ATT_FLAG == 1 only; -999 is dropped.
    is_pos = (d2_meta["att_flag"] == 1).to_numpy()
    attack_id = d2_meta["attack_id"].to_numpy()
    pos_X, pos_attack = d2_F.to_numpy()[is_pos], attack_id[is_pos]
    labeled_attacks = sorted(int(a) for a in np.unique(pos_attack))
    all_attacks = sorted(int(a) for a in np.unique(attack_id) if a > 0)
    unseen_attacks = [a for a in all_attacks if a not in labeled_attacks]

    # Hours scored on Dataset 2: every hour inside a real attack window. Only
    # the ATT_FLAG == 1 ones count as labeled; the rest feed detection delay.
    in_window = attack_id > 0
    win_X = d2_F.to_numpy()[in_window]
    win_attack = attack_id[in_window]
    win_flag = d2_meta["att_flag"].to_numpy()[in_window]
    win_when = d2_meta["datetime"].to_numpy()[in_window]
    win_row = d2_meta["row"].to_numpy()[in_window]

    neg_X = train_F.to_numpy()

    # ---- One model per held-out attack, plus the all-positives model.
    setups = {f"fold{a}": pos_attack != a for a in labeled_attacks}
    setups["final"] = np.ones(len(pos_X), dtype=bool)
    models = {}
    for name, use in setups.items():
        model = fit_xgb(neg_X, pos_X[use], params, seed, args.n_jobs)
        cal_scores = proba(model, cal_F.to_numpy())
        val_scores = proba(model, val_F.to_numpy())
        threshold = find_threshold(
            cal_scores, args.max_alarm_fraction, args.max_episodes_per_week)
        models[name] = {
            "model": model, "threshold": threshold,
            "n_pos_train": int(use.sum()),
            "cal_scores": cal_scores, "val_scores": val_scores,
            "cal_stats": clean_block_stats(cal_scores, threshold),
            "val_stats": clean_block_stats(val_scores, threshold),
            "win_scores": proba(model, win_X)}

    # The model that scores each attack never trained on it.
    scorer = {a: (f"fold{a}" if a in labeled_attacks else "final") for a in all_attacks}

    # ---- Per attack: detection, delay, labeled-hour recall.
    per_attack, ttd_ratios, oof_scores, tp_total, pos_total = [], [], np.zeros(len(win_X)), 0, 0
    for a in all_attacks:
        info = models[scorer[a]]
        sel = win_attack == a
        scores = info["win_scores"][sel]
        oof_scores[sel] = scores
        alarms = scores > info["threshold"]
        hours = int(sel.sum())
        first = np.flatnonzero(alarms)
        delay = int(first[0]) if len(first) > 0 else None  # window rows are hourly from the real start
        labeled = win_flag[sel] == 1
        tp = int((alarms & labeled).sum())
        tp_total += tp
        pos_total += int(labeled.sum())
        ttd_ratios.append((delay if delay is not None else hours) / hours)
        per_attack.append({
            "attack": a, "real_hours": hours, "labeled_hours": int(labeled.sum()),
            "scored_by": scorer[a], "threshold": info["threshold"],
            "detected": delay is not None, "delay_hours": delay,
            "labeled_hours_alarmed": tp,
            "recall_on_labeled": ratio(tp, int(labeled.sum())),
            "window_hours_alarmed": int(alarms.sum())})

    # ---- Clean-hour figures on the validation block, averaged over the models used.
    used = sorted(set(scorer.values()))
    val_alarm_fraction = float(np.mean([models[n]["val_stats"]["alarm_fraction"] for n in used]))
    val_episodes_week = float(np.mean([models[n]["val_stats"]["episodes_per_week"] for n in used]))
    tnr = 1.0 - val_alarm_fraction
    recall = ratio(tp_total, pos_total)
    fp_mean = float(np.mean([models[n]["val_stats"]["alarm_hours"] for n in used]))
    precision = ratio(tp_total, tp_total + fp_mean)
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and precision + recall > 0 else None)

    # Pooled ranking figures: labeled attack hours (out-of-fold) against the
    # validation block, where each negative gets the mean score of the models used.
    val_mean = np.mean([models[n]["val_scores"] for n in used], axis=0)
    pos_scores = oof_scores[win_flag == 1]
    rank_scores = np.concatenate([pos_scores, val_mean])
    rank_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(val_mean))])

    s_ttd = float(1.0 - np.mean(ttd_ratios))
    s_clf = (recall + tnr) / 2.0
    metrics = {
        "seed": seed,
        "attacks_detected": int(sum(r["detected"] for r in per_attack)),
        "attacks_total": len(per_attack),
        "missed_attacks": ",".join(str(r["attack"]) for r in per_attack if not r["detected"]),
        "unseen_attacks": ",".join(map(str, unseen_attacks)),
        "unseen_detected": int(sum(r["detected"] for r in per_attack if r["attack"] in unseen_attacks)),
        "mean_delay_hours": ratio(sum(r["delay_hours"] for r in per_attack if r["detected"]),
                                  sum(r["detected"] for r in per_attack)),
        "labeled_positive_hours": pos_total,
        "val_negative_hours": int(len(val_mean)),
        "recall": recall, "precision": precision, "f1": f1,
        "true_negative_rate": tnr,
        "val_alarm_fraction": val_alarm_fraction,
        "val_false_alarm_episodes_per_week": val_episodes_week,
        "roc_auc": roc_auc(rank_scores, rank_labels),
        "pr_auc": pr_auc(rank_scores, rank_labels),
        "s_ttd": s_ttd, "s_clf": s_clf, "s_score": 0.5 * s_ttd + 0.5 * s_clf,
        "n_features": len(columns),
        "positive_to_negative_in_precision": f"{pos_total}:{len(val_mean)}"}

    # ---- Files.
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(per_attack).to_csv(os.path.join(out_dir, "per_attack.csv"), index=False)
    pd.DataFrame([{
        "model": n, "n_pos_train": m["n_pos_train"], "threshold": m["threshold"],
        "cal_alarm_fraction": m["cal_stats"]["alarm_fraction"],
        "cal_episodes_per_week": m["cal_stats"]["episodes_per_week"],
        "val_alarm_fraction": m["val_stats"]["alarm_fraction"],
        "val_episodes_per_week": m["val_stats"]["episodes_per_week"]}
        for n, m in models.items()]).to_csv(os.path.join(out_dir, "folds.csv"), index=False)

    # Hourly scores for the ensemble: validation and calibration hours carry one
    # held-out-attack model's score each (none of them trained on these hours);
    # Dataset 2 attack-window hours carry the out-of-fold score.
    fold_names = [n for n in models if n != "final"]
    rows = []
    for source, meta, key in (("val", blocks["val"][1], "val_scores"), ("cal", blocks["cal"][1], "cal_scores")):
        # Each hour gets the score of ONE held-out-attack model (round-robin by row), not
        # the mean of all of them: Dataset 2 hours carry a single model's score, and a
        # mean of models has lighter tails, which would make clean hours look calmer.
        stacked = np.stack([models[n][key] for n in fold_names])
        single_scores = stacked[np.arange(stacked.shape[1]) % len(fold_names), np.arange(stacked.shape[1])]
        rows.append(pd.DataFrame({
            "source": source, "datetime": meta["datetime"], "row": meta["row"],
            "att_flag": meta["att_flag"], "label": 0, "attack_id": 0, "xgb_score": single_scores}))
    rows.append(pd.DataFrame({
        "source": "d2", "datetime": win_when, "row": win_row, "att_flag": win_flag,
        "label": 1, "attack_id": win_attack, "xgb_score": oof_scores}))
    pd.concat(rows).to_csv(os.path.join(out_dir, "xgb_hourly_scores.csv"), index=False)

    importance = pd.Series(models["final"]["model"].get_booster().get_score(importance_type="gain"))
    importance = importance.reindex([f"f{i}" for i in range(len(columns))]).fillna(0.0)
    importance.index = columns
    importance = importance.sort_values(ascending=False)
    importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["gain"])
    models["final"]["model"].save_model(os.path.join(out_dir, "final_model.json"))

    plot_importance(importance, out_dir)
    plot_attacks(per_attack, win_attack, win_flag, oof_scores, models, scorer, out_dir)
    return metrics, per_attack


def plot_importance(importance, out_dir, top=30):
    top_features = importance.head(top)[::-1]
    fig, axis = plt.subplots(figsize=(7, 8))
    axis.barh(top_features.index, top_features.values, color="#2a78d6")
    axis.set_xlabel("XGBoost gain (model trained on all 219 positives)")
    axis.set_title(f"Top {top} features")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close(fig)


def plot_attacks(per_attack, win_attack, win_flag, oof_scores, models, scorer, out_dir):
    fig, axes = plt.subplots(len(per_attack), 1, figsize=(10, 2.0 * len(per_attack)))
    for axis, row in zip(np.atleast_1d(axes), per_attack):
        sel = win_attack == row["attack"]
        hours = np.arange(sel.sum())
        axis.plot(hours, oof_scores[sel], color="#2a78d6", linewidth=1)
        axis.axhline(row["threshold"], color="black", linestyle="--", linewidth=0.8)
        labeled = win_flag[sel] == 1
        axis.fill_between(hours, 0, 1, where=labeled, color="#eb6834", alpha=0.18,
                          transform=axis.get_xaxis_transform(), label="ATT_FLAG = 1")
        status = f"detected after {row['delay_hours']} h" if row["detected"] else "MISSED"
        axis.set_title(f"attack {row['attack']} ({row['real_hours']} h, scored by {row['scored_by']}): {status}",
                       loc="left", fontsize=9)
        axis.set_ylabel("P(attack)", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("hours from the real start of the attack")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "attack_scores.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="XGBoost BATADAL detector")
    parser.add_argument("--dataset-path", type=pathlib.Path, required=True,
                        help="Folder of processed files (data/processed).")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-estimators", type=int, default=PAPER_PARAMS["n_estimators"])
    parser.add_argument("--max-depth", type=int, default=PAPER_PARAMS["max_depth"])
    parser.add_argument("--learning-rate", type=float, default=PAPER_PARAMS["learning_rate"])
    parser.add_argument("--no-time-features", action="store_true",
                        help="Leave out the hour-of-day / day-of-week encodings.")
    parser.add_argument("--max-alarm-fraction", type=float, default=0.01)
    parser.add_argument("--max-episodes-per-week", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    blocks = load_blocks(str(args.dataset_path), time_features=not args.no_time_features)
    n_features = blocks["train"][0].shape[1]
    print(f"features: {n_features} | train {len(blocks['train'][0])} val {len(blocks['val'][0])} "
          f"cal {len(blocks['cal'][0])} | Dataset 2 rows {len(blocks['d2'][0])}")

    all_metrics = []
    for seed in args.seeds:
        metrics, per_attack = run_seed(blocks, args, os.path.join(args.out_dir, f"seed{seed}"), seed)
        all_metrics.append(metrics)
        print(f"seed {seed}: detected {metrics['attacks_detected']}/{metrics['attacks_total']} "
              f"(unseen {metrics['unseen_detected']}/{len(metrics['unseen_attacks'].split(','))}) "
              f"delay {metrics['mean_delay_hours']} recall {metrics['recall']:.3f} "
              f"TNR {metrics['true_negative_rate']:.4f} ROC {metrics['roc_auc']:.3f} "
              f"PR {metrics['pr_auc']:.3f} S {metrics['s_score']:.3f}")

    numeric = [k for k, v in all_metrics[0].items()
               if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "seed"]
    summary = {k: {"mean": float(np.mean([m[k] for m in all_metrics if m[k] is not None])),
                   "std": float(np.std([m[k] for m in all_metrics if m[k] is not None]))}
               for k in numeric}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "args": {k: str(v) for k, v in vars(args).items()},
                   "summary": summary}, f, indent=2)


if __name__ == "__main__":
    main()
