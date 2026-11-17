import os
import pathlib
import logging
import argparse

import numpy as np

from model import Transformer_Recon_Model

from batadal_dataset import build_dataloaders

# The scoring and threshold code is shared with the evaluation script, so the
# two can never disagree about how a score or a threshold is computed.
from evaluate_recon_model import (
    SCORE_TYPES,
    restricted_float,
    load_model,
    scaler_matches,
    sensor_errors,
    calibrate,
    save_calibration,
    plot_calibration)

"""
Calibrates a FROZEN Transformer reconstruction model and saves the result.

    python calibrate_recon_model.py \
        --dataset-path data/processed \
        --model-checkpoint runs/recon_24h/best_model.pt \
        --out-dir runs/recon_24h/calibration

Uses clean data from Dataset 1 only:
  calibration block (last 20%)    per-sensor references and the alarm threshold
  validation block (middle 20%)   check that the threshold transfers

Three hourly scores are calibrated (see SCORE_TYPES in evaluate_recon_model.py).
For each, the threshold is the lowest one that keeps false alarms inside the
budget: at most 1 alarm episode per week and at most 1% of hours under alarm.

The saved calibration.pt freezes the thresholds. Pass it to
evaluate_recon_model.py with --calibration to evaluate without recalibrating.
Dataset 2 and the Test set are never read here.
"""


def main():
    project_name = "BATADAL Recon Model Calibration"

    parser = argparse.ArgumentParser(
        description=f"{project_name}")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--dataset-path",
        help="Folder of processed files from Pipeline.ipynb (data/processed).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to the frozen model checkpoint (e.g. best_model.pt).",
        required=True,
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
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    dataset_path = args["dataset_path"]  # Dataset path for dataset.
    model_checkpoint = args["model_checkpoint"]  # Filepath to frozen model.
    batch_size = args["batch_size"]  # Batch size for scoring.
    top_k = args["top_k"]  # Top-k sensor scores.
    max_alarm_fraction = args["max_alarm_fraction"]  # Budget: alarm time.
    max_episodes_per_week = args["max_episodes_per_week"]  # Budget: episodes.
    merge_gap_hours = args["merge_gap_hours"]  # Episode grouping rule.
    out_dir = args["out_dir"]  # Destination path for calibration.
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

    # Same data settings the model was trained with (stored in the checkpoint).
    window = classifier_dict["window"]
    dataloaders, scaler, dataset_info = build_dataloaders(
        str(dataset_path),
        window=window,
        batch_size=batch_size,
        feature_range=tuple(classifier_dict["feature_range"]),
        constant_policy=classifier_dict["constant_policy"])

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
        num_encodings=classifier_dict["num_encodings"],
        num_window=window,
        num_enc_blocks=classifier_dict["num_enc_blocks"],
        num_dec_blocks=classifier_dict["num_dec_blocks"],
        bottleneck_dim=classifier_dict.get("bottleneck_dim", 16),
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
    logging.info(f"Top-k sensors: {top_k:,}")
    logging.info(f"Budget: at most {max_alarm_fraction:.2%} of hours under alarm")
    logging.info(f"Budget: at most {max_episodes_per_week:g} alarm episodes per week")
    logging.info(f"Episode rule: merge runs separated by at most {merge_gap_hours:,} quiet hours")
    logging.info("#" * 100)

    # 1. Per-sensor errors on the two clean blocks.
    calib_errors, calib_rows = sensor_errors(
        model=model,
        dataloader=dataloaders["calib"],
        device=device)
    val_errors, _ = sensor_errors(
        model=model,
        dataloader=dataloaders["val"],
        device=device)

    # 2. References, hourly scores and thresholds for every score type.
    calibration = calibrate(
        calib_errors=calib_errors,
        val_errors=val_errors,
        top_k=top_k,
        max_alarm_fraction=max_alarm_fraction,
        max_episodes_per_week=max_episodes_per_week,
        merge_gap_hours=merge_gap_hours)

    # 3. Report.
    logging.info(f"{'':<34}" + "".join(f"{score_type:<20}" for score_type in SCORE_TYPES))
    rows = [
        ("Threshold", lambda s: f"{calibration['thresholds'][s]:.5g}"),
        ("Calibration hours under alarm", lambda s: f"{calibration['calib_stats'][s]['alarm_fraction']:.2%}"),
        ("Calibration episodes per week", lambda s: f"{calibration['calib_stats'][s]['episodes_per_week']:.2f}"),
        ("Validation hours under alarm", lambda s: f"{calibration['val_stats'][s]['alarm_fraction']:.2%}"),
        ("Validation episodes per week", lambda s: f"{calibration['val_stats'][s]['episodes_per_week']:.2f}"),
        ("Validation inside budget", lambda s: str(calibration["val_in_budget"][s]))]
    for name, cell in rows:
        logging.info(f"{name:<34}" + "".join(f"{cell(s):<20}" for s in SCORE_TYPES))
    logging.info("#" * 100)

    for score_type in SCORE_TYPES:
        if not calibration["val_in_budget"][score_type]:
            logging.info(f"NOTE: {score_type} is OUTSIDE the budget on the validation block.")
        # A threshold at the percentile score's ceiling can never be crossed.
        if score_type == "topk_percentile" and calibration["thresholds"][score_type] >= 1.0:
            logging.info("NOTE: the topk_percentile threshold sits at the score's ceiling "
                         "of 1, so that score cannot raise an alarm.")

    # Sensors that the model reconstructs worst on normal data.
    mean_errors = calib_errors.mean(axis=0)
    worst = np.argsort(mean_errors)[::-1][:5]
    logging.info("Highest mean calibration error: " + ", ".join(
        f"{feature_names[j]} ({mean_errors[j]:.5f})" for j in worst))

    # 4. Plots and files.
    for score_type in SCORE_TYPES:
        plot_calibration(
            calibration["calib_scores"][score_type],
            calibration["val_scores"][score_type],
            calibration["thresholds"][score_type],
            score_type,
            out_dir)

    save_calibration(
        calibration=calibration,
        out_dir=out_dir,
        extra={
            "model_checkpoint": str(model_checkpoint),
            "curr_global_steps": classifier_dict["curr_global_steps"],
            "window": window,
            "feature_names": feature_names,
            "calib_rows": [int(calib_rows[0]), int(calib_rows[-1])]})

    logging.info(f"Saved calibration: {os.path.join(out_dir, 'calibration.pt')}")
    logging.info(f"Saved summary: {os.path.join(out_dir, 'calibration.json')}")

if __name__ == "__main__":
    main()
