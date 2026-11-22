"""
Writes the Transformer's per-hour scores in the form ensemble.py reads.

Run it once per trained model, from models/Transformer's environment:

    python export_transformer_scores.py \
        --dataset-path ../../data/processed \
        --model-checkpoint ../Transformer/runs/recon_24h/best_model.pt \
        --out transformer_hourly_scores.csv

For the validation, calibration and Dataset 2 hours it writes the three scores
of evaluate_recon_model.py (topk_percentile, mean, topk_ratio), each computed
against the per-sensor references of the CALIBRATION block, exactly as the
Transformer's own evaluation does. Rows are file rows of the processed CSVs
(not "hours since the first Dataset 1 hour"), so they match the other
detectors. Dataset 2 hours are written for every scored hour.

Note: the Transformer scores hour t from a window ending at t, so its first
window-1 hours of each block are missing; ensemble.py keeps only the hours all
detectors have.
"""
import sys
import pathlib
import argparse

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Transformer"))
from model import Transformer_Recon_Model  # noqa: E402
from batadal_dataset import build_dataloaders, DATASET2_FILE  # noqa: E402
from evaluate_recon_model import (  # noqa: E402
    SCORE_TYPES, load_model, scaler_matches, sensor_errors, calibrate, hourly_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=pathlib.Path, required=True)
    parser.add_argument("--model-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ok, ckpt = load_model(args.model_checkpoint)
    if not ok:
        raise SystemExit("could not load the checkpoint")
    window = ckpt["window"]
    loaders, scaler, info = build_dataloaders(
        str(args.dataset_path), str(args.dataset_path / DATASET2_FILE),
        window=window, batch_size=args.batch_size,
        feature_range=tuple(ckpt.get("feature_range", (0.0, 1.0))),
        constant_policy=ckpt.get("constant_policy", "drop"))
    if ckpt["feature_names"] != info["feature_names"] or not scaler_matches(scaler, ckpt):
        raise SystemExit("checkpoint and processed data do not match")

    model = Transformer_Recon_Model(
        in_dim=ckpt["in_dim"], embedding_dim=ckpt["embedding_dim"], hidden_dim=ckpt["hidden_dim"],
        num_heads=ckpt["num_heads"], num_latent_tokens=ckpt["num_latent_tokens"], num_window=window,
        num_enc_blocks=ckpt["num_enc_blocks"], num_dec_blocks=ckpt["num_dec_blocks"],
        latent_dim=ckpt.get("latent_dim", 16), dropout=ckpt["dropout"],
        activation_type=ckpt["activation_type"])
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device)

    errors, rows = {}, {}
    for name in ("cal", "val", "dev"):
        errors[name], rows[name] = sensor_errors(model, loaders[name], args.device)
    calibration = calibrate(errors["cal"], errors["val"], top_k=args.top_k)
    reference = calibration["reference"]

    frames = []
    for name, source in (("val", "val"), ("cal", "cal"), ("dev", "d2")):
        row = rows[name]
        if name in info["blocks"]:  # Dataset 1 blocks: undo the hours-since-first-hour offset
            row = row - info["blocks"][name][0]
        frame = pd.DataFrame({"source": source, "row": row})
        for score_type in SCORE_TYPES:
            frame[f"tr_{score_type}"] = hourly_scores(errors[name], reference, score_type, args.top_k)
        frames.append(frame)
    pd.concat(frames).to_csv(args.out, index=False)
    print(f"wrote {args.out} (window {window} h)")


if __name__ == "__main__":
    main()
