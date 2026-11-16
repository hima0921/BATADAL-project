import os
import json
import shutil
import pathlib
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # Write image files without needing a display.
import matplotlib.pyplot as plt

from model import Transformer_Recon_Model

from batadal_dataset import build_dataloaders

"""
Trains the Transformer reconstruction model on the TRAIN block of Dataset 1 and
validates it on the VALIDATION block (60/20/20 split, done in batadal_dataset.py).

The calibration block, Dataset 2 and the Test set are never read here. They are
for thresholds and evaluation, after the model is frozen.
"""

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.:
        raise argparse.ArgumentTypeError("%r not in range > 0."%(x,))
    return x


# Same call signatures as utils.model_utils in train_recon_model.py, kept in
# this file so the script runs on its own. Swap for the utils versions if the
# project already has them.
def checkpoint_model(data_dict, out_dir, model, global_steps, model_optim, logging):
    checkpoint_dict = dict(data_dict)
    checkpoint_dict["model"] = model.state_dict()
    checkpoint_dict["optimizer"] = model_optim.state_dict()

    checkpoint_path = os.path.join(out_dir, f"{global_steps}_model.pt")
    torch.save(checkpoint_dict, checkpoint_path)
    logging(f"Saved model checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_model(checkpoint_path):
    try:
        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        logging.error(f"Could not load checkpoint {checkpoint_path}: {e}")
        return False, None
    return True, checkpoint_dict


def train_step(model, model_optim, tr_windows, tr_targets):
    """One optimizer update on one training batch. Returns the batch loss."""
    model.train()

    model_optim.zero_grad()

    tr_recon_windows = model(tr_windows)

    recon_loss = F.mse_loss(
        tr_recon_windows,
        tr_targets)
    if torch.isnan(recon_loss):
        raise Exception("NaN encountered during training.")

    recon_loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    model_optim.step()
    return recon_loss.item()


@torch.no_grad()
def validate_model(model, dataloader, device):
    """Mean reconstruction MSE over a whole dataloader, in eval mode."""
    model.eval()

    total_loss, total_count = 0.0, 0
    for windows, targets, _, _ in dataloader:
        windows = windows.to(device)  # (N,Seq_window,Dim_in)
        targets = targets.to(device)  # (N,Seq_window,Dim_in)

        recon_windows = model(windows)

        loss = F.mse_loss(
            recon_windows,
            targets,
            reduction="sum")
        total_loss += loss.item()
        total_count += targets.numel()
    return total_loss / total_count


def plot_progress(history, out_dir):
    """Save the train and validation loss curves as an image and as JSON."""
    plt.figure(figsize=(7, 4))
    plt.plot(history["steps"], history["train"], marker="o", markersize=3, label="train")
    plt.plot(history["steps"], history["val"], marker="o", markersize=3, label="validation")
    if history["best_step"] is not None:
        plt.axvline(history["best_step"], color="green", linestyle=":", label="best validation")
    plt.xlabel("global steps")
    plt.ylabel("reconstruction MSE")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "recon_loss.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "recon_loss.json"), "w") as history_f:
        json.dump(history, history_f, indent=2)


def main():
    project_name = "BATADAL Recon Model"

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
        help="File path for the normal BATADAL training CSV (Training dataset 1).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--batch-size",
        help="Batch size of training dataset.",
        type=int,
        default=128)
    parser.add_argument(
        "--checkpoint-steps",
        help="Steps for checkpointing and validating model.",
        type=int,
        default=1_000)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint to load from (if any).",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "--lr-steps",
        help="Global steps in between halving learning rate.",
        default=3_500,
        type=int)
    parser.add_argument(
        "--lr-gamma",
        help="Learning Rate gamma value.",
        default=0.5,
        type=restricted_float)
    parser.add_argument(
        "--load-optim",
        action='store_true',
        help="Load model's optimizer's weights and parameters, if loading model.")
    parser.add_argument(
        "--constant-policy",
        help="Drop or keep the features that never change in training.",
        choices=['drop', 'keep'],
        type=str,
        default="drop")
    parser.add_argument(
        "--seed",
        help="Random seed for model weights and batch order.",
        type=int,
        default=0)
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to JSON config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    dataset_path = args["dataset_path"]  # Dataset path for dataset.
    batch_size = args["batch_size"]  # Batch size of training dataset.
    checkpoint_steps = args["checkpoint_steps"]  # Steps to checkpoint model.
    model_checkpoint = args["model_checkpoint"]  # Filepath to models saved.
    load_optim = args["load_optim"]  # Reload saved optimizer weights.
    lr_steps = args["lr_steps"]  # Global steps in between halving learning rate.
    lr_gamma = args["lr_gamma"]  # Learning Rate gamma value.
    constant_policy = args["constant_policy"]  # Drop / keep constant features.
    seed = args["seed"]  # Random seed.
    out_dir = args["out_dir"]  # Destination path for model.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    torch.manual_seed(seed)

    # Config Dict.
    config_json = args["config_path"]  # Load and Parse config JSON.
    with open(config_json, 'r') as config_f:
        data_json = config_f.read()
    config_dict = json.loads(data_json)

    # Training params.
    curr_global_steps = 0
    max_global_steps = config_dict["max_global_steps"]

    # Model Params (From config file).
    model_lr = config_dict["model_lr"]
    window = config_dict["window"]
    dropout = config_dict["dropout"]
    num_heads = config_dict["num_heads"]
    num_enc_blocks = config_dict["num_enc_blocks"]
    num_dec_blocks = config_dict["num_dec_blocks"]
    hidden_dim = config_dict["hidden_dim"]
    embedding_dim = config_dict["embedding_dim"]
    activation_type = config_dict["activation_type"]

    # Bottleneck parameters.
    num_encodings = config_dict["num_encodings"]
    bottleneck_dim = config_dict["bottleneck_dim"]

    # BATADAL Datasets: Dataset 1 split 60/20/20 by time, with 24-hour gaps.
    # feature_range uses (0, 1) as per the proposal.
    feature_range = (0.0, 1.0)
    dataloaders, scaler, dataset_info = build_dataloaders(
        str(dataset_path),
        window=window,
        batch_size=batch_size,
        feature_range=feature_range,
        constant_policy=constant_policy,
        seed=seed)

    # Train Dataloader (first 60%, shuffled) and Validation Dataloader
    # (next 20%, in time order). dataloaders["calib"] is left untouched.
    tr_dataloader = dataloaders["train"]
    val_dataloader = dataloaders["val"]

    in_dim = dataset_info["in_dim"]
    feature_names = dataset_info["feature_names"]

    # Train / Validation loss at every checkpoint step, for the progress plot.
    history = {
        "steps": [],
        "train": [],
        "val": [],
        "best_step": None,
        "best_val": None}

    # Recon Model.
    model = Transformer_Recon_Model(
        in_dim=in_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_encodings=num_encodings,
        num_window=window,
        num_enc_blocks=num_enc_blocks,
        num_dec_blocks=num_dec_blocks,
        bottleneck_dim=bottleneck_dim,
        dropout=dropout,
        activation_type=activation_type)

    # Load model checkpoints if any.
    if model_checkpoint is not None:
        logging.info(f"Loading pretrained model checkpoint!")
        classifier_status, classifier_dict = load_model(model_checkpoint)
        if not classifier_status:
            raise Exception(f"An error occured while loading pretrained model checkpoint!")

        if classifier_dict["feature_names"] != feature_names:
            raise Exception("Checkpoint was trained on a different set of features.")

        model.load_state_dict(classifier_dict["model"])
        model = model.to(device)

        model_optim = torch.optim.Adam(
            model.parameters(),
            lr=model_lr)

        # Load Optimizer params and global steps params.
        if load_optim:
            curr_global_steps = classifier_dict["curr_global_steps"]

            # Continue the loss curves from the checkpoint.
            history = classifier_dict.get("history", history)

            logging.info("Resuming training using saved optimizer weights and curr_global_steps.")
            model_optim.load_state_dict(classifier_dict["optimizer"])
    else:
        model = model.to(device)

        model_optim = torch.optim.Adam(
            model.parameters(),
            lr=model_lr)

    # Model parameter size.
    model_params_size = sum(param.numel() for param in model.parameters())

    # Learning Rate Scheduler.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=model_optim,
        step_size=lr_steps,
        gamma=lr_gamma)

    # Log file path.
    log_path = os.path.join(
        out_dir,
        f"{project_name}.log")

    # Logs Info to parent directory.
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()],
        level=logging.DEBUG,
        force=True)

    # Keep matplotlib's and PIL's own debug messages out of the training log.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    tr_start, tr_end = dataset_info["blocks"]["train"]
    val_start, val_end = dataset_info["blocks"]["val"]

    logging.info(f"{project_name}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("#" * 100)
    logging.info("Dataset Parameters.")
    logging.info(f"Window (hours): {window:,}")
    logging.info(f"Features: {in_dim:,}")
    logging.info(f"Constant Features ({constant_policy}): {', '.join(dataset_info['constant_in_train']) or 'none'}")
    logging.info(f"Training Batch Size: {batch_size:,}")
    logging.info(f"Train Rows: {tr_start:,} to {tr_end - 1:,} | Windows: {len(tr_dataloader.dataset):,}")
    logging.info(f"Validation Rows: {val_start:,} to {val_end - 1:,} | Windows: {len(val_dataloader.dataset):,}")
    logging.info(f"Seed: {seed:,}")
    logging.info("#" * 100)
    logging.info("Model Parameters.")
    logging.info(f"Total Model Param size: {model_params_size:,}")
    logging.info("-" * 100)
    logging.info(f"Num Encoder Blocks: {num_enc_blocks:,}")
    logging.info(f"Num Decoder Blocks: {num_dec_blocks:,}")
    logging.info(f"Num Encodings: {num_encodings:,}")
    logging.info(f"Bottleneck dim: {bottleneck_dim:,}")
    logging.info(f"Activation Type: {activation_type}")
    logging.info(f"Hidden Dimension: {hidden_dim:,}")
    logging.info(f"Embedding Dimension: {embedding_dim:,}")
    logging.info(f"Dropout: {dropout:,}")
    logging.info(f"Model Learning-Rate Gamma: {lr_gamma:,}")
    logging.info(f"Num Heads: {num_heads:,}")
    logging.info(f"Model Learning Rate: {model_optim.param_groups[0]['lr']:,}")
    logging.info("#" * 100)
    logging.info("Training Parameters.")
    logging.info(f"Current Global Steps: {curr_global_steps:,}")
    logging.info(f"Max Global step: {max_global_steps:,}")
    logging.info(f"Model Learning-Rate half-steps: {lr_steps:,}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps:,}")
    logging.info("#" * 100)

    # Model's parameters to be saved. The scaler values are saved with the
    # model, because scoring new data needs the exact same scaling.
    model_data_dict = {
        "in_dim": in_dim,
        "window": window,
        "dropout": dropout,
        "lr_gamma": lr_gamma,
        "lr_steps": lr_steps,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "num_encodings": num_encodings,
        "bottleneck_dim": bottleneck_dim,
        "num_enc_blocks": num_enc_blocks,
        "num_dec_blocks": num_dec_blocks,
        "activation_type": activation_type,
        "seed": seed,
        "feature_names": feature_names,
        "feature_range": list(feature_range),
        "constant_policy": constant_policy,
        "scaler_all_names": list(scaler.names),
        "scaler_min": scaler.min_.tolist(),
        "scaler_span": scaler.span_.tolist(),
        "scaler_keep": scaler.keep_.tolist(),
        "val_loss": None,
        "history": history,
        "curr_global_steps": curr_global_steps}

    def validate_and_checkpoint(global_steps):
        """Validation pass, progress plot, checkpoint, best-model copy."""
        # Both losses use the whole block in eval mode, so they are comparable.
        tr_loss = validate_model(
            model=model,
            dataloader=tr_dataloader,
            device=device)
        val_loss = validate_model(
            model=model,
            dataloader=val_dataloader,
            device=device)

        logging.info("Global Steps: {:,} | Train Recon Loss: {:,.5f} | Validation Recon Loss: {:,.5f}".format(
            global_steps,
            tr_loss,
            val_loss))

        # A resumed run re-validates the step it was saved at: do not log it twice.
        if len(history["steps"]) == 0 or history["steps"][-1] != global_steps:
            history["steps"].append(global_steps)
            history["train"].append(tr_loss)
            history["val"].append(val_loss)

        # Model selection uses the validation block only.
        is_best = history["best_val"] is None or val_loss < history["best_val"]
        if is_best:
            history["best_val"] = val_loss
            history["best_step"] = global_steps

        plot_progress(
            history=history,
            out_dir=out_dir)

        model_data_dict["val_loss"] = val_loss
        model_data_dict["history"] = history
        model_data_dict["curr_global_steps"] = global_steps
        checkpoint_path = checkpoint_model(
            data_dict=model_data_dict,
            out_dir=out_dir,
            model=model,
            global_steps=global_steps,
            model_optim=model_optim,
            logging=logging.info)

        if is_best:
            shutil.copyfile(
                checkpoint_path,
                os.path.join(out_dir, "best_model.pt"))
            logging.info("Best validation loss so far: saved best_model.pt (step {:,}).".format(
                global_steps))

    # Training starts here.
    stop_training = False
    if curr_global_steps >= max_global_steps:
        stop_training = True

    while not stop_training:
        for index, (tr_windows, tr_targets, _, _) in enumerate(tr_dataloader):
            tr_windows = tr_windows.to(device)  # (N,Seq_window,Dim_in)
            tr_targets = tr_targets.to(device)  # (N,Seq_window,Dim_in)

            # Validate and checkpoint model.
            if curr_global_steps % checkpoint_steps == 0:
                validate_and_checkpoint(curr_global_steps)

            # Train Operation.
            recon_loss = train_step(
                model=model,
                model_optim=model_optim,
                tr_windows=tr_windows,
                tr_targets=tr_targets)

            message = "Global Steps: {:,} | Steps: {:,} / {:,} | Recon Loss: {:,.5f} | LR: {:.3E}".format(
                curr_global_steps + 1,
                index + 1,
                len(tr_dataloader),
                recon_loss,
                model_optim.param_groups[0]['lr'])
            logging.info(message)

            curr_global_steps = curr_global_steps + 1

            # Stop training when stopping criteria is met.
            if curr_global_steps >= max_global_steps:
                stop_training = True
                break

            lr_scheduler.step()

    # Last validation and model checkpoint.
    validate_and_checkpoint(curr_global_steps)

    logging.info("Best Validation Recon Loss: {:,.5f} at step {:,}.".format(
        history["best_val"],
        history["best_step"]))

if __name__ == "__main__":
    main()