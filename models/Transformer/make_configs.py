import os
import json
import pathlib
import argparse

"""
Writes the config files for an EVEN ablation study: the same few variations
for the 12-hour and the 24-hour window, each changing ONE setting from that
window's baseline.

    python make_configs.py --out-dir configs/ablation
    python make_configs.py --out-dir configs/ablation --max-global-steps 10000

One folder per window, <out-dir>/w12/ and <out-dir>/w24/, each holding
baseline.json and one <setting>-<value>.json per variation. Run each folder on
its own, with its own runs folder:

    python run_ablations.py --configs-dir configs/ablation/w12 \
        --dataset-path data/processed --runs-dir runs/ablation/w12
    python run_ablations.py --configs-dir configs/ablation/w24 \
        --dataset-path data/processed --runs-dir runs/ablation/w24

Edit WINDOWS, BASELINE, ABLATIONS and EXTRAS below to change what is
generated. Fix this list before running anything, so settings are not chosen
after seeing results.
"""

# Same list as ALLOWED_WINDOWS in batadal_dataset.py: the 24-hour gaps between
# the blocks of the processed data were sized for these two.
ALLOWED_WINDOWS = (12, 24)

# Every window gets the same set of configurations.
WINDOWS = [12, 24]

# The reference model for each window. Same keys as
# 24_batadal_recon_config.json; "window" is filled in per window.
BASELINE = {
    "max_global_steps": 15_000,
    "model_lr": 3e-3,
    "dropout": 0.1,
    "num_heads": 8,
    "num_enc_blocks": 2,
    "num_dec_blocks": 2,
    "hidden_dim": 128,
    "embedding_dim": 64,
    "num_latent_tokens": 4,
    "latent_dim": 16,
    "activation_type": "gelu"}

# Setting -> values to try, for every window. The baseline's own value is
# skipped automatically. Only the two settings that made a difference in the
# first study are kept:
#   num_latent_tokens  8 was the best change at 24 hours.
#   num_enc_blocks     1 was slightly better than 2.
# Dropped: 1 or 2 latent tokens and latent_dim 4 or 8 (worse), latent_dim 32
# and 3 blocks (no gain), decoder depth (no clear effect).
ABLATIONS = {
    "num_latent_tokens": [4, 8],
    "num_enc_blocks": [1, 2]}

# Named combinations that change MORE than one setting from the baseline,
# also written for every window. Name -> the settings that differ. Empty by
# default. To test the two changes together, use:
#   EXTRAS = {"tokens8_num_enc_blocks-1": {"num_latent_tokens": 8, "num_enc_blocks": 1}}
EXTRAS = {}


def check_config(name, config):
    """Stop early on settings the model or loader cannot run."""
    if config["embedding_dim"] % config["num_heads"] != 0:
        raise ValueError(f"{name}: embedding_dim must be divisible by num_heads")
    # The decoder applies LayerNorm to each bottleneck token, which removes
    # its mean and scale: 1 value carries nothing and 2 carry one bit.
    if config["latent_dim"] < 4:
        raise ValueError(f"{name}: latent_dim below 4 is degenerate")
    if config["num_latent_tokens"] < 1:
        raise ValueError(f"{name}: needs at least one latent token")
    # The loader accepts only these windows (batadal_dataset.py).
    if config["window"] not in ALLOWED_WINDOWS:
        raise ValueError(f"{name}: window must be one of {ALLOWED_WINDOWS}")
    if config["num_enc_blocks"] < 1 or config["num_dec_blocks"] < 1:
        raise ValueError(f"{name}: needs at least one encoder and one decoder block")
    if config["max_global_steps"] < 1:
        raise ValueError(f"{name}: max_global_steps must be positive")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ablation config files.")

    parser.add_argument(
        "--out-dir",
        help="Folder to write the config files to.",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--max-global-steps",
        help="Override the training length for every config (shorter ablations).",
        type=int,
        default=None)

    args = vars(parser.parse_args())

    out_dir = args["out_dir"]

    total = 0
    for window in WINDOWS:
        baseline = dict(BASELINE)
        baseline["window"] = window
        if args["max_global_steps"] is not None:
            baseline["max_global_steps"] = args["max_global_steps"]

        # Baseline first, then one config per changed setting.
        configs = {"baseline": baseline}
        for setting, values in ABLATIONS.items():
            if setting not in baseline or setting == "window":
                raise KeyError(f"{setting} cannot be ablated here")
            for value in values:
                if value == baseline[setting]:
                    continue  # Same as the baseline run.
                config = dict(baseline)
                config[setting] = value
                configs[f"{setting}-{value}"] = config

        # Then the named combinations.
        for name, changes in EXTRAS.items():
            if name in configs:
                raise KeyError(f"{name}: an extra has the same name as another config")
            unknown = [setting for setting in changes if setting not in baseline or setting == "window"]
            if len(unknown) > 0:
                raise KeyError(f"{name}: {unknown} cannot be set in an extra")
            if all(baseline[setting] == value for setting, value in changes.items()):
                raise ValueError(f"{name}: an extra must differ from the baseline")
            config = dict(baseline)
            config.update(changes)
            configs[name] = config

        window_dir = os.path.join(out_dir, f"w{window}")
        os.makedirs(window_dir, exist_ok=True)
        for name, config in configs.items():
            check_config(f"w{window}/{name}", config)
            config_path = os.path.join(window_dir, f"{name}.json")
            with open(config_path, "w") as config_f:
                json.dump(config, config_f, indent=4)
                config_f.write("\n")
            print(f"Wrote {config_path}")
        total += len(configs)

    print(f"{total} config files: {total // len(WINDOWS)} for each of {len(WINDOWS)} windows, in {out_dir}")

if __name__ == "__main__":
    main()
