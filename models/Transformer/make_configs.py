import os
import json
import pathlib
import argparse

"""
Writes one JSON config per ablation setting, changing ONE setting at a time
from the baseline.

    python make_configs.py --out-dir configs/ablation
    python make_configs.py --out-dir configs/ablation --max-global-steps 10000

Edit BASELINE and ABLATIONS below to change what is generated. Fix this list
before running anything, so settings are not chosen after seeing results.
"""

# The reference model. Same keys as 24_batadal_recon_config.json.
BASELINE = {
    "max_global_steps": 15_000,
    "model_lr": 3e-3,
    "window": 24,
    "dropout": 0.1,
    "num_heads": 8,
    "num_enc_blocks": 2,
    "num_dec_blocks": 2,
    "hidden_dim": 128,
    "embedding_dim": 64,
    "num_encodings": 4,
    "bottleneck_dim": 16,
    "activation_type": "gelu"}

# Setting -> values to try. The baseline's own value is skipped automatically.
ABLATIONS = {
    "window": [12, 24],
    "num_encodings": [1, 2, 4, 8],
    "bottleneck_dim": [4, 8, 16, 32],
    "num_enc_blocks": [1, 2, 3],
    "num_dec_blocks": [1, 2, 3]}


def check_config(name, config):
    """Stop early on settings the model or loader cannot run."""
    if config["embedding_dim"] % config["num_heads"] != 0:
        raise ValueError(f"{name}: embedding_dim must be divisible by num_heads")
    # The decoder applies LayerNorm to each bottleneck token, which removes
    # its mean and scale: 1 value carries nothing and 2 carry one bit.
    if config["bottleneck_dim"] < 4:
        raise ValueError(f"{name}: bottleneck_dim below 4 is degenerate")
    # The loader's gap between blocks is 24 hours and must cover the window.
    if config["window"] > 24:
        raise ValueError(f"{name}: window above 24 needs a larger gap_hours in the loader")
    if config["num_enc_blocks"] < 1 or config["num_dec_blocks"] < 1:
        raise ValueError(f"{name}: needs at least one encoder and one decoder block")


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
    os.makedirs(out_dir, exist_ok=True)

    baseline = dict(BASELINE)
    if args["max_global_steps"] is not None:
        baseline["max_global_steps"] = args["max_global_steps"]

    # Baseline first, then one config per changed setting.
    configs = {"baseline": baseline}
    for setting, values in ABLATIONS.items():
        if setting not in baseline:
            raise KeyError(f"{setting} is not a config setting")
        for value in values:
            if value == baseline[setting]:
                continue  # Same as the baseline run.
            config = dict(baseline)
            config[setting] = value
            configs[f"{setting}-{value}"] = config

    for name, config in configs.items():
        check_config(name, config)
        config_path = os.path.join(out_dir, f"{name}.json")
        with open(config_path, "w") as config_f:
            json.dump(config, config_f, indent=4)
            config_f.write("\n")
        print(f"Wrote {config_path}")

    print(f"{len(configs)} config files in {out_dir}")

if __name__ == "__main__":
    main()
