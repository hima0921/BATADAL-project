import os
import sys
import csv
import glob
import json
import pathlib
import argparse
import subprocess

"""
Runs train -> calibrate -> evaluate for every config file and every seed, then
collects the results into one CSV and draws the ablation plots.

    python run_ablations.py \
        --configs-dir configs/ablation \
        --dataset-path "data/Training dataset 1.csv" \
        --runs-dir runs/ablation \
        --seeds 0 1 2

Each run goes to <runs-dir>/<config name>/seed<k>/. A stage that already
finished is skipped, so the script can be stopped and started again.
Use --dry-run to print the commands without running them.

Plots go to <runs-dir>/plots/ and come from plot_ablations.py, which must sit
next to this file. Use --no-plots to skip them.
"""

STAGES = ["train", "calibrate", "evaluate"]


def run_command(command, dry_run):
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if dry_run:
        return True
    return subprocess.run(command).returncode == 0


def collect_results(runs_dir):
    """One row per run, from the JSON files each stage writes."""
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*", "seed*"))):
        row = {
            "config": os.path.basename(os.path.dirname(run_dir)),
            "seed": os.path.basename(run_dir).replace("seed", "")}

        loss_path = os.path.join(run_dir, "recon_loss.json")
        if os.path.exists(loss_path):
            with open(loss_path, "r") as loss_f:
                history = json.load(loss_f)
            row["best_val_loss"] = history["best_val"]
            row["best_step"] = history["best_step"]

        calib_path = os.path.join(run_dir, "calibration", "calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r") as calib_f:
                for key, value in json.load(calib_f).items():
                    # One "<score>_<figure>" column per calibrated score.
                    if any(key.startswith(prefix) for prefix in ("topk_", "mean_")):
                        row[key] = value

        # The evaluation stage should write a flat JSON of its metrics here.
        eval_path = os.path.join(run_dir, "evaluation", "metrics.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as eval_f:
                for key, value in json.load(eval_f).items():
                    if not isinstance(value, (list, dict)):
                        row[key] = value
        rows.append(row)

    if len(rows) == 0:
        print("No finished runs to collect.")
        return False

    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    results_path = os.path.join(runs_dir, "results.csv")
    with open(results_path, "w", newline="") as results_f:
        writer = csv.DictWriter(results_f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected {len(rows)} runs into {results_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train, calibrate and evaluate every config and seed.")

    parser.add_argument(
        "--configs-dir",
        help="Folder of JSON config files (from make_configs.py).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--dataset-path",
        help="File path for the normal BATADAL training CSV (Training dataset 1).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--dataset2-path",
        help="File path for Training dataset 2 (needed by the evaluate stage).",
        type=pathlib.Path,
        default=None)
    parser.add_argument(
        "--runs-dir",
        help="Folder that will hold one sub-folder per config and seed.",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--seeds",
        help="Seeds to run for every config.",
        type=int,
        nargs="+",
        default=[0, 1, 2])
    parser.add_argument(
        "--stages",
        help="Stages to run, in order.",
        choices=STAGES,
        nargs="+",
        default=STAGES)
    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--checkpoint-steps",
        help="Steps between checkpoints during training.",
        type=int,
        default=2_500)
    parser.add_argument(
        "--keep-only-best",
        action='store_true',
        help="After training, delete the numbered checkpoints and keep best_model.pt.")
    parser.add_argument(
        "--no-plots",
        action='store_true',
        help="Do not draw the ablation plots at the end.")
    parser.add_argument(
        "--plot-metrics",
        help="Columns of results.csv to plot. Default: every numeric column.",
        nargs="+",
        default=None)
    parser.add_argument(
        "--dry-run",
        action='store_true',
        help="Print the commands without running them.")

    args = vars(parser.parse_args())

    configs_dir = args["configs_dir"]
    dataset_path = str(args["dataset_path"])
    dataset2_path = args["dataset2_path"]
    runs_dir = str(args["runs_dir"])
    seeds = args["seeds"]
    stages = args["stages"]
    device = args["device"]
    dry_run = args["dry_run"]

    config_paths = sorted(glob.glob(os.path.join(configs_dir, "*.json")))
    if len(config_paths) == 0:
        raise SystemExit(f"No config files found in {configs_dir}")

    # The scripts are expected next to this file.
    here = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(here, "train_batadal_recon_model.py")
    calib_script = os.path.join(here, "calibrate_recon_model.py")
    eval_script = os.path.join(here, "evaluate_recon_model.py")

    total = len(config_paths) * len(seeds)
    print(f"{len(config_paths)} configs x {len(seeds)} seeds = {total} runs")

    failed = []
    for config_path in config_paths:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        with open(config_path, "r") as config_f:
            max_global_steps = json.load(config_f)["max_global_steps"]

        for seed in seeds:
            run_dir = os.path.join(runs_dir, config_name, f"seed{seed}")
            best_model = os.path.join(run_dir, "best_model.pt")
            final_model = os.path.join(run_dir, f"{max_global_steps}_model.pt")
            calib_dir = os.path.join(run_dir, "calibration")
            eval_dir = os.path.join(run_dir, "evaluation")
            train_done = os.path.join(run_dir, "train_done.txt")
            print(f"\n=== {config_name} | seed {seed} ===")

            # Train. Finished when the marker file exists.
            if "train" in stages:
                if os.path.exists(train_done):
                    print("train: already done, skipping")
                else:
                    ok = run_command([
                        sys.executable, train_script,
                        "--dataset-path", dataset_path,
                        "-c", config_path,
                        "--out-dir", run_dir,
                        "--seed", str(seed),
                        "--device", device,
                        "--checkpoint-steps", str(args["checkpoint_steps"]),
                        # Four learning-rate halvings' worth of steps per run,
                        # whatever the run length.
                        "--lr-steps", str(max(1, max_global_steps // 4))], dry_run)
                    if not ok:
                        failed.append((config_name, seed, "train"))
                        continue
                    if not dry_run:
                        with open(train_done, "w") as done_f:
                            done_f.write("done\n")
                        if args["keep_only_best"]:
                            for path in glob.glob(os.path.join(run_dir, "*_model.pt")):
                                if os.path.basename(path) != "best_model.pt":
                                    os.remove(path)

            # Calibrate. Finished when calibration.json exists.
            if "calibrate" in stages:
                if os.path.exists(os.path.join(calib_dir, "calibration.json")):
                    print("calibrate: already done, skipping")
                elif not dry_run and not os.path.exists(best_model):
                    print("calibrate: no best_model.pt, skipping")
                else:
                    ok = run_command([
                        sys.executable, calib_script,
                        "--dataset-path", dataset_path,
                        "--model-checkpoint", best_model,
                        "--out-dir", calib_dir,
                        "--device", device], dry_run)
                    if not ok:
                        failed.append((config_name, seed, "calibrate"))
                        continue

            # Evaluate on Dataset 2. Finished when metrics.json exists.
            if "evaluate" in stages:
                if not os.path.exists(eval_script):
                    print("evaluate: evaluate_recon_model.py not found, skipping")
                elif dataset2_path is None:
                    print("evaluate: no --dataset2-path given, skipping")
                elif os.path.exists(os.path.join(eval_dir, "metrics.json")):
                    print("evaluate: already done, skipping")
                else:
                    command = [
                        sys.executable, eval_script,
                        "--dataset-path", dataset_path,
                        "--dataset2-path", str(dataset2_path),
                        "--model-checkpoint", best_model,
                        "--out-dir", eval_dir,
                        "--device", device]
                    # Use the frozen thresholds from the calibrate stage.
                    calibration_file = os.path.join(calib_dir, "calibration.pt")
                    if dry_run or os.path.exists(calibration_file):
                        command += ["--calibration", calibration_file]
                    ok = run_command(command, dry_run)
                    if not ok:
                        failed.append((config_name, seed, "evaluate"))

    print()
    if not dry_run:
        collected = collect_results(runs_dir)

        # Plots last, and never at the cost of the results: a plotting problem
        # is reported but does not stop the run summary below.
        if collected and not args["no_plots"]:
            try:
                from plot_ablations import make_plots
                make_plots(
                    runs_dir=runs_dir,
                    configs_dir=configs_dir,
                    metrics=args["plot_metrics"])
            except (Exception, SystemExit) as error:
                print(f"Plots were not drawn: {error}")
                print("Redraw them with: python plot_ablations.py "
                      f"--runs-dir {runs_dir} --configs-dir {configs_dir}")
    for config_name, seed, stage in failed:
        print(f"FAILED: {config_name} seed {seed} at {stage}")

if __name__ == "__main__":
    main()
