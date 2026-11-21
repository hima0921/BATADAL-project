import os
import sys
import csv
import glob
import json
import pathlib
import argparse
import subprocess

import torch

"""
Runs train -> calibrate -> evaluate for every config file and every seed, then
collects the results into one CSV and draws the ablation plots.

    python run_ablations.py \
        --configs-dir configs/ablation \
        --dataset-path data/processed \
        --runs-dir runs/ablation \
        --seeds 0 1 2

Every configuration and seed is handled on its own, in
<runs-dir>/<config name>/seed<k>/:
  train      its own model            best_model.pt, recon_loss.json
  calibrate  its own thresholds       calibration/calibration.pt
  evaluate   its own Dataset 2 scores evaluation/metrics.json
Thresholds are never shared between runs: a model that reconstructs better
needs a lower threshold, so a shared one would make the comparison meaningless.

Dataset 2 is read from dataset2_eval.csv in --dataset-path. The Test set is
never read.

--configs-dir can also hold one sub-folder of configs per group, which is what
make_configs.py writes for an even study (w12/ and w24/). Every group is then
run in turn, into <runs-dir>/<group>/, with its own results.csv and plots. A
combined <runs-dir>/results.csv (with a `group` column) and plots that put the
groups side by side are written at the end.

A stage that already finished is skipped, so the script can be stopped and
started again. A stage is redone when the file it depends on is newer than its
result (a retrained model, a new calibration), or with --force.
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


def is_fresh(result_path, *source_paths):
    """True when the result exists and is newer than everything it was made from."""
    if not os.path.exists(result_path):
        return False
    made = os.path.getmtime(result_path)
    return all(os.path.getmtime(path) <= made for path in source_paths if os.path.exists(path))


def find_groups(configs_dir):
    """
    [(group name, configs folder)]. A folder with config files directly in it
    is one unnamed group; otherwise every sub-folder with config files is one.
    """
    if len(glob.glob(os.path.join(configs_dir, "*.json"))) > 0:
        return [("", str(configs_dir))]
    groups = []
    for sub_dir in sorted(glob.glob(os.path.join(configs_dir, "*", ""))):
        if len(glob.glob(os.path.join(sub_dir, "*.json"))) > 0:
            groups.append((os.path.basename(os.path.dirname(sub_dir)), sub_dir.rstrip(os.sep)))
    return groups


def combine_results(runs_dir, group_names):
    """One results.csv for all groups, with the group as the first column."""
    rows, columns = [], ["group"]
    for group_name in group_names:
        group_path = os.path.join(runs_dir, group_name, "results.csv")
        if not os.path.exists(group_path):
            continue
        with open(group_path, "r", newline="") as group_f:
            for row in csv.DictReader(group_f):
                row = {"group": group_name, **row}
                rows.append(row)
                for key in row:
                    if key not in columns:
                        columns.append(key)
    if len(rows) == 0:
        return False

    results_path = os.path.join(runs_dir, "results.csv")
    with open(results_path, "w", newline="") as results_f:
        writer = csv.DictWriter(results_f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Combined {len(rows)} runs from {len(group_names)} groups into {results_path}")
    return True


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
        help="Folder of JSON config files (from make_configs.py), or a folder of such folders.",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--dataset-path",
        help="Folder of processed files from Pipeline.ipynb (data/processed).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--dataset2-path",
        help="Dataset 2 with labels. Default: dataset2_eval.csv in --dataset-path.",
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
        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--checkpoint-steps",
        help="Steps between checkpoints during training.",
        type=int,
        default=2_500)
    parser.add_argument(
        "--top-k",
        help="Calibration: number of highest sensor scores averaged. Default: the script's own.",
        type=int,
        default=None)
    parser.add_argument(
        "--merge-gap-hours",
        help="Calibration: alarm runs this many quiet hours apart are one episode.",
        type=int,
        default=None)
    parser.add_argument(
        "--grace-hours",
        help="Evaluation: hours after each attack left out of the figures.",
        type=int,
        default=None)
    parser.add_argument(
        "--force",
        action='store_true',
        help="Redo the selected stages even if their results exist.")
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
        help="Columns of results.csv to plot, or 'all'. Default: the headline metrics.",
        nargs="+",
        default=None)
    parser.add_argument(
        "--plot-score",
        help="Hourly score the default plots show.",
        type=str,
        default="topk_percentile")
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
    force = args["force"]

    # The same calibration settings for every run, so runs stay comparable.
    calib_options = []
    if args["top_k"] is not None:
        calib_options += ["--top-k", str(args["top_k"])]
    if args["merge_gap_hours"] is not None:
        calib_options += ["--merge-gap-hours", str(args["merge_gap_hours"])]

    # One group, or one group per sub-folder (an even study: w12/, w24/).
    groups = find_groups(configs_dir)
    if len(groups) == 0:
        raise SystemExit(f"No config files found in {configs_dir} or its sub-folders")
    group_runs_dir = {name: os.path.join(runs_dir, name) if name else runs_dir
                      for name, _ in groups}

    # (group name, config file) for every configuration, group by group.
    config_jobs = []
    for group_name, group_configs_dir in groups:
        for config_path in sorted(glob.glob(os.path.join(group_configs_dir, "*.json"))):
            config_jobs.append((group_name, config_path))

    # The scripts are expected next to this file.
    here = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(here, "train_batadal_recon_model.py")
    calib_script = os.path.join(here, "calibrate_recon_model.py")
    eval_script = os.path.join(here, "evaluate_recon_model.py")

    total = len(config_jobs) * len(seeds)
    if len(groups) > 1:
        print(f"Groups: {', '.join(name for name, _ in groups)}")
    print(f"{len(config_jobs)} configs x {len(seeds)} seeds = {total} runs")

    failed = []
    for group_name, config_path in config_jobs:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        # Shown in the log and in the failure list.
        run_name = f"{group_name}/{config_name}" if group_name else config_name
        with open(config_path, "r") as config_f:
            max_global_steps = json.load(config_f)["max_global_steps"]

        for seed in seeds:
            run_dir = os.path.join(group_runs_dir[group_name], config_name, f"seed{seed}")
            best_model = os.path.join(run_dir, "best_model.pt")
            final_model = os.path.join(run_dir, f"{max_global_steps}_model.pt")
            calib_dir = os.path.join(run_dir, "calibration")
            eval_dir = os.path.join(run_dir, "evaluation")
            train_done = os.path.join(run_dir, "train_done.txt")
            calib_file = os.path.join(calib_dir, "calibration.pt")
            calib_json = os.path.join(calib_dir, "calibration.json")
            metrics_json = os.path.join(eval_dir, "metrics.json")
            print(f"\n=== {run_name} | seed {seed} ===")

            # Train. Finished when the marker file exists.
            if "train" in stages:
                if os.path.exists(train_done) and not force:
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
                        failed.append((run_name, seed, "train"))
                        continue
                    if not dry_run:
                        with open(train_done, "w") as done_f:
                            done_f.write("done\n")
                        if args["keep_only_best"]:
                            for path in glob.glob(os.path.join(run_dir, "*_model.pt")):
                                if os.path.basename(path) != "best_model.pt":
                                    os.remove(path)

            # Calibrate this model on its own. Finished when calibration.json
            # exists and is newer than the model it was made from.
            if "calibrate" in stages:
                if not force and is_fresh(calib_json, best_model):
                    print("calibrate: already done, skipping")
                elif not dry_run and not os.path.exists(best_model):
                    print("calibrate: no best_model.pt, skipping")
                else:
                    ok = run_command([
                        sys.executable, calib_script,
                        "--dataset-path", dataset_path,
                        "--model-checkpoint", best_model,
                        "--out-dir", calib_dir,
                        "--device", device] + calib_options, dry_run)
                    if not ok:
                        failed.append((run_name, seed, "calibrate"))
                        continue

            # Evaluate on Dataset 2. Finished when metrics.json exists and is
            # newer than the model and the calibration it was made from.
            if "evaluate" in stages:
                if not force and is_fresh(metrics_json, best_model, calib_file):
                    print("evaluate: already done, skipping")
                elif not dry_run and not os.path.exists(best_model):
                    print("evaluate: no best_model.pt, skipping")
                else:
                    command = [
                        sys.executable, eval_script,
                        "--dataset-path", dataset_path,
                        "--model-checkpoint", best_model,
                        "--out-dir", eval_dir,
                        "--device", device]
                    # Without this the script reads dataset2_eval.csv from
                    # --dataset-path.
                    if dataset2_path is not None:
                        command += ["--dataset2-path", str(dataset2_path)]
                    if args["grace_hours"] is not None:
                        command += ["--grace-hours", str(args["grace_hours"])]
                    # Use this run's frozen thresholds from the calibrate stage.
                    # Without them the script calibrates by itself, with the
                    # same settings.
                    if dry_run or os.path.exists(calib_file):
                        command += ["--calibration", calib_file]
                    else:
                        command += calib_options
                    ok = run_command(command, dry_run)
                    if not ok:
                        failed.append((run_name, seed, "evaluate"))

    print()
    if not dry_run:
        # One results.csv per group, then one for all groups together.
        collected = False
        for group_name, _ in groups:
            collected = collect_results(group_runs_dir[group_name]) or collected
        if collected and len(groups) > 1:
            combine_results(runs_dir, [name for name, _ in groups])

        # Plots last, and never at the cost of the results: a plotting problem
        # is reported but does not stop the run summary below.
        if collected and not args["no_plots"]:
            try:
                from plot_ablations import make_all_plots
                make_all_plots(
                    runs_dir=runs_dir,
                    configs_dir=configs_dir,
                    metrics=args["plot_metrics"],
                    score=args["plot_score"])
            except (Exception, SystemExit) as error:
                print(f"Plots were not drawn: {error}")
                print("Redraw them with: python plot_ablations.py "
                      f"--runs-dir {runs_dir} --configs-dir {configs_dir}")
    for run_name, seed, stage in failed:
        print(f"FAILED: {run_name} seed {seed} at {stage}")

if __name__ == "__main__":
    main()
