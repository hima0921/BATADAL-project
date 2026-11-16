import os
import csv
import glob
import json
import pathlib
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")  # Write image files without needing a display.
import matplotlib.pyplot as plt

"""
Plots the ablation results written by run_ablations.py.

run_ablations.py calls make_plots() from this file when it finishes, so this
script only needs running by hand to redraw the plots.

    python plot_ablations.py \
        --runs-dir runs/ablation \
        --configs-dir configs/ablation

Reads <runs-dir>/results.csv and writes to <runs-dir>/plots/:
  ablation_<metric>.png   one panel per ablated setting: every seed as a dot,
                          the mean across seeds as a line, the baseline marked.
  ablation_val_curves.png validation loss during training, one panel per
                          setting, one line per value (mean across seeds).
  ablation_train_val_curves.png
                          train and validation loss during training, one panel
                          per configuration (thin lines: seeds, thick: mean).

Config names are expected as "baseline" and "<setting>-<value>", which is what
make_configs.py writes.
"""

# One hue for single-series panels; a fixed order for the multi-line panels.
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Columns in results.csv that are not results.
NOT_METRICS = {"config", "seed", "best_step"}


def style_axis(axis):
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)


def to_number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def load_results(results_path):
    with open(results_path, "r", newline="") as results_f:
        return list(csv.DictReader(results_f))


def group_by_setting(rows, baseline_config):
    """
    {setting: {value: [rows]}}. The baseline's rows are added to every setting
    at the baseline's own value, so each panel includes the reference point.
    """
    baseline_rows = [row for row in rows if row["config"] == "baseline"]

    groups = {}
    for row in rows:
        if row["config"] == "baseline" or "-" not in row["config"]:
            continue
        setting, value = row["config"].rsplit("-", 1)
        number = to_number(value)
        groups.setdefault(setting, {}).setdefault(
            number if number is not None else value, []).append(row)

    for setting, by_value in groups.items():
        if setting in baseline_config and len(baseline_rows) > 0:
            by_value.setdefault(baseline_config[setting], []).extend(baseline_rows)
    return groups


def sorted_values(by_value):
    return sorted(by_value, key=lambda v: (isinstance(v, str), v))


def format_value(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def plot_metric(groups, baseline_config, metric, out_dir, log_scale):
    settings = [s for s in groups if any(
        to_number(row.get(metric)) is not None
        for value_rows in groups[s].values() for row in value_rows)]
    if len(settings) == 0:
        return None

    fig, axes = plt.subplots(
        1, len(settings),
        figsize=(3.4 * len(settings), 3.6),
        sharey=True,  # Same metric in every panel, so one shared scale.
        squeeze=False)

    for axis, setting in zip(axes[0], settings):
        by_value = groups[setting]
        values = sorted_values(by_value)
        positions = np.arange(len(values))

        means = []
        for position, value in zip(positions, values):
            numbers = [to_number(row.get(metric)) for row in by_value[value]]
            numbers = [n for n in numbers if n is not None]
            means.append(np.mean(numbers) if len(numbers) > 0 else np.nan)

            # Every seed as its own dot, so the spread is visible.
            axis.scatter(
                [position] * len(numbers), numbers,
                s=26, color=SERIES[0], alpha=0.45, linewidths=0, zorder=2)

        # Mean across seeds.
        axis.plot(
            positions, means,
            color=SERIES[0], linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, zorder=3)

        # The baseline is marked with a ring and named under its tick, so it
        # is not identified by colour alone and the label cannot hide a dot.
        baseline_value = baseline_config.get(setting)
        tick_labels = [format_value(v) for v in values]
        if baseline_value in values:
            index = values.index(baseline_value)
            axis.scatter(
                [positions[index]], [means[index]],
                s=190, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
            tick_labels[index] += "\n(baseline)"

        axis.set_xticks(positions)
        axis.set_xticklabels(tick_labels)
        axis.set_xlim(-0.5, len(values) - 0.5)
        axis.set_xlabel(setting, color=INK, fontsize=10)
        if log_scale:
            axis.set_yscale("log")
        style_axis(axis)

    axes[0][0].set_ylabel(metric, color=INK, fontsize=10)
    fig.suptitle(
        f"{metric} by ablated setting (dots: seeds, line: mean)",
        x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    plot_path = os.path.join(out_dir, f"ablation_{metric}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def load_val_curve(runs_dir, config_name):
    """Mean validation curve across the seeds of one config."""
    curves = []
    for loss_path in sorted(glob.glob(
            os.path.join(runs_dir, config_name, "seed*", "recon_loss.json"))):
        with open(loss_path, "r") as loss_f:
            history = json.load(loss_f)
        curves.append((history["steps"], history["val"]))
    if len(curves) == 0:
        return None, None

    # Seeds share the same checkpoint steps; trim to the shortest in case one
    # run stopped early.
    length = min(len(steps) for steps, _ in curves)
    steps = curves[0][0][:length]
    mean_val = np.mean([val[:length] for _, val in curves], axis=0)
    return steps, mean_val


def plot_val_curves(groups, baseline_config, runs_dir, out_dir):
    settings = list(groups)
    if len(settings) == 0:
        return None

    fig, axes = plt.subplots(
        1, len(settings),
        figsize=(3.8 * len(settings), 3.6),
        sharey=True,
        squeeze=False)

    drew_any = False
    for axis, setting in zip(axes[0], settings):
        values = sorted_values(groups[setting])
        line_ends = []
        for index, value in enumerate(values):
            is_baseline = baseline_config.get(setting) == value
            config_name = "baseline" if is_baseline else f"{setting}-{format_value(value)}"
            steps, mean_val = load_val_curve(runs_dir, config_name)
            if steps is None:
                continue
            drew_any = True

            color = SERIES[index % len(SERIES)]
            label = format_value(value) + (" (baseline)" if is_baseline else "")
            axis.plot(steps, mean_val, color=color, linewidth=2, label=label)
            line_ends.append((mean_val[-1], steps[-1], format_value(value)))

        # Direct label at the end of each line, in neutral ink. If any two
        # lines end too close together, the whole panel is left to its legend:
        # a label on only some of a bundle of lines would mislead.
        ends = sorted(end_value for end_value, _, _ in line_ends)
        crowded = any(b / a < 1.12 for a, b in zip(ends, ends[1:]))
        if not crowded:
            for end_value, end_step, text in line_ends:
                axis.annotate(
                    text, (end_step, end_value),
                    textcoords="offset points", xytext=(5, 0),
                    va="center", fontsize=8, color=MUTED)

        axis.set_yscale("log")
        axis.set_xlabel("global steps", color=INK, fontsize=10)
        axis.set_title(setting, loc="left", fontsize=10, color=INK)
        axis.margins(x=0.12)
        style_axis(axis)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8, frameon=False, labelcolor=MUTED)

    if not drew_any:
        plt.close(fig)
        return None

    axes[0][0].set_ylabel("validation reconstruction MSE", color=INK, fontsize=10)
    fig.suptitle(
        "Validation loss during training (mean across seeds)",
        x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    plot_path = os.path.join(out_dir, "ablation_val_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def load_run_curves(runs_dir, config_name):
    """[(steps, train, val)] for every seed of one config."""
    curves = []
    for loss_path in sorted(glob.glob(
            os.path.join(runs_dir, config_name, "seed*", "recon_loss.json"))):
        with open(loss_path, "r") as loss_f:
            history = json.load(loss_f)
        curves.append((history["steps"], history["train"], history["val"]))
    return curves


def plot_train_val_curves(config_names, runs_dir, out_dir, columns=4):
    """Train and validation loss for each configuration, on one shared scale."""
    with_curves = [(name, load_run_curves(runs_dir, name)) for name in config_names]
    with_curves = [(name, curves) for name, curves in with_curves if len(curves) > 0]
    if len(with_curves) == 0:
        return None

    columns = min(columns, len(with_curves))
    n_rows = int(np.ceil(len(with_curves) / columns))
    fig, axes = plt.subplots(
        n_rows, columns,
        figsize=(3.6 * columns, 2.9 * n_rows),
        sharex=True, sharey=True,  # One scale, so panels can be compared.
        squeeze=False)
    flat_axes = axes.ravel()

    for axis, (name, curves) in zip(flat_axes, with_curves):
        length = min(len(steps) for steps, _, _ in curves)
        steps = curves[0][0][:length]
        for index, (label, column) in enumerate((("train", 1), ("validation", 2))):
            color = SERIES[index]
            for curve in curves:  # Every seed, faintly.
                axis.plot(steps, curve[column][:length], color=color, linewidth=0.8, alpha=0.35)
            mean_curve = np.mean([curve[column][:length] for curve in curves], axis=0)
            axis.plot(steps, mean_curve, color=color, linewidth=2, label=label)
        axis.set_yscale("log")
        axis.set_title(f"{name}  ({len(curves)} seeds)", loc="left", fontsize=9, color=INK)
        style_axis(axis)

    for axis in flat_axes[len(with_curves):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("global steps", color=INK, fontsize=10)
    for row_axes in axes:
        row_axes[0].set_ylabel("reconstruction MSE", color=INK, fontsize=10)

    # One legend for the whole figure; the two series are the same everywhere.
    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.995),
               ncol=2, fontsize=9, frameon=False, labelcolor=MUTED)
    fig.suptitle(
        "Train and validation loss per configuration (thin lines: seeds, thick: mean)",
        x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (2.9 * n_rows + 0.5)))

    plot_path = os.path.join(out_dir, "ablation_train_val_curves.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def print_summary(groups, metrics):
    """Mean and spread per setting value, as a plain table."""
    for metric in metrics:
        print(f"\n{metric}")
        print(f"  {'setting':<18}{'value':<10}{'seeds':<7}{'mean':<12}{'min':<12}{'max':<12}")
        for setting, by_value in groups.items():
            for value in sorted_values(by_value):
                numbers = [to_number(row.get(metric)) for row in by_value[value]]
                numbers = [n for n in numbers if n is not None]
                if len(numbers) == 0:
                    continue
                print(f"  {setting:<18}{format_value(value):<10}{len(numbers):<7}"
                      f"{np.mean(numbers):<12.5g}{np.min(numbers):<12.5g}{np.max(numbers):<12.5g}")


def make_plots(runs_dir, configs_dir, metrics=None, log_scale=False):
    """
    Draw every ablation plot and print the summary table. Called by main()
    below and by run_ablations.py after it has collected results.csv.
    """
    runs_dir, configs_dir = str(runs_dir), str(configs_dir)

    results_path = os.path.join(runs_dir, "results.csv")
    if not os.path.exists(results_path):
        raise SystemExit(f"{results_path} not found. Run run_ablations.py first.")
    baseline_path = os.path.join(configs_dir, "baseline.json")
    if not os.path.exists(baseline_path):
        raise SystemExit(f"{baseline_path} not found. Run make_configs.py first.")

    rows = load_results(results_path)
    with open(baseline_path, "r") as baseline_f:
        baseline_config = json.load(baseline_f)
    baseline_config = {
        key: float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        for key, value in baseline_config.items()}

    groups = group_by_setting(rows, baseline_config)
    if len(groups) == 0:
        raise SystemExit("No ablation runs found in results.csv.")

    if metrics is None:
        columns = []
        for row in rows:
            for column in row:
                if column not in columns:
                    columns.append(column)
        metrics = [
            column for column in columns
            if column not in NOT_METRICS
            and any(to_number(row.get(column)) is not None for row in rows)]

    out_dir = os.path.join(runs_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    for metric in metrics:
        plot_path = plot_metric(
            groups=groups,
            baseline_config=baseline_config,
            metric=metric,
            out_dir=out_dir,
            log_scale=log_scale)
        print(f"Saved {plot_path}" if plot_path else f"No numeric values for {metric}, skipped")

    plot_path = plot_val_curves(
        groups=groups,
        baseline_config=baseline_config,
        runs_dir=runs_dir,
        out_dir=out_dir)
    print(f"Saved {plot_path}" if plot_path else "No recon_loss.json files found, skipped curves")

    # Baseline first, then the other configurations in name order.
    config_names = sorted({row["config"] for row in rows}, key=lambda n: (n != "baseline", n))
    plot_path = plot_train_val_curves(
        config_names=config_names,
        runs_dir=runs_dir,
        out_dir=out_dir)
    print(f"Saved {plot_path}" if plot_path else "No recon_loss.json files found, skipped train/val curves")

    print_summary(groups, metrics)
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Plot ablation results.")

    parser.add_argument(
        "--runs-dir",
        help="Folder given to run_ablations.py (holds results.csv).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--configs-dir",
        help="Folder of config files (holds baseline.json).",
        type=pathlib.Path,
        required=True)
    parser.add_argument(
        "--metrics",
        help="Columns of results.csv to plot. Default: every numeric column.",
        nargs="+",
        default=None)
    parser.add_argument(
        "--log-scale",
        action='store_true',
        help="Use a log scale for the metric axis.")

    args = vars(parser.parse_args())

    make_plots(
        runs_dir=args["runs_dir"],
        configs_dir=args["configs_dir"],
        metrics=args["metrics"],
        log_scale=args["log_scale"])

if __name__ == "__main__":
    main()
