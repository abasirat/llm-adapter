#!/usr/bin/env python3
"""
Plot training statistics and evaluation results.

Usage:
    python plot_stats.py --config configs/plots/training_plots.yaml

The config YAML controls output format, axis labels, ticks, and which metrics
to plot.  Add new entries under `figures` to extend to any metric logged in
history.jsonl or any key in results.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_history(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_eval(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Smoothing ────────────────────────────────────────────────────────────────

def smooth_ema(values: list, alpha: float) -> list:
    """Exponential moving average smoothing."""
    result = []
    s = None
    for v in values:
        s = v if s is None else alpha * v + (1.0 - alpha) * s
        result.append(s)
    return result


def smooth_window(values: list, window: int) -> list:
    """Uniform moving average via convolution."""
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


def apply_smoothing(values: list, smooth_cfg) -> list:
    if not smooth_cfg:
        return values
    method = smooth_cfg.get("method", "ema")
    if method == "ema":
        return smooth_ema(values, smooth_cfg.get("alpha", 0.1))
    if method == "window":
        return smooth_window(values, smooth_cfg.get("window", 10))
    return values


# ─── Axis helpers ─────────────────────────────────────────────────────────────

def apply_axis_settings(ax: plt.Axes, axis_cfg: dict, which: str) -> None:
    if not axis_cfg:
        return

    label = axis_cfg.get("label")
    if label:
        (ax.set_xlabel if which == "x" else ax.set_ylabel)(label)

    scale = axis_cfg.get("scale", "linear")
    (ax.set_xscale if which == "x" else ax.set_yscale)(scale)

    lim = axis_cfg.get("lim")
    if lim:
        (ax.set_xlim if which == "x" else ax.set_ylim)(lim)

    ticks_cfg = axis_cfg.get("ticks")
    if ticks_cfg:
        values = ticks_cfg.get("values")
        if values is None:
            start = ticks_cfg.get("start")
            stop = ticks_cfg.get("stop")
            step = ticks_cfg.get("step")
            if start is not None and stop is not None and step is not None:
                values = list(np.arange(start, stop + step / 2, step))

        if values is not None:
            (ax.set_xticks if which == "x" else ax.set_yticks)(values)

        fmt = ticks_cfg.get("format")
        if fmt:
            formatter = ticker.FormatStrFormatter(fmt)
            (ax.xaxis if which == "x" else ax.yaxis).set_major_formatter(formatter)

        rotation = ticks_cfg.get("rotation")
        if rotation is not None:
            plt.setp(
                (ax.get_xticklabels if which == "x" else ax.get_yticklabels)(),
                rotation=rotation,
            )


# ─── Shared figure finalization ────────────────────────────────────────────────

def _finalize_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    fig_cfg: dict,
    output_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    title = fig_cfg.get("title", "")
    if title:
        ax.set_title(title)

    apply_axis_settings(ax, fig_cfg.get("x_axis"), "x")
    apply_axis_settings(ax, fig_cfg.get("y_axis"), "y")

    legend_cfg = fig_cfg.get("legend")
    handles, _ = ax.get_legend_handles_labels()
    if handles and legend_cfg is not False:
        loc = "best"
        if isinstance(legend_cfg, dict):
            loc = legend_cfg.get("location", "best")
        ax.legend(loc=loc)

    grid_cfg = fig_cfg.get("grid", True)
    if isinstance(grid_cfg, dict):
        ax.grid(
            grid_cfg.get("enabled", True),
            axis=grid_cfg.get("axis", "both"),
            alpha=grid_cfg.get("alpha", 0.3),
        )
    elif grid_cfg:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    name = fig_cfg.get("name", "figure")
    out_path = os.path.join(output_dir, f"{name}.{fmt}")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─── Figure: history (line plots) ─────────────────────────────────────────────

def plot_history_figure(
    fig_cfg: dict,
    runs: list[dict],
    base_dir: str,
    output_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    figsize = fig_cfg.get("figsize", [8, 5])
    fig, ax = plt.subplots(figsize=figsize)

    x_field = fig_cfg.get("x_field", "step")
    y_fields = fig_cfg.get("y_fields", [])
    smooth_cfg = fig_cfg.get("smooth")

    prop_colors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    color_idx = 0

    for run in runs:
        history_path = run.get("history")
        if not history_path:
            continue
        if not os.path.isabs(history_path):
            history_path = os.path.join(base_dir, history_path)
        if not os.path.exists(history_path):
            print(f"  [warn] history not found: {history_path}")
            continue

        records = load_history(history_path)

        for y_cfg in y_fields:
            field = y_cfg["field"]
            pairs = [
                (r[x_field], r[field])
                for r in records
                if x_field in r and field in r and r[field] is not None
            ]
            if not pairs:
                print(f"  [warn] no data for field '{field}' in {history_path}")
                continue

            xs, ys = zip(*pairs)
            ys = apply_smoothing(list(ys), smooth_cfg)

            label = y_cfg.get("label", field)
            if len(runs) > 1:
                label = f"{run.get('label', '')} \u2013 {label}"

            color = y_cfg.get("color", prop_colors[color_idx % len(prop_colors)])
            linestyle = y_cfg.get("linestyle", "-")
            linewidth = y_cfg.get("linewidth", 1.5)
            ax.plot(xs, ys, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
            color_idx += 1

    _finalize_figure(fig, ax, fig_cfg, output_dir, fmt, dpi)


# ─── Figure: eval results (bar charts) ────────────────────────────────────────

def plot_eval_figure(
    fig_cfg: dict,
    runs: list[dict],
    base_dir: str,
    output_dir: str,
    fmt: str,
    dpi: int,
) -> None:
    figsize = fig_cfg.get("figsize", [6, 5])
    fig, ax = plt.subplots(figsize=figsize)

    metric = fig_cfg.get("metric")
    fields = fig_cfg.get("fields", [])
    if not fields:
        print(f"  [warn] no fields defined for eval figure '{fig_cfg.get('name')}'")
        plt.close(fig)
        return

    n_runs = len(runs)
    n_fields = len(fields)
    x_pos = np.arange(n_fields)
    bar_width = 0.8 / max(n_runs, 1)

    for run_idx, run in enumerate(runs):
        eval_path = run.get("eval")
        if not eval_path:
            continue
        if not os.path.isabs(eval_path):
            eval_path = os.path.join(base_dir, eval_path)
        if not os.path.exists(eval_path):
            print(f"  [warn] eval not found: {eval_path}")
            continue

        eval_data = load_eval(eval_path)
        # Navigate: results -> <metric> -> summary
        summary = eval_data.get("results", {}).get(metric, {}).get("summary", {})

        values = [summary.get(f["key"]) for f in fields]
        offset = (run_idx - n_runs / 2 + 0.5) * bar_width
        bars = ax.bar(
            x_pos + offset,
            [v if v is not None else 0 for v in values],
            width=bar_width,
            label=run.get("label", f"run {run_idx}"),
        )
        for bar, val in zip(bars, values):
            if val is not None:
                ax.annotate(
                    f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f["label"] for f in fields])

    _finalize_figure(fig, ax, fig_cfg, output_dir, fmt, dpi)


# ─── Dispatch table ───────────────────────────────────────────────────────────

PLOTTERS = {
    "history": plot_history_figure,
    "eval": plot_eval_figure,
}


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot training stats and evaluation results.")
    parser.add_argument("--config", required=True, help="Path to the plot config YAML.")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Base directory for relative paths (default: parent of the configs/ folder).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.exit(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Default base_dir: experiment root (3 levels up from configs/plots/<file>.yaml)
    base_dir = args.base_dir or str(config_path.parent.parent.parent)

    style = cfg.get("style")
    if style:
        plt.style.use(style)

    output_cfg = cfg.get("output", {})
    fmt = output_cfg.get("format", "png")
    dpi = output_cfg.get("dpi", 150)
    output_dir = output_cfg.get("directory", "outputs/plots")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    runs = cfg.get("runs", [])
    figures = cfg.get("figures", [])

    for fig_cfg in figures:
        name = fig_cfg.get("name", "<unnamed>")
        source = fig_cfg.get("source", "history")
        print(f"Plotting '{name}'  [source={source}]")
        plotter = PLOTTERS.get(source)
        if plotter is None:
            print(f"  [warn] unknown source '{source}', skipping.")
            continue
        plotter(fig_cfg, runs, base_dir, output_dir, fmt, dpi)

    print("Done.")


if __name__ == "__main__":
    main()
