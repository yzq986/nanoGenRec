#!/usr/bin/env python
"""Generate README result figures from experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "results" / "readme"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def moving_average(values: list[float], window: int = 7) -> list[float]:
    if not values:
        return []
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        window_values = values[start : idx + 1]
        out.append(sum(window_values) / len(window_values))
    return out


def extract_curve(run_name: str) -> tuple[list[float], list[float], list[float]]:
    rows = read_jsonl(REPO_ROOT / "experiments" / "ntp_checkpoints" / run_name / "train_log.jsonl")
    wall_mins = [float(row.get("wall_s", 0.0)) / 60.0 for row in rows]
    clip = [float(row.get("clip_fraction", 0.0)) for row in rows]
    behavior = [float(row.get("reward/behavior_mean", row.get("reward_mean", 0.0))) for row in rows]
    return wall_mins, moving_average(clip), moving_average(behavior)


def annotate_bars(ax, bars, values):
    for bar, value in zip(bars, values):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + 1.6,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_post_training_alignment() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 8.8), dpi=180)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.25)

    ax_bar = fig.add_subplot(grid[0, :])
    names = [
        "SFT\nEXP-020",
        "off-policy\nECPO",
        "on-policy\nECPO",
        "Feature SFT\nEXP-036",
        "SP/RF-DPO\nEXP-037/038B",
        "ECPO\nEXP-039B",
    ]
    recalls = [66.2, 2.0, 67.8, 59.0, 62.1, 65.7]
    colors = ["#4C78A8", "#D95F59", "#2A9D8F", "#7A6FAC", "#E9A441", "#2A9D8F"]
    bars = ax_bar.bar(names, recalls, color=colors, width=0.68)
    annotate_bars(ax_bar, bars, recalls)
    ax_bar.set_title("Post-training alignment: full-recall R@500", loc="left", fontsize=16, fontweight="bold")
    ax_bar.set_ylabel("Recall@500")
    ax_bar.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax_bar.set_ylim(0, 78)
    ax_bar.grid(axis="y", alpha=0.22)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.text(
        1,
        10,
        "candidate drift\ncollapse",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#8B2F2F",
    )
    ax_bar.text(
        2,
        72,
        "on-policy candidates\nrecover alignment",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#166B61",
    )

    runs = [
        ("EXP-029 on-policy ECPO", "exp029-ecpo-onpolicy-w003-r100", fig.add_subplot(grid[1, 0])),
        ("EXP-039B ECPO after DPO", "exp039b-ecpo-from-spdpo", fig.add_subplot(grid[1, 1])),
    ]
    for title, run_name, ax in runs:
        wall_mins, clip, behavior = extract_curve(run_name)
        ax.plot(wall_mins, clip, color="#D95F59", linewidth=2.0, label="clip_fraction")
        ax.plot(wall_mins, behavior, color="#2A9D8F", linewidth=2.0, label="behavior reward")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("Wall time (min)")
        ax.set_ylabel("smoothed training metric")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="lower right", frameon=False)

    fig.suptitle(
        "Recommendation post-training is an alignment problem, not just a loss-minimization problem",
        x=0.02,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.015,
        "Sources: EXP-020, EXP-028, EXP-029, EXP-036, EXP-037, EXP-038B, EXP-039B full eval and train logs.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(OUT_DIR / "post_training_alignment.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_post_training_alignment()


if __name__ == "__main__":
    main()
