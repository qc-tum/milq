"""Data processing for the benchmark results."""

from dataclasses import dataclass, asdict

import json
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class MakespanResult:
    """Holds the makespan values for the baseline, heuristic, and rl algorithms"""

    baseline: float
    heuristic: float
    rl: float


@dataclass
class ScoreResult:
    """Holds the M score values for the baseline, heuristic, and rl algorithms."""

    baseline: float
    heuristic: float
    rl: float


@dataclass
class NoiseResult:
    """Holds the noise values for the baseline, heuristic, and rl algorithms"""

    baseline: float
    heuristic: float
    rl: float


def analyze_heuristic_benchmarks(in_file: str) -> None:
    """Visualizes the benchmark results and calculates the average improvements.
    Args:
        in_file (str): The file containing the benchmark results.

    """
    with open(in_file, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    for setting in data:
        title = str(setting["setting"])
        benchmarks = setting["benchmarks"]
        makespans, scores, noises = [], [], []
        # Loop through each benchmark
        for benchmark in benchmarks:
            makespans.append(
                MakespanResult(
                    baseline=benchmark["baseline"]["makespan"],
                    heuristic=benchmark["heuristic"]["makespan"],
                    rl=benchmark["rl"]["makespan"],
                )
            )
            scores.append(
                ScoreResult(
                    baseline=benchmark["baseline"]["metric"],
                    heuristic=benchmark["heuristic"]["metric"],
                    rl=benchmark["rl"]["metric"],
                )
            )
            noises.append(
                NoiseResult(
                    baseline=benchmark["baseline"]["noise"],
                    heuristic=benchmark["heuristic"]["noise"],
                    rl=benchmark["rl"]["noise"],
                )
            )

        _plot_boxplots(title, makespans, scores, noises)
    # Display the resulting plot
    plt.tight_layout()
    plt.savefig(in_file.replace(".json", ".pdf"))


def _plot_boxplots(
    title: str,
    makespans: list[MakespanResult],
    scores: list[ScoreResult],
    noises: list[NoiseResult],
) -> None:
    props = {
        "boxprops": {"facecolor": "#98c6ea", "edgecolor": "#154060"},
        "whiskerprops": {"color": "#154060"},
        "medianprops": {"color": "#e37222"},
        "capprops": {"color": "#154060"},
        "patch_artist": True,
        "widths": 0.25,
        "positions": [0.5, 1, 1.5],
    }
    makespan_data = pd.DataFrame([asdict(result) for result in makespans])
    score_data = pd.DataFrame([asdict(result) for result in scores])
    noise_data = pd.DataFrame([asdict(result) for result in noises])

    # Plotting boxplots
    fig, axes = plt.subplots(3, 1, figsize=(5, 7))
    fig.suptitle(title)

    makespan_data.plot(
        kind="box",
        ax=axes[0],
        **props,
    )
    axes[0].set_title("Makespan")
    axes[0].set_ylabel(r"Time [$\mu s$]")

    score_data.plot(kind="box", ax=axes[1], **props)
    axes[1].set_title("Score")
    axes[1].set_ylabel(r"$\mathbf{M}$")

    noise_data.plot(kind="box", ax=axes[2], **props)
    axes[2].set_title("Noise")
    axes[2].set_ylabel(r"$\mathbf{F}$")

    for ax in axes:
        ax.set_xticklabels(["Baseline", "Heuristic", "RL"])
