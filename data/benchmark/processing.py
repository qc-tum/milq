from collections import defaultdict
from dataclasses import dataclass, asdict

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class MakespanResult:
    """Holds the makespan values for the baseline, simple, and extended algorithms."""

    baseline: float
    simple: float
    extended: float


@dataclass
class TimingResult:
    """Holds the timing values for the baseline, simple, and extended algorithms."""

    baseline: float
    simple: float
    extended: float


@dataclass
class ImprovementResult:
    """Holds the improvement values for the simple and extended algorithms."""

    simple_makespan: float
    simple_time: float
    extended_makespan: float
    extended_time: float

    def __repr__(self) -> str:
        return (
            f"Simple Makespan: {self.simple_makespan:.2f}%\n"
            + f"Simple Time: {self.simple_time:.2f}\n"
            + f"Extended Makespan: {self.extended_makespan:.2f}%\n"
            + f"Extended Time: {self.extended_time:.2f}"
        )


def process_benchmarks(in_file: str) -> dict[str, ImprovementResult]:
    # Load the JSON file
    """Visualizes the benchmark results and calculates the average improvement"""
    with open(in_file, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    numbers = {}
    for idx, setting in enumerate(data):
        title = str(setting["setting"])
        benchmarks = setting["benchmarks"]
        makespans, times = [], []
        # Loop through each benchmark
        for benchmark in benchmarks:
            # Extract the makespan values
            makespans.append(
                MakespanResult(
                    baseline=benchmark["baseline"]["makespan"],
                    simple=benchmark["simple"]["makespan"],
                    extended=benchmark["extended"]["makespan"],
                )
            )
            times.append(
                TimingResult(
                    baseline=benchmark["baseline"]["time"],
                    simple=benchmark["simple"]["time"],
                    extended=benchmark["extended"]["time"],
                )
            )

        _plot_benchmark_result(makespans, title, (1, len(data), idx + 1))
        numbers[title] = _caclulate_improvements(makespans, times)
        # Display the resulting plot
    plt.tight_layout()
    plt.savefig(in_file.replace(".json", ".pdf"))
    return numbers


def _plot_benchmark_result(
    makespans: list[MakespanResult],
    title: str,
    subplot: tuple[int, int, int],
    bar_width=0.25,
) -> None:
    """Plot the makespan values for the baseline, simple, and extended algorithms."""

    data = pd.DataFrame(asdict(result) for result in makespans)
    x_pos_1 = np.arange(len(data["baseline"]))
    x_pos_2 = [x + bar_width for x in x_pos_1]
    x_pos_3 = [x + bar_width for x in x_pos_2]
    plt.subplot(*subplot)

    plt.bar(
        x_pos_1,
        data["baseline"],
        width=bar_width,
        label="baseline",
        edgecolor="grey",
        color="#A2AD00",
    )
    plt.bar(
        x_pos_2,
        data["simple"],
        width=bar_width,
        label="simple",
        edgecolor="grey",
        color="#E37222",
    )
    plt.bar(
        x_pos_3,
        data["extended"],
        width=bar_width,
        label="extended",
        edgecolor="grey",
        color="#0065BD",
    )

    plt.xlabel("Trial", fontweight="bold")
    plt.ylabel("Total Makespan", fontweight="bold")
    plt.xticks(x_pos_2, [str(x) for x in x_pos_1])
    plt.title(title, fontweight="bold")
    plt.legend()


def _caclulate_improvements(
    makespans: list[MakespanResult], times: list[TimingResult]
) -> ImprovementResult:
    """Calculates the average improvement for the simple and extended algorithms."""
    simple_makespans, extended_makespans = [], []
    simple_times, extended_times = [], []
    for makespan, time in zip(makespans, times):
        baseline = makespan.baseline
        baseline_time = time.baseline
        simple_makespans.append((baseline - makespan.simple) / baseline * 100)
        extended_makespans.append((baseline - makespan.extended) / baseline * 100)

        simple_times.append(
            (baseline - makespan.simple) / (time.simple - baseline_time)
        )
        extended_times.append(
            (baseline - makespan.extended) / (time.extended - baseline_time)
        )

    return ImprovementResult(
        float(np.average(simple_makespans)),
        float(np.average(simple_times)),
        float(np.average(extended_makespans)),
        float(np.average(extended_times)),
    )
