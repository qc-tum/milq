"""Data processing for the benchmark results."""
from dataclasses import dataclass, asdict

import json
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
            f"Simple Makespan: {self.simple_makespan:.2%}\n"
            + f"Simple Time: {self.simple_time:.2e}\n"
            + f"Extended Makespan: {self.extended_makespan:.2%}\n"
            + f"Extended Time: {self.extended_time:.2e}"
        )


def analyze_benchmarks(in_file: str) -> dict[str, ImprovementResult]:
    """Visualizes the benchmark results and calculates the average improvements.

    Calculates Makespan and Timing improvements for the simple and extended algorithms.
    Makespan improvements are calculated as (baseline - algorithm) / baseline * 100
    Timing improvements are calculated as (algorithm / baseline)

    Args:
        in_file (str): The file containing the benchmark results.

    Returns:
        dict[str, ImprovementResult]: The calculated improvements for each setting.
    """
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
            results = benchmark["results"]
            makespans.append(
                MakespanResult(
                    baseline=results["baseline"]["makespan"],
                    simple=results["simple"]["makespan"],
                    extended=results["extended"]["makespan"],
                )
            )
            times.append(
                TimingResult(
                    baseline=results["baseline"]["time"],
                    simple=results["simple"]["time"],
                    extended=results["extended"]["time"],
                )
            )

        hide_x_axis = idx < len(data) - 1
        _plot_benchmark_result(
            makespans, title, (len(data), 1, idx + 1), hide_x_axis=hide_x_axis
        )
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
    hide_x_axis: bool = False,
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
        edgecolor="white",
        color="#154060",
    )
    plt.bar(
        x_pos_2,
        data["simple"],
        width=bar_width,
        label="simple",
        edgecolor="white",
        color="#527a9c",
    )
    plt.bar(
        x_pos_3,
        data["extended"],
        width=bar_width,
        label="extended",
        edgecolor="white",
        color="#98c6ea",
    )

    if not hide_x_axis:
        plt.xlabel("Trial", fontweight="bold")
    plt.xticks(x_pos_2, [str(x) for x in x_pos_1])
    plt.ylabel("Total Makespan", fontweight="bold")
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

        # TODO still not sure what the best metric for this is
        simple_times.append(time.simple / baseline_time)
        extended_times.append(time.extended / baseline_time)

    return ImprovementResult(
        float(np.average(simple_makespans)),
        float(np.average(simple_times)),
        float(np.average(extended_makespans)),
        float(np.average(extended_times)),
    )
