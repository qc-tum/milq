from dataclasses import dataclass, asdict

import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Holds the makespan values for the baseline, simple, and extended algorithms."""

    baseline: float
    simple: float
    extended: float


def visualize_benchmarks(in_file: str) -> None:
    # Load the JSON file
    """Visualizes the benchmark results."""
    with open(in_file, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    for idx, setting in enumerate(data):
        title = str(setting["setting"])
        benchmarks = setting["benchmarks"]
        results = []
        # Loop through each benchmark
        for benchmark in benchmarks:
            # Extract the makespan values
            results.append(
                BenchmarkResult(
                    baseline=benchmark["baseline"]["makespan"],
                    simple=benchmark["simple"]["makespan"],
                    extended=benchmark["extended"]["makespan"],
                )
            )

        _plot_benchmark_result(results, title, (1, len(data), idx + 1))

        # Display the resulting plot
    plt.tight_layout()
    plt.savefig(in_file.replace(".json", ".pdf"))


def _plot_benchmark_result(
    results: list[BenchmarkResult],
    title: str,
    subplot: tuple[int, int, int],
    bar_width=0.25,
) -> None:
    """Plot the makespan values for the baseline, simple, and extended algorithms."""

    data = pd.DataFrame(asdict(result) for result in results)
    x_pos_1 = np.arange(len(data["baseline"]))
    x_pos_2 = [x + bar_width for x in x_pos_1]
    x_pos_3 = [x + bar_width for x in x_pos_2]
    cmap = matplotlib.colormaps.get_cmap("tab10")
    color_mapping = {
        m: cmap(i / (len(data.columns) - 1)) for i, m in enumerate(data.columns)
    }
    plt.subplot(*subplot)

    plt.bar(
        x_pos_1,
        data["baseline"],
        width=bar_width,
        label="baseline",
        edgecolor="grey",
        color=color_mapping["baseline"],
    )
    plt.bar(
        x_pos_2,
        data["simple"],
        width=bar_width,
        label="simple",
        edgecolor="grey",
        color=color_mapping["simple"],
    )
    plt.bar(
        x_pos_3,
        data["extended"],
        width=bar_width,
        label="extended",
        edgecolor="grey",
        color=color_mapping["extended"],
    )

    plt.xlabel("Trial", fontweight="bold")
    plt.ylabel("Total Makespan", fontweight="bold")
    plt.xticks(x_pos_2, [str(x) for x in x_pos_1])
    plt.title(title, fontweight="bold")
    plt.legend()
