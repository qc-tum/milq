"""Generates the benchmark data."""
from dataclasses import is_dataclass, asdict
from typing import Any
import json

# import numpy as np

from data.benchmark import run_experiments, analyze_benchmarks


class DataclassJSONEncoder(json.JSONEncoder):
    """Helper to serialize dataclasses."""

    def default(self, o) -> dict[str, Any] | Any:
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


# np.random.seed(42)

# Define the maximum circuit size
# MAX_SIZE = 25
NUM_BATCHES = 1
CIRCUITS_PER_BATCH = 5
SETTINGS = [
    {"A": 5, "B": 5},
    {"A": 5, "B": 6, "C": 20},
]
T_MAX = 2**6
if __name__ == "__main__":
    experiment_results = run_experiments(
        CIRCUITS_PER_BATCH, SETTINGS, T_MAX, NUM_BATCHES
    )
    with open(
        "./data/results/benchmark_results_default.json", "w+", encoding="utf-8"
    ) as f:
        json.dump(experiment_results, f, cls=DataclassJSONEncoder)

    experiment_results = run_experiments(
        CIRCUITS_PER_BATCH, SETTINGS, T_MAX, NUM_BATCHES, get_integers=True
    )
    with open(
        "./data/results/benchmark_results_integer.json", "w+", encoding="utf-8"
    ) as f:
        json.dump(experiment_results, f, cls=DataclassJSONEncoder)

    numbers = analyze_benchmarks("./data/results/benchmark_results_default.json")
    for setting, result in numbers.items():
        print(f"Setting: {setting}")
        print(result)

    numbers = analyze_benchmarks("./data/results/benchmark_results_integer.json")
    for setting, result in numbers.items():
        print(f"Setting: {setting}")
        print(result)
