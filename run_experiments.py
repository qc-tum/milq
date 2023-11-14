"""Generates the benchmark data."""
from dataclasses import is_dataclass, asdict
from typing import Any
import json

import numpy as np

from data.benchmark import run_experiments


class DataclassJSONEncoder(json.JSONEncoder):
    """Helper to serialize dataclasses."""

    def default(self, o) -> dict[str, Any] | Any:
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


np.random.seed(42)

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
    with open("benchmark_results.json", "w+", encoding="utf-8") as f:
        json.dump(experiment_results, f, cls=DataclassJSONEncoder)

    # TODO: Visualize results
