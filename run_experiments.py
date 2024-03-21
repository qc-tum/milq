"""Generates the benchmark data."""

import logging

from dataclasses import is_dataclass, asdict
from typing import Any
import json

import numpy as np

from src.provider import Accelerator, IBMQBackend

from data.benchmark import (
    run_experiments,
    analyze_benchmarks,
    run_heuristic_experiments,
)


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
T_MAX = 200
ACC_SETTINGS = [
    [
        Accelerator(IBMQBackend.BELEM, shot_time=5, reconfiguration_time=12),
        Accelerator(IBMQBackend.NAIROBI, shot_time=7, reconfiguration_time=12),
    ],
    # [
    #     Accelerator(IBMQBackend.BELEM, shot_time=5, reconfiguration_time=12),
    #     Accelerator(IBMQBackend.NAIROBI, shot_time=7, reconfiguration_time=12),
    #     Accelerator(IBMQBackend.QUITO, shot_time=2, reconfiguration_time=16),
    # ],
]


def run_default() -> None:
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


def run_heuristic() -> None:
    experiment_results = run_heuristic_experiments(
        CIRCUITS_PER_BATCH, ACC_SETTINGS, T_MAX, NUM_BATCHES
    )
    if len(experiment_results) < 1:
        return
    with open(
        "./data/results/benchmark_results_heuristic.json", "w+", encoding="utf-8"
    ) as f:
        json.dump(experiment_results, f, cls=DataclassJSONEncoder)


if __name__ == "__main__":
    for setting in ACC_SETTINGS:
        for acc in setting:
            acc.queue.extend([0] * np.random.randint(0, 10))
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("circuit_knitting").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    run_heuristic()
