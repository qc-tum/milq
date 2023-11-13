"""Generates the benchmark data."""
from time import perf_counter
from typing import Collection
import json

from mqt.bench import get_benchmark
from qiskit import QuantumCircuit
import numpy as np

from .generate_baseline_schedules import generate_baseline_schedule

from .generate_milp_schedules import (
    generate_extended_schedule,
    generate_simple_schedule,
    set_up_base_lp,
)
from .types import Result

np.random.seed(42)

# Define the maximum circuit size
# MAX_SIZE = 25
NUM_BATCHES = 10
CIRCUITS_PER_BATCH = 5
SETTINGS = [
    {"A": 5, "B": 5},
    {"A": 5, "B": 6, "C": 20},
]
T_MAX = 2**6


def _generate_batch(max_size: int, circuits_per_batch: int) -> list[QuantumCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(2, max_size + 1)
        circuit = get_benchmark(benchmark_name="random", level=0, circuit_size=size)
        batch.append(circuit)

    return batch


def run_experiments(
    circuits_per_batch: int, settings: list[dict[str, int]], t_max: int
) -> list[dict[str, Collection[Collection[str]]]]:
    """Runs the benchmakr experiments."""
    results = []
    for setting in settings:
        max_size = max(setting.values())
        benchmarks = [
            _generate_batch(max_size, circuits_per_batch) for _ in range(NUM_BATCHES)
        ]
        benchmark_results = []
        for benchmark in benchmarks:
            lp_instance = set_up_base_lp(
                benchmark, setting, big_m=1000, timesteps=list(range(t_max))
            )
            p_times = _get_processing_times(benchmark, setting)
            s_times = _get_setup_times(benchmark, setting, default_value=2**5)
            result = {}
            t_0 = perf_counter()
            makespan, jobs = generate_baseline_schedule(
                benchmark, setting, p_times, s_times
            )
            t_1 = perf_counter()
            result["baseline"] = Result(makespan, jobs, t_1 - t_0)

            makespan, jobs = generate_simple_schedule(lp_instance, p_times, s_times)
            t_2 = perf_counter()
            result["simple"] = Result(makespan, jobs, t_2 - t_1)
            makespan, jobs = generate_extended_schedule(lp_instance, p_times, s_times)
            t_3 = perf_counter()
            result["extended"] = Result(makespan, jobs, t_3 - t_2)
            benchmark_results.append(result)

            results.append({"setting": setting, "benchmarks": benchmark_results})
    return results


def _get_processing_times(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
) -> list[list[float]]:
    return [
        [np.random.random() * 2 + job.num_qubits / 10 for _ in accelerators]
        for job in base_jobs
    ]


def _get_setup_times(
    base_jobs: list[QuantumCircuit], accelerators: dict[str, int], default_value: float
) -> list[list[list[float]]]:
    return [
        [
            [
                default_value
                if job_i == job_j
                else np.random.random() * 5 + (job_i.num_qubits + job_j.num_qubits) / 20
                for _ in accelerators
            ]
            for job_i in base_jobs
        ]
        for job_j in base_jobs
    ]


if __name__ == "__main__":
    experiment_results = run_experiments(CIRCUITS_PER_BATCH, SETTINGS, T_MAX)
    with open("benchmark_results.json", "w+", encoding="uft-8") as f:
        json.dump(experiment_results, f)

    # TODO: Visualize results
