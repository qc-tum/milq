"""Generates the benchmark data."""

from uuid import uuid4
import logging

from mqt.bench import get_benchmark
import numpy as np

from src.common import UserCircuit
from src.provider import Accelerator
from src.scheduling import (
    Benchmark,
    Result,
)
from src.scheduling.heuristics import (
    generate_heuristic_info_schedule as heuristic_schedule,
)
from src.scheduling.learning import generate_rl_schedule
from src.utils.helpers import Timer

from .evaluate_baseline import evaluate_baseline


def _generate_batch(
    max_qubits: int, circuits_per_batch: int, accelerators: list[Accelerator]
) -> list[UserCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(2, max_qubits + 1)
        circuit = get_benchmark(benchmark_name="ghz", level=1, circuit_size=size)
        circuit.remove_final_measurements(inplace=True)
        user_circuit = UserCircuit(
            circuit,
            1024,
            np.random.randint(1, 10),
            str(accelerators[np.random.randint(len(accelerators))].uuid),
            np.random.randint(1, 3),
            uuid4(),
        )
        batch.append(user_circuit)

    return batch


def run_heuristic_experiments(
    circuits_per_batch: int,
    settings: list[list[Accelerator]],
    num_batches: int,
) -> Benchmark:
    """Generates the benchmarks and executes scheduling."""
    results: Benchmark = []
    for setting in settings:
        logging.info("New Setting started...")
        max_size = sum(s.qubits for s in setting)
        benchmarks = [
            _generate_batch(max_size, circuits_per_batch, setting)
            for _ in range(num_batches)
        ]
        benchmark_results: list[dict[str, Result]] = []
        for benchmark in benchmarks:
            result: dict[str, Result] = {}

            logging.info("Running benchmark for setting.")
            # Run the baseline model
            with Timer() as t0:
                baseline, jobs = evaluate_baseline(benchmark, setting)
            result["baseline"] = Result(
                baseline[0], jobs, t0.elapsed, baseline[1], baseline[2]
            )
            logging.info("Baseline model done: Makespan: %d.", baseline[0])
            # Run the reinforcement learning  model

            # with Timer() as t1:
            #     makespan, jobs = generate_rl_schedule(benchmark, setting)
            # result["RL"] = Result(makespan, jobs, t1.elapsed)
            # logging.info("RL model done: Makespan: %d.", makespan)
            # Run the heurstic model
            with Timer() as t2:
                heuristic, jobs = heuristic_schedule(
                    benchmark,
                    setting,
                    num_iterations=128,
                    partition_size=4,
                    num_cores=16,
                )
            result["heuristic"] = Result(
                heuristic[0], jobs, t2.elapsed, heuristic[1], heuristic[2]
            )
            logging.info("Heuristic model done: Makespan: %d.", heuristic[0])
            # Store results
            benchmark_results.append(result)
        if len(benchmark_results) > 0:
            results.append(
                {
                    "setting": {str(acc.uuid): acc.qubits for acc in setting},
                    "benchmarks": benchmark_results,
                }
            )
    return results
