"""Generates the benchmark data."""
from copy import deepcopy

from mqt.bench import get_benchmark
from qiskit import QuantumCircuit
import numpy as np

from utils.helpers import Timer

from .generate_baseline_schedules import generate_baseline_schedule
from .generate_milp_schedules import (
    generate_extended_schedule,
    generate_simple_schedule,
    set_up_base_lp,
)
from .types import Benchmark, PTimes, Result, STimes


def _generate_batch(max_qubits: int, circuits_per_batch: int) -> list[QuantumCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(2, max_qubits + 1)
        circuit = get_benchmark(benchmark_name="random", level=0, circuit_size=size)
        batch.append(circuit)

    return batch


def run_experiments(
    circuits_per_batch: int,
    settings: list[dict[str, int]],
    t_max: int,
    num_batches: int,
    get_integers: bool = False,
) -> Benchmark:
    """Generates the benchmarks and executes scheduling.

    A setting represents a set of machines with their respective capacities
    e.g. {"BELEM": 5, "QUITO": 5}.
    Generates a total of num_batches * circuits_per_batch circuits for each setting.
    The maximum circuit size is determined by the maximum capacity in the setting.
    In each setting the three algorithms 'simple', 'extended', and 'baseline' are executed.
    The results for timings and makespan  are stored in a list of dictionaries.


    Args:
        circuits_per_batch (int): The number of circuits per batch.
        settings (list[dict[str, int]]): The list of settings to run.
        t_max (int): Metaparameter for timesteps.
        num_batches (int): The number of batches to generate.
        get_integers (bool, optional): Switch to use integer inputs. Defaults to False.

    Returns:
        list[dict[str, dict[str, int] | list[dict[str, list[list[float]] |
            list[list[list[float]]] | dict[str, Result]]]]]
            The results of the experiments.
    """
    results: Benchmark = []
    for setting in settings:
        max_size = max(setting.values())
        benchmarks = [
            _generate_batch(max_size, circuits_per_batch) for _ in range(num_batches)
        ]
        benchmark_results: list[dict[str, PTimes | STimes | dict[str, Result]]] = []
        for benchmark in benchmarks:
            lp_instance = set_up_base_lp(
                benchmark, setting, big_m=1000, timesteps=list(range(t_max))
            )
            p_times = _get_processing_times(benchmark, setting, get_integers)
            s_times = _get_setup_times(
                benchmark, setting, default_value=2**5, get_integers=get_integers
            )
            result: dict[str, Result] = {}

            # Run the baseline model
            with Timer() as t0:
                makespan, jobs = generate_baseline_schedule(
                    benchmark, setting, p_times, s_times
                )
            result["baseline"] = Result(makespan, jobs, t0.elapsed)

            # Run the simple model
            lp_instance_copy = deepcopy(lp_instance)
            with Timer() as t1:
                makespan, jobs = generate_simple_schedule(
                    lp_instance_copy, p_times, s_times
                )
            result["simple"] = Result(makespan, jobs, t1.elapsed)

            # Run the extended model
            with Timer() as t2:
                makespan, jobs = generate_extended_schedule(
                    lp_instance, p_times, s_times
                )
            result["extended"] = Result(makespan, jobs, t2.elapsed)

            # Store results
            benchmark_results.append(
                {"results": result, "s_times": s_times, "p_times": p_times}
            )

        results.append({"setting": setting, "benchmarks": benchmark_results})
    return results


def _get_processing_times(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    get_integers: bool = False,
) -> PTimes:
    return [
        [np.random.randint(0, 3) + job.num_qubits // 2 for _ in accelerators]
        if get_integers
        else [np.random.random() * 10 + job.num_qubits / 5 for _ in accelerators]
        for job in base_jobs
    ]


def _get_setup_times(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    default_value: float,
    get_integers: bool = False,
) -> STimes:
    return [
        [
            [
                default_value
                if id_i in [id_j, 0]
                else _calc_setup_times(job_i, job_j, get_integers)
                for _ in accelerators
            ]
            for id_i, job_i in enumerate([None] + base_jobs)
        ]
        for id_j, job_j in enumerate([None] + base_jobs)
    ]


def _calc_setup_times(
    job_i: QuantumCircuit,
    job_j: QuantumCircuit,
    get_integers: bool = False,
) -> float:
    if get_integers:
        return np.random.randint(0, 2) + (job_i.num_qubits + job_j.num_qubits) // 8
    return np.random.random() * 10 + (job_i.num_qubits + job_j.num_qubits) / 10
