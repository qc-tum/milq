"""Generates the benchmark data."""

import logging

from mqt.bench import get_benchmark
from qiskit import QuantumCircuit
import numpy as np

from src.common import jobs_from_experiment
from src.provider import Accelerator
from src.scheduling import (
    Benchmark,
    InfoProblem,
    PTimes,
    Result,
    SchedulerType,
    STimes,
    generate_schedule,
)
from src.scheduling.heuristics import (
    generate_heuristic_info_schedule as heuristic_schedule,
)

from src.tools import cut_circuit
from src.utils.helpers import Timer


def _generate_batch(max_qubits: int, circuits_per_batch: int) -> list[QuantumCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(2, max_qubits + 1)
        circuit = get_benchmark(benchmark_name="ghz", level=1, circuit_size=size)
        circuit.remove_final_measurements(inplace=True)
        batch.append(circuit)

    return batch


def run_heuristic_experiments(
    circuits_per_batch: int,
    settings: list[list[Accelerator]],
    t_max: int,
    num_batches: int,
) -> Benchmark:
    """Generates the benchmarks and executes scheduling."""
    results: Benchmark = []
    for setting in settings:
        logging.info("New Setting started...")
        max_size = sum(s.qubits for s in setting)
        benchmarks = [
            _generate_batch(max_size, circuits_per_batch) for _ in range(num_batches)
        ]
        benchmark_results: list[Result] = []
        for benchmark in benchmarks:
            result: dict[str, Result] = {}

            logging.info("Running benchmark for setting.")
            # Run the baseline model
            with Timer() as t0:
                problem_circuits = _cut_circuits(benchmark, setting)
                logging.info("Setting up times...")

                p_times = _get_benchmark_processing_times(problem_circuits, setting)
                s_times = _get_benchmark_setup_times(
                    problem_circuits,
                    setting,
                    default_value=2**5,
                )
                logging.info("Setting up problems...")
                problem = InfoProblem(
                    base_jobs=problem_circuits,
                    accelerators={str(acc.uuid): acc.qubits for acc in setting},
                    big_m=1000,
                    timesteps=t_max,
                    process_times=p_times,
                    setup_times=s_times,
                )
                makespan, jobs, _ = generate_schedule(problem, SchedulerType.BASELINE)
            result["baseline"] = Result(makespan, jobs, t0.elapsed)
            logging.info("Baseline model done: Makespan: %d.", makespan)
            # Run the simple model
            # if makespan > t_max:
            #     continue

            # with Timer() as t1:
            #     makespan, jobs, _ = generate_schedule(problem, SchedulerType.SIMPLE)
            # result["simple"] = Result(makespan, jobs, t1.elapsed)
            # logging.info("Simple model done: Makespan: %d.", makespan)
            # Run the heurstic model
            with Timer() as t2:
                # TODO convert ScheduledJob to JobResultInfo
                makespan, jobs = heuristic_schedule(
                    benchmark,
                    setting,
                    num_iterations=128,
                    partition_size=4,
                    num_cores=16,
                )
            result["heuristic"] = Result(makespan, jobs, t2.elapsed)
            logging.info("Heuristic model done: Makespan: %d.", makespan)
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


def _cut_circuits(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator]
) -> list[QuantumCircuit]:
    """Cuts the circuits into smaller circuits."""
    partitions = _generate_partitions(
        [circuit.num_qubits for circuit in circuits], accelerators
    )
    logging.debug(
        "Partitions: generated: %s",
        " ".join(str(partition) for partition in partitions),
    )
    jobs = []
    logging.debug("Cutting circuits...")
    for idx, circuit in enumerate(circuits):
        logging.debug("Cutting circuit %d", idx)
        if len(partitions[idx]) > 1:
            experiments, _ = cut_circuit(circuit, partitions[idx])
            jobs += [
                job.circuit
                for experiment in experiments
                for job in jobs_from_experiment(experiment)
            ]
        else:
            # assumption for now dont cut to any to smaller
            jobs.append(circuit)
    return jobs


def _generate_partitions(
    circuit_sizes: list[int], accelerators: list[Accelerator]
) -> list[list[int]]:
    partitions = []
    qpu_sizes = [acc.qubits for acc in accelerators]
    num_qubits: int = sum(qpu_sizes)
    for circuit_size in circuit_sizes:
        if circuit_size > num_qubits:
            partition = qpu_sizes
            remaining_size = circuit_size - num_qubits
            while remaining_size > num_qubits:
                partition += qpu_sizes
                remaining_size -= num_qubits
            if remaining_size == 1:
                partition[-1] = partition[-1] - 1
                partition.append(2)
            else:
                partition += _partition_big_to_small(remaining_size, accelerators)
            partitions.append(partition)
        elif circuit_size > max(qpu_sizes):
            partition = _partition_big_to_small(circuit_size, accelerators)
            partitions.append(partition)
        else:
            partitions.append([circuit_size])
    return partitions


def _partition_big_to_small(size: int, accelerators: list[Accelerator]) -> list[int]:
    partition = []
    for qpu in sorted(accelerators, key=lambda a: a.qubits, reverse=True):
        take_qubits = min(size, qpu.qubits)
        if size - take_qubits == 1:
            # We can't have a partition of size 1
            # So in this case we take one qubit less to leave a partition of two
            take_qubits -= 1
        partition.append(take_qubits)
        size -= take_qubits
        if size == 0:
            break
    else:
        raise ValueError(
            "Circuit is too big to fit onto the devices,"
            + f" {size} qubits left after partitioning."
        )
    return partition


def _get_benchmark_processing_times(
    base_jobs: list[QuantumCircuit],
    accelerators: list[Accelerator],
) -> PTimes:
    return [
        [accelerator.compute_processing_time(job) for accelerator in accelerators]
        for job in base_jobs
    ]


def _get_benchmark_setup_times(
    base_jobs: list[QuantumCircuit],
    accelerators: list[Accelerator],
    default_value: float,
) -> STimes:
    return [
        [
            [
                (
                    default_value
                    if id_i in [id_j, 0]
                    else (
                        0
                        if job_j is None
                        else accelerator.compute_setup_time(job_i, job_j)
                    )
                )
                for accelerator in accelerators
            ]
            for id_i, job_i in enumerate([None] + base_jobs)
        ]
        for id_j, job_j in enumerate([None] + base_jobs)
    ]
