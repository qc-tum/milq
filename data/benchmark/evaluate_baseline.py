"""Evalute the baseline scheduler."""

from collections import defaultdict
from uuid import UUID
import logging

from qiskit import QuantumCircuit

from src.common import jobs_from_experiment, UserCircuit
from src.provider import Accelerator
from src.scheduling import (
    InfoProblem,
    JobResultInfo,
    PTimes,
    SchedulerType,
    STimes,
    generate_schedule,
)
from src.scheduling.common import MakespanInfo, makespan_function
from src.tools import cut_circuit


def _cut_circuits(
    circuits: list[UserCircuit], accelerators: list[Accelerator]
) -> dict[UUID, list[QuantumCircuit]]:
    """Cuts the circuits into smaller circuits."""
    partitions = _generate_partitions(
        [circuit.num_qubits for circuit in circuits], accelerators
    )
    logging.debug(
        "Partitions: generated: %s",
        " ".join(str(partition) for partition in partitions),
    )
    jobs = defaultdict(list)
    logging.debug("Cutting circuits...")
    for idx, circuit in enumerate(circuits):
        logging.debug("Cutting circuit %d", idx)
        if len(partitions[idx]) > 1:
            experiments, _ = cut_circuit(circuit.circuit, partitions[idx])
            jobs[circuit.name] = [
                job.circuit
                for experiment in experiments
                for job in jobs_from_experiment(experiment)
            ]
        else:
            # assumption for now dont cut to any to smaller
            jobs[circuit.name].append(circuit.circuit)
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


def evaluate_baseline(
    benchmark: list[UserCircuit], setting: list[Accelerator]
) -> tuple[tuple[float, float, float], list[JobResultInfo]]:
    """Evaluates the baseline scheduler.

    Args:
        benchmark (list[UserCircuit]): The batch of circuits to schedule.
        setting (list[Accelerator]): The list of accelerators to schedule on.

    Returns:
        tuple[tuple[float, float, float], list[JobResultInfo]]:
            The makespan, metric, noise, and the list of job results.
    """
    circuits = _cut_circuits(benchmark, setting)
    problem_circuits = [
        circuit for circuit_list in circuits.values() for circuit in circuit_list
    ]
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
        timesteps=1000,
        process_times=p_times,
        setup_times=s_times,
    )
    result = generate_schedule(problem, SchedulerType.BASELINE)
    assert isinstance(result, tuple)
    makespan, jobs, _ = result
    machines = _machines_from_jobs(jobs, setting, problem_circuits, benchmark, circuits)

    metrics, noises = [], []
    for machine, machine_jobs in machines.items():
        metrics.append(
            makespan_function([info for _, info in machine_jobs], str(machine.uuid))
        )
        noises.append(sum(machine.compute_noise(job) for job, _ in machine_jobs))
    return (makespan, max(metrics), sum(noises)), jobs


def _machines_from_jobs(
    jobs: list[JobResultInfo],
    setting: list[Accelerator],
    problem_circuits: list[QuantumCircuit],
    benchmark: list[UserCircuit],
    circuits: dict[UUID, list[QuantumCircuit]],
) -> dict[Accelerator, list[tuple[QuantumCircuit, MakespanInfo]]]:
    machines: dict[Accelerator, list[tuple[QuantumCircuit, MakespanInfo]]] = {
        acc: [] for acc in setting
    }
    for job in jobs:
        job_circuit = next(
            (circuit for circuit in problem_circuits if circuit.name == job.name)
        )
        user_circuit_uuid = next(
            uuid for uuid, circuits in circuits.items() if job_circuit in circuits
        )
        user_circuit = next(
            circuit for circuit in benchmark if circuit.name == user_circuit_uuid
        )
        info = MakespanInfo(
            job=None,
            start_time=job.start_time,
            completion_time=job.completion_time,
            capacity=job_circuit.num_qubits,
            preselection=user_circuit.machine_preference,
            priority=user_circuit.n_shots,
            strictness=user_circuit.strictness,
            n_shots=user_circuit.n_shots,
        )

        machine = next(acc for acc in setting if str(acc.uuid) == job.machine)
        machines[machine].append((job_circuit, info))
    return machines
