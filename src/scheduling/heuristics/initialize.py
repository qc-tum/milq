"""_summary_"""
from qiskit import QuantumCircuit

from src.common import CircuitJob, jobs_from_experiment, job_from_circuit, ScheduledJob
from src.provider import Accelerator

from src.tools import cut_circuit

from .types import Schedule
from ..bin_schedule import generate_bin_executable_schedule as bin_schedule


def initialize_population(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator]
) -> list[Schedule]:
    """
    TODO: use bin packing with the following parameters:
    - greedy cutting
    - even cutting
    - "informed cuts"""

    scheduled_jobs = []
    for option in OPTIONS:
        partitions = option(circuits, accelerators)
        jobs = _convert_to_jobs(circuits, partitions)
        schedule = bin_schedule(jobs, accelerators)
        if schedule is not None:
            scheduled_jobs.append(schedule)
    return [_convert_to_schedule(schedule) for schedule in scheduled_jobs]


def _greedy_partitioning(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator]
) -> list[list[int]]:
    """taken from scheduler.py"""
    partitions = []
    qpu_sizes = [acc.qubits for acc in accelerators]
    total_qubits = sum(qpu_sizes)
    circuit_sizes = [
        circ.circuit.num_qubits for circ in circuits if circ.circuit is not None
    ]
    for circuit_size in sorted(circuit_sizes, reverse=True):
        if circuit_size > total_qubits:
            partition = qpu_sizes
            remaining_size = circuit_size - total_qubits
            while remaining_size > total_qubits:
                partition += qpu_sizes
                remaining_size -= total_qubits
            if remaining_size == 1:
                partition[-1] = partition[-1] - 1
                partition.append(2)
            else:
                partition += _partition_big_to_small(remaining_size, qpu_sizes)
            partitions.append(partition)
        elif circuit_size > max(qpu_sizes):
            partition = _partition_big_to_small(circuit_size, qpu_sizes)
            partitions.append(partition)
        else:
            partitions.append([circuit_size])

    return partitions


def _partition_big_to_small(size: int, qpu_sizes: list[int]) -> list[int]:
    partition = []
    for qpu_size in qpu_sizes:
        take_qubits = min(size, qpu_size)
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


def _even_partitioning(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator]
) -> list[Schedule]:
    return []


def _informed_partitioning(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator]
) -> list[Schedule]:
    return []


def _random_partitioning(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator]
) -> list[Schedule]:
    return []


def _convert_to_jobs(circuits, partitions) -> list[CircuitJob]:
    jobs = []
    for idx, circuit in enumerate(circuits):
        if len(partitions[idx]) > 1:
            experiments, _ = cut_circuit(circuit, partitions[idx])
            jobs += [
                job
                for experiment in experiments
                for job in jobs_from_experiment(experiment)
            ]
        else:
            # assumption for now dont cut to any to smaller
            circuit = job_from_circuit(circuit)
            jobs.append(circuit)
    return jobs


def _convert_to_schedule(jobs: list[ScheduledJob]) -> Schedule:
    return Schedule([], 0)


OPTIONS = [
    _greedy_partitioning,
    _even_partitioning,
    _informed_partitioning,
    _random_partitioning,
]
