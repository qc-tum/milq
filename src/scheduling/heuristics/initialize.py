"""_summary_"""

from collections import Counter

from qiskit import QuantumCircuit
import numpy as np

from src.common import CircuitJob, jobs_from_experiment, job_from_circuit, ScheduledJob
from src.provider import Accelerator

from src.tools import cut_circuit

from .types import Schedule, Machine, Bucket
from ..bin_schedule import _do_bin_pack


def initialize_population(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator]
) -> list[Schedule]:
    """
    TODO: use bin packing with the following parameters:
    - greedy cutting
    - even cutting
    - "informed cuts"""

    machines = []
    for option in OPTIONS:
        partitions = option(circuits, accelerators)
        jobs = _convert_to_jobs(circuits, partitions)
        machines.append(_bin_schedule(jobs, accelerators))
    return [Schedule(machine, 0) for machine in machines]


def _greedy_partitioning(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> list[list[int]]:
    """taken from scheduler.py"""
    partitions = []
    qpu_sizes = [acc.qubits for acc in accelerators]
    total_qubits = sum(qpu_sizes)
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in sorted(circuit_sizes, reverse=True):
        if circuit_size > total_qubits:
            partition = qpu_sizes.copy()
            remaining_size = circuit_size - total_qubits
            while remaining_size > total_qubits:
                partition += qpu_sizes
                remaining_size -= total_qubits
            if remaining_size == 1:
                if partition[-1] <= 2:
                    partition[-1] += 1
                else:
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
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> list[list[int]]:
    """Partition circuit in similar sized chunks"""
    partitions = []
    partition_size = sum(acc.qubits for acc in accelerators) // len(accelerators)
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in sorted(circuit_sizes, reverse=True):
        if circuit_size > partition_size:
            partition = [partition_size] * (circuit_size // partition_size)
            if circuit_size % partition_size != 0:
                partition.append(circuit_size % partition_size)
            if partition[-1] == 1:
                partition[-1] = 2
                partition[-2] -= 1
            partitions.append(partition)
        else:
            partitions.append([circuit_size])
    return partitions


def _informed_partitioning(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator], **kwargs
) -> list[list[int]]:
    """Finds cuts by recursively cutting the line with the least cnots"""
    partitions = []
    max_qpu_size = max(acc.qubits for acc in accelerators)

    for circuit in sorted(
        circuits,
        key=lambda circ: circ.num_qubits,
        reverse=True,
    ):
        counts = _count_cnots(circuit)
        cuts = sorted(_find_cuts(counts, 0, circuit.num_qubits, max_qpu_size))
        if len(cuts) == 0:
            partitions.append([circuit.num_qubits])
        else:
            partition = []
            current = -1
            for cut in cuts:
                partition.append(cut - current)
                current = cut
            partition.append(circuit.num_qubits - current - 1)
            partitions.append(partition)
    return partitions


def _count_cnots(circuit: QuantumCircuit) -> Counter[tuple[int, int]]:
    counter: Counter[tuple[int, int]] = Counter()
    for instruction in circuit.data:
        if instruction.operation.name == "cx":
            first_qubit = circuit.find_bit(instruction.qubits[0]).index
            second_qubit = circuit.find_bit(instruction.qubits[1]).index
            if abs(first_qubit - second_qubit) <= 1:
                counter[(first_qubit, second_qubit)] += 1
    return counter


def _find_cuts(
    counts: Counter[tuple[int, int]], start: int, end: int, max_qpu_size: int
) -> list[int]:
    if end - start <= max_qpu_size:
        return []
    possible_cuts = [_calulate_cut(counts, cut) for cut in range(start + 1, end - 2)]
    best_cut = min(possible_cuts, key=lambda cut: cut[1])[0]
    partitions = (
        [best_cut]
        + _find_cuts(counts, start, best_cut, max_qpu_size)
        + _find_cuts(counts, best_cut + 1, end, max_qpu_size)
    )

    return partitions


def _calulate_cut(counts: Counter[tuple[int, int]], cut: int) -> tuple[int, int]:
    left = sum(
        count for (first, second), count in counts.items() if first <= cut < second
    )
    right = sum(
        count for (first, second), count in counts.items() if second <= cut < first
    )
    return cut, left + right


def _choice_partitioning(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> list[list[int]]:
    partitions = []
    qpu_sizes = [acc.qubits for acc in accelerators]
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in sorted(circuit_sizes, reverse=True):
        partition = []
        remaining_size = circuit_size
        while remaining_size > 0:
            qpu = np.random.choice(qpu_sizes)
            take_qubits = min(remaining_size, qpu)
            if remaining_size - take_qubits == 1:
                # We can't have a partition of size 1
                # So in this case we take one qubit less to leave a partition of two
                take_qubits -= 1
            partition.append(take_qubits)
            remaining_size -= take_qubits
        partitions.append(partition)
    return partitions


def _random_partitioning(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> list[list[int]]:
    partitions = []
    max_qpu_size = max(acc.qubits for acc in accelerators) + 1
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in sorted(circuit_sizes, reverse=True):
        partition = []
        remaining_size = circuit_size
        if circuit_size <= 3:
            partitions.append([circuit_size])
            continue
        while remaining_size > 0:
            qpu = np.random.randint(2, max_qpu_size)
            take_qubits = min(remaining_size, qpu)
            if remaining_size - take_qubits == 1:
                # We can't have a partition of size 1
                # So in this case we take one qubit less to leave a partition of two
                take_qubits -= 1
            partition.append(take_qubits)
            remaining_size -= take_qubits
        partitions.append(partition)
    return partitions


def _fixed_partitioning(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> list[list[int]]:
    if "partition_size" not in kwargs:
        partition_size = 10
    else:
        partition_size = kwargs["partition_size"]
    partitions = []
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in sorted(circuit_sizes, reverse=True):
        if circuit_size > partition_size:
            partition = [partition_size] * (circuit_size // partition_size)
            if circuit_size % partition_size != 0:
                partition.append(circuit_size % partition_size)
            if partition[-1] == 1:
                partition[-1] = 2
                partition[-2] -= 1
            partitions.append(partition)
        else:
            partitions.append([circuit_size])
    return partitions


def _convert_to_jobs(
    circuits: list[QuantumCircuit], partitions: list[list[int]]
) -> list[CircuitJob]:
    jobs = []
    for idx, circuit in enumerate(
        sorted(circuits, key=lambda circ: circ.num_qubits, reverse=True)
    ):
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


def _bin_schedule(
    jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[Machine]:
    closed_bins = _do_bin_pack(jobs, [qpu.qubits for qpu in accelerators])
    # Build combined jobs from bins
    machines = []
    for acc in accelerators:
        machines.append(
            Machine(
                capacity=acc.qubits,
                id=str(acc.uuid),
                jobs=[],
                buckets=[],
            )
        )

    for _bin in sorted(closed_bins, key=lambda x: x.index):
        machines[_bin.qpu].jobs += _bin.jobs
        machines[_bin.qpu].buckets.append(Bucket(jobs=_bin.jobs))
    return machines


OPTIONS = [
    # _greedy_partitioning,
    # _even_partitioning,
    # _informed_partitioning,
    # _random_partitioning,
    _choice_partitioning,
    # _fixed_partitioning,
]
