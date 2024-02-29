"""_summary_"""

from collections import Counter
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Protocol
import logging

from qiskit import QuantumCircuit
import numpy as np

from src.common import CircuitJob
from src.provider import Accelerator
from src.scheduling.common import (
    Schedule,
    Machine,
    Bucket,
    CircuitProxy,
    convert_to_jobs,
)
from ..bin_schedule import _do_bin_pack


class Option(Protocol):
    """Helper to typehint init options"""

    def __call__(
        self, circuits: QuantumCircuit, accelerators: list[Accelerator], **kwargs
    ) -> list[list[int]]: ...


def initialize_population(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator], **kwargs
) -> list[Schedule]:
    """Initializes a population of schedules for the given circuits and accelerators.

    At the moment supports following partitioning methods:
    - greedy_partitioning: Partitions the circuits in a greedy way, by trying to fit the biggest
        circuits first.
    - even_partitioning: Partitions the circuits in similar sized chunks.
    - informed_partitioning: Finds cuts by recursively cutting the line with the least cnots.
    - choice_partitioning: Randomly chooses a qpu and cuts the current circuit according
        to qpu size.
    - random_partitioning: Randomly chooses the cutsize between 1 - max(qpu_sizes).
    - fixed_partitioning: Partitions the circuits in chunks of a fixed size.

    Args:
        circuits (list[QuantumCircuit]): The initial batch of circuits to schedule.
        accelerators (list[Accelerator]): The available accelerators to schedule the circuits on.

    Returns:
        list[Schedule]: Initial scheduel candidates.
    """

    schedules = []
    num_cores = max(len(OPTIONS), cpu_count())
    with Pool(processes=num_cores) as pool:
        work = partial(_task, circuits=circuits, accelerators=accelerators, **kwargs)
        schedules = pool.map(work, OPTIONS)
    return schedules


def _task(
    option: Option,
    circuits: QuantumCircuit,
    accelerators: list[Accelerator],
    **kwargs,
) -> Schedule:
    logging.debug("Starting init on... %s", option.__name__)
    partitions = option(circuits, accelerators, **kwargs)
    jobs: list[CircuitJob] = convert_to_jobs(circuits, partitions)
    logging.debug("%s  init done.", option.__name__)
    return Schedule(_bin_schedule(jobs, accelerators), 0.0)


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


def _bin_schedule(
    jobs: list[CircuitProxy], accelerators: list[Accelerator]
) -> list[Machine]:
    closed_bins = _do_bin_pack(jobs, [qpu.qubits for qpu in accelerators])
    # Build combined jobs from bins
    machines = []
    for acc in accelerators:
        machines.append(
            Machine(
                capacity=acc.qubits,
                id=str(acc.uuid),
                buckets=[],
            )
        )

    for _bin in sorted(closed_bins, key=lambda x: x.index):
        machines[_bin.qpu].buckets.append(Bucket(jobs=_bin.jobs))
    return machines


OPTIONS = [
    _greedy_partitioning,
    _even_partitioning,
    _informed_partitioning,
    _random_partitioning,
    _choice_partitioning,
    _fixed_partitioning,
]
