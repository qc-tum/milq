"""_summary_"""

from collections import Counter
from typing import Protocol
import logging

from qiskit import QuantumCircuit
import numpy as np

from src.common import CircuitJob, UserCircuit
from src.resource_estimation import ResourceEstimator, GroupingMethod
from src.provider import Accelerator
from src.scheduling.common import (
    Schedule,
    Machine,
    Bucket,
    CircuitProxy,
    convert_circuits,
    do_bin_pack_proxy as do_bin_pack,
)
from src.tools import generate_subcircuit


class PartitioningScheme(Protocol):
    """Helper to typehint init options"""

    def __call__(
        self, circuits: QuantumCircuit, accelerators: list[Accelerator], **kwargs
    ) -> list[list[int]]: ...


def initialize_population(
    circuits: list[QuantumCircuit | UserCircuit],
    accelerators: list[Accelerator],
    **kwargs,
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
        circuits (list[QuantumCircuit | UserCircuit]): The initial batch of circuits to schedule.
        accelerators (list[Accelerator]): The available accelerators to schedule the circuits on.

    Returns:
        list[Schedule]: Initial schedule candidates.
    """

    schedules = []
    # Azure resource estimator might not work correctly with pool
    for option in OPTIONS:
        logging.info("Starting init on... %s", option.__name__)
        circuits = sorted(circuits, key=lambda circ: circ.num_qubits, reverse=True)
        schedules.append(
            _task(
                option,
                circuits,
                accelerators,
                **kwargs,
            )
        )
    for method in GroupingMethod:
        schedules.append(_cut_task(method, circuits, accelerators))
    return schedules


def _task(
    option: PartitioningScheme,
    circuits: list[QuantumCircuit | UserCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> Schedule:
    logging.debug("Starting init on... %s", option.__name__)
    quantum_circuits = [
        circuit if isinstance(circuit, QuantumCircuit) else circuit.circuit
        for circuit in circuits
    ]
    partitions = option(quantum_circuits, accelerators, **kwargs)
    partitions = _reformat(partitions)
    jobs: list[CircuitJob] = convert_circuits(circuits, accelerators, partitions)
    logging.debug("%s  init done.", option.__name__)
    return Schedule(_bin_schedule(jobs, accelerators), 0.0)


def _reformat(partitions: list[list[int]]) -> list[list[int]]:
    new_partitions = []
    for partition in partitions:
        new_partition = []
        for idx, part in enumerate(partition):
            new_partition += [idx] * part
        new_partitions.append(new_partition)
    return new_partitions


def _cut_task(
    method: GroupingMethod,
    circuits: list[QuantumCircuit | UserCircuit],
    accelerators: list[Accelerator],
) -> Schedule:
    logging.debug("Starting init on... %s", method.value)
    quantum_circuits = [
        circuit if isinstance(circuit, QuantumCircuit) else circuit.circuit
        for circuit in circuits
    ]
    partitions = _better_partitioning(quantum_circuits, accelerators, method)
    jobs: list[CircuitJob] = convert_circuits(circuits, accelerators, partitions)
    logging.debug("%s  init done.", method.value)
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
    for circuit_size in circuit_sizes:
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
    partition_size = sum(acc.qubits for acc in accelerators) // len(accelerators)
    kwargs.update({"partition_size": partition_size})
    return _fixed_partitioning(circuits, accelerators, **kwargs)


def _informed_partitioning(
    circuits: list[QuantumCircuit], accelerators: list[Accelerator], **kwargs
) -> list[list[int]]:
    """Finds cuts by recursively cutting the line with the least cnots"""
    partitions = []
    max_qpu_size = max(acc.qubits for acc in accelerators)

    for circuit in circuits:
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
    cuts = (
        [best_cut]
        + _find_cuts(counts, start, best_cut, max_qpu_size)
        + _find_cuts(counts, best_cut + 1, end, max_qpu_size)
    )

    return cuts


def _calulate_cut(counts: Counter[tuple[int, int]], cut: int) -> tuple[int, int]:
    left = sum(
        count for (first, second), count in counts.items() if first <= cut < second
    )
    right = sum(
        count for (first, second), count in counts.items() if second <= cut < first
    )
    return cut, left + right


def _better_partitioning(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    grouping_method: GroupingMethod,
) -> list[list[int]]:
    """Finds cuts by using ResourceEstimator.resource_optimal"""
    partitions = []
    qpu_sizes = [acc.qubits for acc in accelerators]
    for circuit in circuits:
        resource_estimator = ResourceEstimator(circuit)
        partition = []
        if circuit.num_qubits <= max(qpu_sizes):
            partitions.append([0] * circuit.num_qubits)
            continue
        while circuit.num_qubits > max(qpu_sizes):
            _, partition1, partition2 = resource_estimator.resource_optimal(
                epsilon=0.1,
                delta=0.1,
                size=np.random.choice(qpu_sizes),
                method=grouping_method,
            )
            partition.append(partition1)
            if len(partition2) <= max(qpu_sizes):
                if len(partition2) == 1:
                    partition2.append(partition[-1].pop())
                partition.append(partition2)
                break
            circuit = generate_subcircuit(circuit, partition2)
            resource_estimator = ResourceEstimator(circuit)
        new_partition = [0] * circuit.num_qubits
        for idx, part in enumerate(partition):
            for qubit in part:
                new_partition[qubit] = idx
        partitions.append(new_partition)

    return partitions


def _choice_partitioning(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> list[list[int]]:
    partitions = []
    qpu_sizes = [acc.qubits for acc in accelerators]
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in circuit_sizes:
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
    # +1 to because np.randint is [low, high)
    max_qpu_size = max(acc.qubits for acc in accelerators) + 1
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in circuit_sizes:
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
    partition_size = kwargs.pop("partition_size", 10)
    partitions = []
    circuit_sizes = [circ.num_qubits for circ in circuits]
    for circuit_size in circuit_sizes:
        if circuit_size > partition_size:
            partition = [partition_size] * (circuit_size // partition_size)
            if circuit_size % partition_size != 0:
                partition.append(circuit_size % partition_size)
            if partition[-1] == 1:
                partition[-1] = 2
                assert partition[-2] > 2, "Partition size too small."
                partition[-2] -= 1
            partitions.append(partition)
        else:
            partitions.append([circuit_size])
    return partitions


def _bin_schedule(
    jobs: list[CircuitProxy], accelerators: list[Accelerator]
) -> list[Machine]:
    closed_bins = do_bin_pack(jobs, accelerators)
    # Build combined jobs from bins
    machines = []
    for acc in accelerators:
        machines.append(
            Machine(
                capacity=acc.qubits,
                id=str(acc.uuid),
                buckets=[],
                queue_length=len(acc.queue),
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
