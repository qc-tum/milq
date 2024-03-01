"""_summary_"""

from bisect import bisect_left
from functools import reduce
from operator import mul
from qiskit import QuantumCircuit

from src.resource_estimation import ResourceEstimator

from .types import CircuitProxy


def partion_circuit(circuit: CircuitProxy, partitions: list[int]) -> list[CircuitProxy]:
    """Partitions a circuit into subcircuits based on the partitions.

    Repeateadly cuts the circuit into subcircuits based on the partitions.
    Update the number of shots for each subcircuit.
    # TODO: Check if the number of shots is correct.
    
    Args:
        circuit (CircuitProxy): _description_
        partitions (list[int]): _description_

    Returns:
        list[CircuitProxy]: _description_
    """
    assert len(partitions) == circuit.num_qubits, "Partitions must match qubits."
    if len(set(partitions)) == 2:
        return _bipartition(circuit, partitions)
    original_circuit = circuit.origin
    proxies = []
    samples = []
    multiplier = 1
    for partition in range(len(set(partitions))):
        if partition == len(set(partitions)) - 2:
            proxies += (
                _bipartition(
                    circuit,
                    [
                        int(value > partition)
                        for value in partitions
                        if value >= partition
                    ],
                )
                * multiplier
            )
            samples.append(proxies[-1].n_shots * multiplier)
            break
        additional_proxies = _partition(
            circuit,
            [int(value > partition) for value in partitions if value >= partition],
            partitions,
            partition,
        )
        proxies += additional_proxies * multiplier
        multiplier *= len(additional_proxies)
        samples.append(proxies[-1].n_shots)
        # remove all qubits from the current partition
        circuit.origin = _subcircuit(
            circuit.origin,
            [idx for idx, value in enumerate(partitions) if value > partition],
        )

    n_shots = reduce(mul, samples, 1)
    for proxy in proxies:
        proxy.n_shots = n_shots  # TODO check if this is correct
        proxy.origin = original_circuit
    return proxies


def _bipartition(
    circuit: CircuitProxy,
    binary_partition: list[int],
) -> list[CircuitProxy]:
    """Bipartitions the circut, giving back both parts, not to be cut further."""
    estimator = ResourceEstimator(circuit.origin)
    resource = estimator.resource(
        binary=binary_partition, epsilon=0.1, delta=0.1, method="simple"
    )
    n_shots = resource.n_samples // (2 * resource.n_circuit_pairs)
    proxies = []
    for _ in range(resource.n_circuit_pairs):
        indices_1 = [idx for idx, value in enumerate(binary_partition) if value == 0]
        proxy_part_1 = CircuitProxy(
            origin=circuit.origin,
            processing_time=estimate_runtime_proxy(circuit, indices_1),
            num_qubits=len(indices_1),
            indices=indices_1,
            uuid=circuit.uuid,
            n_shots=n_shots,
        )
        indices_2 = [idx for idx, value in enumerate(binary_partition) if value == 1]
        proxy_part_2 = CircuitProxy(
            origin=circuit.origin,
            processing_time=estimate_runtime_proxy(circuit, indices_2),
            num_qubits=len(indices_2),
            indices=indices_2,
            uuid=circuit.uuid,
            n_shots=n_shots,
        )
        proxies += [proxy_part_1, proxy_part_2]
    return proxies


def _partition(
    circuit: CircuitProxy,
    binary_partition: list[int],
    all_partitions: list[int],
    index: int,
) -> list[CircuitProxy]:
    """Cuts of a partition for a circuit, rest will be cut further.

    Args:
        circuit (CircuitProxy): The circuit to cut. (likely a subcircuit of the original circuit)
        binary_partition (list[int]): Binary do indicate where to cut (0 = cut, 1 = keep)
        all_partitions (list[int]): The original list of partitions to construct the subcircuit.
        index (int): Current index of the partition to contruct the subcircuit.

    Returns:
        list[CircuitProxy]: The n_circuit_pairs proxies for the subcircuit after cutting.
    """
    estimator = ResourceEstimator(circuit.origin)
    resource = estimator.resource(
        binary=binary_partition, epsilon=0.1, delta=0.1, method="simple"
    )
    n_shots = resource.n_samples // (2 * resource.n_circuit_pairs)
    proxies = []
    for _ in range(resource.n_circuit_pairs):
        indices = [idx for idx, value in enumerate(all_partitions) if value == index]
        proxy = CircuitProxy(
            origin=circuit.origin,
            processing_time=estimate_runtime_proxy(circuit, indices),
            num_qubits=len(indices),
            indices=indices,
            uuid=circuit.uuid,
            n_shots=n_shots,
        )
        proxies.append(proxy)

    return proxies


def _subcircuit(circuit: QuantumCircuit, indices: list[int]) -> QuantumCircuit:
    """Builds a new subcircuit. only retain the qubits in the indices"""
    quantum_circuit = QuantumCircuit(len(indices))
    for gate in circuit.data:
        if all(circuit.find_bit(qubit).index in indices for qubit in gate[1]):
            quantum_circuit.append(
                gate[0],
                [
                    bisect_left(indices, circuit.find_bit(qubit).index)
                    for qubit in gate[1]
                ],
            )
    return quantum_circuit


def estimate_runtime_proxy(circuit: CircuitProxy, indices: list[int]) -> float:
    """Calculate runtime based on original circuit."""
    quantum_circuit = _subcircuit(circuit.origin, indices)
    return circuit.processing_time * quantum_circuit.depth() / circuit.origin.depth()


def cut_proxies(
    circuits: list[CircuitProxy], partitions: list[list[int]]
) -> list[CircuitProxy]:
    """Cuts the proxies according to their partitions.

    Args:
        circuits (list[CircuitProxy]): The proxies to cut.
        partitions (list[list[int]]): The partitions to cut the proxies into.

    Returns:
        list[CircuitProxy]: The resulting proxies.
    """
    jobs = []
    for idx, circuit in enumerate(
        sorted(circuits, key=lambda circ: circ.num_qubits, reverse=True)
    ):
        if len(partitions[idx]) > 1:
            jobs += partion_circuit(circuit, partitions[idx])
        else:
            jobs.append(circuit)
    return jobs