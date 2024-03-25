"""_summary_"""

from functools import reduce
from operator import mul

from src.tools import generate_subcircuit
from src.resource_estimation import ResourceEstimator

from .estimate import estimate_noise_proxy, estimate_runtime_proxy
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
        circuit.origin = generate_subcircuit(
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
    # TODO find grouping method once its supported by knitting toolbox
    resource = estimator.resource(binary=binary_partition, epsilon=0.1, delta=0.1)
    n_shots = resource.n_samples // resource.n_circuits
    proxies = []
    for _ in range(resource.n_circuits // 2):
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
        list[CircuitProxy]: The n_circuits proxies for the subcircuit after cutting.
    """
    estimator = ResourceEstimator(circuit.origin)
    # TODO find grouping method once its supported by knitting toolbox
    resource = estimator.resource(binary=binary_partition, epsilon=0.1, delta=0.1)
    n_shots = resource.n_samples // resource.n_circuits
    proxies = []
    for _ in range(resource.n_circuits):
        indices = [idx for idx, value in enumerate(all_partitions) if value == index]
        proxy = CircuitProxy(
            origin=circuit.origin,
            processing_time=estimate_runtime_proxy(circuit, indices),
            num_qubits=len(indices),
            indices=indices,
            uuid=circuit.uuid,
            n_shots=n_shots,
            noise=estimate_noise_proxy(circuit, indices),
        )
        proxies.append(proxy)

    return proxies


def cut_proxies(
    circuits: list[CircuitProxy], partitions: list[list[int]]
) -> list[CircuitProxy]:
    """Cuts the proxies according to their partitions.

    Args:
        circuits (list[CircuitProxy]): The proxies to cut.
            Has to be sorted descending by num_qubits.
        partitions (list[list[int]]): The partitions to cut the proxies into.

    Returns:
        list[CircuitProxy]: The resulting proxies.
    """
    jobs = []
    for partition, circuit in zip(
        partitions,
        circuits,
        strict=True,
    ):
        if len(partition) > 1:
            jobs += partion_circuit(circuit, partition)
        else:
            jobs.append(circuit)
    return jobs
