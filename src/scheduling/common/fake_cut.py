"""_summary_"""

from src.resource_estimation import ResourceEstimator

from .types import CircuitProxy


def fake_cut(circuit: CircuitProxy, partition: list[int]) -> list[CircuitProxy]:
    """Fake the cutting of a circuit.
    # TODO multiple partitions
    """
    estimator = ResourceEstimator(circuit.origin)
    resource = estimator.resource(
        binary=partition, epsilon=0.1, delta=0.1, method="simple"
    )
    n_shots = resource.n_samples // (2 * resource.n_circuit_pairs)
    proxies = []
    for _ in range(resource.n_circuit_pairs):

        proxy_part_1 = CircuitProxy(
            origin=circuit.origin,
            processing_time=estimate_runtime_proxy(circuit, partition) * n_shots,
            num_qubits=circuit.num_qubits,
            indices=partition,
            uuid=circuit.uuid,
        )
        proxy_part_2 = CircuitProxy(
            origin=circuit.origin,
            processing_time=estimate_runtime_proxy(circuit, partition) * n_shots,
            num_qubits=circuit.num_qubits,
            indices=partition,
            uuid=circuit.uuid,
        )
        proxies.append(proxy_part_1, proxy_part_2)
    return proxies


def estimate_runtime_proxy(circuit: CircuitProxy, partition: list[int]) -> float:
    """Calculate runtime based on original circuit."""
    return 0.0  # TODO calculate


def cut_proxies(
    circuits: list[CircuitProxy], partitions: list[list[int]]
) -> list[CircuitProxy]:
    """Cuts the proxies according to their partitions.

    Args:
        circuits (list[CircuitProxy]): The proxies to cut.
        partitions (list[list[int]]): THe partitions to cut the proxies into.

    Returns:
        list[CircuitProxy]: The resulting proxies.
    """
    jobs = []
    for idx, circuit in enumerate(
        sorted(circuits, key=lambda circ: circ.num_qubits, reverse=True)
    ):
        if len(partitions[idx]) > 1:
            jobs += fake_cut(circuit, partitions[idx])

        else:
            jobs.append(circuit)
    return jobs
