"""_summary_"""

from uuid import uuid4

from qiskit import QuantumCircuit

from src.resource_estimation import estimate_runtime

from .fake_cut import cut_proxies
from .types import CircuitProxy


def convert_circuits(
    circuits: list[QuantumCircuit], partitions: list[list[int]] | None = None
) -> list[CircuitProxy]:
    """Converts to their proxy representation.

    This adds the runtime information.
    If partitions are given, the circuits are cut into the partitions
    And new proxies are created instead of new circuits.
    TODO: So far only works with a two partitions per circuit.

    Args:
        circuits (list[QuantumCircuit]): The circuits to convert.
        partitions (list[list[int]] | None, optional): The possible partitions. Defaults to None.

    Returns:
        list[CircuitProxy]: The converted circuits with runtime estimate.
    """
    if partitions is None:
        return [convert_to_proxy(circuit) for circuit in circuits]
    proxies = [convert_to_proxy(circuit) for circuit in circuits]
    return cut_proxies(proxies, partitions)


def convert_to_proxy(circuit: QuantumCircuit, n_shots: int = 1024) -> CircuitProxy:
    """Convert a quantum circuit to a CircuitProxy.

    This is used to only calculate the runtime of a circuit once.
    Args:
        circuit (QuantumCircuit): The quantum circuit to convert.
        n_shots (int, optional): The number of shots. Defaults to 1024.

    Returns:
        CircuitProxy: The new proxy to the circuit.
    """
    processing_time = estimate_runtime(circuit)
    return CircuitProxy(
        origin=circuit,
        processing_time=processing_time,
        num_qubits=circuit.num_qubits,
        indices=list(range(circuit.num_qubits)),
        uuid=uuid4(),
        n_shots=n_shots,
    )
