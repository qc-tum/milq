"""_summary_"""

from uuid import uuid4

from qiskit import QuantumCircuit

from src.common import UserCircuit
from src.provider import Accelerator
from src.resource_estimation import estimate_runtime, predict_device

from .partition import cut_proxies
from .types import CircuitProxy


def convert_circuits(
    circuits: list[QuantumCircuit | UserCircuit],
    accelerators: list[Accelerator],
    partitions: list[list[int]] | None = None,
) -> list[CircuitProxy]:
    """Converts to their proxy representation.

    This adds the runtime information.
    If partitions are given, the circuits are cut into the partitions
    And new proxies are created instead of new circuits.

    Args:
        circuits (list[QuantumCircuit | Usercircuit]): The circuits to convert.
        accelerators (list[Accelerator]): The list of accelerators for noise calculation.
        partitions (list[list[int]] | None, optional): The possible partitions. Defaults to None.

    Returns:
        list[CircuitProxy]: The converted circuits with runtime estimate.
    """

    circuits = sorted(circuits, key=lambda circ: circ.num_qubits, reverse=True)
    if partitions is None:
        return [convert_to_proxy(circuit, accelerators) for circuit in circuits]
    proxies = [convert_to_proxy(circuit, accelerators) for circuit in circuits]
    return cut_proxies(proxies, partitions)


def convert_to_proxy(
    circuit: QuantumCircuit | UserCircuit,
    accelerators: list[Accelerator],
    n_shots: int = 1024,
) -> CircuitProxy:
    """Convert a quantum circuit to a CircuitProxy.

    This is used to only calculate the runtime and noise of a circuit once.
    Args:
        circuit (QuantumCircuit | Usercircuit): The quantum circuit to convert.
        accelerators (list[Accelerator]): The list of accelerators for noise calculation.
        n_shots (int, optional): The number of shots. Defaults to 1024.

    Returns:
        CircuitProxy: The new proxy to the circuit.
    """
    tmp_circuit = circuit
    if isinstance(circuit, UserCircuit):
        tmp_circuit = circuit.circuit
    processing_time = estimate_runtime(tmp_circuit)
    noise = max(
        accelerator.compute_noise(tmp_circuit)
        for accelerator in accelerators
        if accelerator is not None
    )
    proxy = CircuitProxy(
        origin=tmp_circuit,
        processing_time=processing_time,
        num_qubits=tmp_circuit.num_qubits,
        indices=list(range(tmp_circuit.num_qubits)),
        uuid=uuid4(),
        n_shots=n_shots,
        noise=noise,
    )
    preference = predict_device(tmp_circuit, [str(acc.uuid) for acc in accelerators])
    if isinstance(circuit, QuantumCircuit):
        proxy.preselection = processing_time
        return proxy

    proxy.priority = circuit.priority
    proxy.strictness = circuit.strictness
    proxy.preselection = (
        circuit.machine_preference if preference is None else preference
    )
