"""Mapping to hardware using mqt QMAP."""
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from src.common import IBMQBackend


def map_circuit(
    circuit: QuantumCircuit, backend: IBMQBackend
) -> tuple[QuantumCircuit, PassManager]:
    """Maps a circuit to the layout of the given backend.

    Args:
        circuit (QuantumCircuit): The circuit to map
        backend (IBMQBackend): The backend to map to

    Returns:
        tuple[QuantumCircuit, PassManager]: _description_
    """
    target = backend.value().target
    mapping_and_routing = PassManager([SabreLayout(target), SabreSwap(target)])
    circuit = mapping_and_routing.run(circuit)
    return circuit, mapping_and_routing
