"""Mapping to hardware using mqt QMAP."""
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from src.provider import IBMQBackend


def map_circuit(circuit: QuantumCircuit, backend: IBMQBackend) -> QuantumCircuit:
    """_summary_

    Args:
        circuit (QuantumCircuit): _description_

    Returns:
        QuantumCircuit: _description_
    """
    target = backend.value().target
    mapping_and_routing = PassManager([SabreLayout(target), SabreSwap(target)])
    circuit = mapping_and_routing.run(circuit)
    return circuit
