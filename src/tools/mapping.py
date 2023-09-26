"""Mapping to hardware using mqt QMAP."""
from typing import Tuple

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

from src.common import IBMQBackend


def map_circuit(
    circuit: QuantumCircuit, backend: IBMQBackend
) -> Tuple[QuantumCircuit, PassManager]:
    """_summary_

    Args:
        circuit (QuantumCircuit): _description_
        backend (IBMQBackend): _description_

    Returns:
        Tuple[QuantumCircuit, PassManager]: _description_
    """
    target = backend.value().target
    mapping_and_routing = PassManager([SabreLayout(target), SabreSwap(target)])
    circuit = mapping_and_routing.run(circuit)
    return circuit, mapping_and_routing
