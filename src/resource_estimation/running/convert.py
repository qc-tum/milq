"""Wrapper for the QIR conversion function using qiskit-qir."""

from qiskit import QuantumCircuit
from qiskit_qir import to_qir_module


def convert_to_qir(circuit: QuantumCircuit) -> bytes:
    """Convert a quantum circuit to QIR.

    Args:
        quantum_circuit (QuantumCircuit): The quantum circuit to convert.

    Returns:
        bytes: The QIR program as bytecode.
    """
    module, _ = to_qir_module(circuit)
    return module.bitcode
