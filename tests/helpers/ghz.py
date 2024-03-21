"""GHZ state circuit generators."""
from functools import lru_cache

from qiskit import QuantumCircuit


@lru_cache
def create_ghz(n_qubits: int) -> QuantumCircuit:
    """Generates a n-qubit GHZ state.

    Includes measurement.
    Args:
        n_qubits (int): Number of qubits.

    Returns:
        QuantumCircuit: The quantum circuit object.
    """
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit


@lru_cache
def create_quantum_only_ghz(n_qubits: int) -> QuantumCircuit:
    """Generater a n-qubit GHZ state.

    Without measurement.
    Args:
        n_qubits (int): Number of qubits.

    Returns:
        QuantumCircuit: The quantum circuit object.
    """
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    return circuit
