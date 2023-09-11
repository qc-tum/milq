""""""
from functools import lru_cache

from qiskit import QuantumCircuit


@lru_cache
def create_ghz(n_qubits: int) -> QuantumCircuit:
    """_summary_

    Args:
        n_qubits (int): _description_

    Returns:
        QuantumCircuit: _description_
    """
    circuit = QuantumCircuit(n_qubits, n_qubits)
    circuit.h(0)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    circuit.measure(list(range(n_qubits)), list(range(n_qubits)))
    return circuit
