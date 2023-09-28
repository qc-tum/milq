"""Assemble a single circuit from multiple independent ones."""
from qiskit import QuantumCircuit


def assemble_circuit(circuits: list[QuantumCircuit]) -> QuantumCircuit:
    """_summary_

    Args:
        circuits (list[QuantumCircuit]): _description_

    Returns:
        QuantumCircuit: _description_
    """
    qubits, clbits = [
        sum(x)
        for x in zip(
            *[(circuit.num_qubits, circuit.num_clbits) for circuit in circuits]
        )
    ]
    composed_circuit = QuantumCircuit(qubits, clbits)
    qubits, clbits = 0, 0
    for circuit in circuits:
        composed_circuit.compose(
            circuit,
            qubits=list(range(qubits, qubits + circuit.num_qubits)),
            clbits=list(range(clbits, clbits + circuit.num_clbits)),
            inplace=True,
        )
        qubits += circuit.num_qubits
        clbits += circuit.num_clbits
    return composed_circuit
