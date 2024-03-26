"""Generate subcircuits from a quantum circuit."""

from bisect import bisect_left

from qiskit import QuantumCircuit


def generate_subcircuit(circuit: QuantumCircuit, indices: list[int]) -> QuantumCircuit:
    """Builds a new subcircuit. Only retain the qubits in the indices

    Args:
        circuit (QuantumCircuit): A quantum circuit to generate a subcircuit from.
        indices (list[int]): Indixes of the qubits to keep.

    Returns:
        QuantumCircuit: Subcircuit of size len(indices).
    """
    quantum_circuit = QuantumCircuit(len(indices))
    for instruction in circuit.data:
        [op, qubits, _] = instruction
        if all(circuit.find_bit(qubit).index in indices for qubit in qubits):
            quantum_circuit.append(
                op,
                [
                    bisect_left(indices, circuit.find_bit(qubit).index)
                    for qubit in qubits
                ],
            )
    return quantum_circuit
