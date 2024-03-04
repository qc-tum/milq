"""Proxy resource estimation."""

from bisect import bisect_left

from qiskit import QuantumCircuit

from .types import CircuitProxy


def subcircuit(circuit: QuantumCircuit, indices: list[int]) -> QuantumCircuit:
    """Builds a new subcircuit. only retain the qubits in the indices"""
    quantum_circuit = QuantumCircuit(len(indices))
    for gate in circuit.data:
        if all(circuit.find_bit(qubit).index in indices for qubit in gate[1]):
            quantum_circuit.append(
                gate[0],
                [
                    bisect_left(indices, circuit.find_bit(qubit).index)
                    for qubit in gate[1]
                ],
            )
    return quantum_circuit


def estimate_runtime_proxy(circuit: CircuitProxy, indices: list[int]) -> float:
    """Calculate noise based on original circuit."""
    quantum_circuit = subcircuit(circuit.origin, indices)
    if circuit.origin.depth() == 0:
        return circuit.processing_time
    return circuit.processing_time * quantum_circuit.depth() / circuit.origin.depth()


def estimate_noise_proxy(circuit: CircuitProxy, indices: list[int]) -> float:
    """Calculate noise based on original circuit."""
    quantum_circuit = subcircuit(circuit.origin, indices)
    return circuit.origin * quantum_circuit.depth() / circuit.origin.depth()
