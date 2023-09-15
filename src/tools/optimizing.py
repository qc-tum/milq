"""_summary_"""

from qiskit import QuantumCircuit, transpile

from src.provider import IBMQBackend


def optimize_circuit_offline(circuit: QuantumCircuit) -> QuantumCircuit:
    ...


def optimize_circuit_online(
    circuit: QuantumCircuit, backend: IBMQBackend
) -> QuantumCircuit:
    """Optimization with hardware information.

    Should only run lowlevel optimization.
    For now as placeholder a full transpile pass from qiskit.

    Args:
        circuit (QuantumCircuit): _description_
        backend (IBMQBackend): _description_

    Returns:
        QuantumCircuit: _description_
    """
    return transpile(circuit, backend.value())
