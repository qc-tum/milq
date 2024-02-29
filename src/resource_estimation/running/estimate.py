"""Module for estimating the runtime of a quantum circuit using the resource estimator."""

from qiskit import QuantumCircuit

from .convert import convert_to_qir
from .query import query_resource_estimator


def estimate_runtime(circuit: QuantumCircuit, error_budget: float = 0.05) -> float:
    """Estimate the runtime of a quantum circuit using the resource estimator.

    Args:
        circuit (QuantumCircuit): The quantum circuit to estimate the runtime for.
        error_budget (float, optional): Error budget for the calculation. Defaults to 0.05.

    Returns:
        float: The estimated runtime of the circuit in nano seconds.
    """
    quantum_circuit = circuit.copy()
    quantum_circuit.t([0])  # To make sure the circuit is not empty
    bytecode = convert_to_qir(quantum_circuit)
    result = query_resource_estimator(bytecode, error_budget=error_budget)
    # assert result.get["status"] == "Succeeded"
    runtime = result.data()["physicalCounts"]["runtime"]
    return runtime
