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
    quantum_circuit = circuit.remove_final_measurements(inplace=False)
    assert quantum_circuit is not None

    # Workaround because Azure Estimator does not work with circuits of only Clifford gates
    quantum_circuit.t(0) 

    bytecode = convert_to_qir(quantum_circuit)
    try:
        result = query_resource_estimator(bytecode, error_budget=error_budget)
    except RuntimeError:
        # Default value if no T state is present
        return 28800.0
    runtime = result.data()["physicalCounts"]["runtime"]
    return runtime
