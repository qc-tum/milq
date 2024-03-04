"""Noise estimation for IBM devices."""

from mapomatic import deflate_circuit, matching_layouts, evaluate_layouts
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def estimate_noise(circuit: QuantumCircuit, simulator: AerSimulator) -> float:
    """Estimates the noise of a circuit on an IBM backend.

    Args:
        circuit (QuantumCircuit): The circuit to estimate the noise for.
        simulator (Simulator): The simulator (or any IBM backend) to estimate the noise on.

    Returns:
        float: The estimated noise of the circuit on the accelerator.
    """
    circuit = transpile(circuit, simulator)
    small_circuit = deflate_circuit(circuit)
    layouts = matching_layouts(small_circuit, simulator, call_limit=1000)
    scores = evaluate_layouts(small_circuit, layouts, simulator)
    if len(scores) == 0:
        return 0.1  # High Value unfortunately evaluate layouts is a bit buggy
    return scores[0][1]
