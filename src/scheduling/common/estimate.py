"""Proxy resource estimation."""

from src.circuits import generate_subcircuit
from .types import CircuitProxy


def estimate_runtime_proxy(circuit: CircuitProxy, indices: list[int]) -> float:
    """Calculate noise based on original circuit."""
    quantum_circuit = generate_subcircuit(circuit.origin, indices)
    if circuit.origin.depth() == 0:
        return circuit.processing_time
    return circuit.processing_time * quantum_circuit.depth() / circuit.origin.depth()


def estimate_noise_proxy(circuit: CircuitProxy, indices: list[int]) -> float:
    """Calculate noise based on original circuit."""
    quantum_circuit = generate_subcircuit(circuit.origin, indices)
    return circuit.noise * quantum_circuit.depth() / circuit.origin.depth()
