"""_summary_"""

from qiskit_aer import AerSimulator

from src.circuits import create_quantum_only_ghz
from src.common import IBMQBackend

from src.resource_estimation import estimate_noise


def test_estimate_noise() -> None:
    """_summary_"""
    backend = IBMQBackend.NAIROBI
    simulator = AerSimulator.from_backend(backend.value())
    circuit = create_quantum_only_ghz(5)
    noise = estimate_noise(circuit, simulator)
    assert noise < 0.1
