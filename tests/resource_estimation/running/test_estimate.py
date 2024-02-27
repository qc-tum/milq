"""_summary_"""

from os import environ

import pytest


from src.circuits import create_quantum_only_ghz
from src.resource_estimation import estimate_runtime


def test_estimate_runtime() -> None:
    """Test the runtime estimation."""
    if environ.get("AZURE_CLIENT_ID", None) is None:
        pytest.skip("Azure Quantum is not available in the CI environment.")
    circuit = create_quantum_only_ghz(10)
    circuit.t([0])  # Resource Estimation requires a magic state source
    circuit.cx([0], [1])
    runtime = estimate_runtime(circuit)
    assert runtime == 28800
