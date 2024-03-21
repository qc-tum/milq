"""_summary_"""

from os import environ

import pytest

from src.resource_estimation import estimate_runtime
from tests.helpers import create_quantum_only_ghz


def test_estimate_runtime() -> None:
    """Test the runtime estimation."""
    if environ.get("AZURE_CLIENT_ID", None) is None:
        pytest.skip("Azure Quantum is not available in the CI environment.")
    circuit = create_quantum_only_ghz(10)
    runtime = estimate_runtime(circuit)
    assert runtime == 28800
