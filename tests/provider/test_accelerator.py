"""Tests for Accelerator."""

from pytest import approx

from src.provider import Accelerator, IBMQBackend
from src.tools import optimize_circuit_offline
from tests.helpers import create_ghz


def test_accelerator_run() -> None:
    """_summary_"""
    backend = IBMQBackend.BELEM
    accelerator = Accelerator(backend)
    circuit = create_ghz(3)
    circuit = optimize_circuit_offline(circuit, backend)
    counts = accelerator.run_and_get_counts(circuit)
    assert len(counts) == 2**3
    assert counts["000"] / 1024 == approx(0.5, 0.2)
    assert counts["111"] / 1024 == approx(0.5, 0.2)
