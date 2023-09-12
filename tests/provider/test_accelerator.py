"""Tests for Accelerator."""

from pytest import approx

from src.circuits import create_ghz
from src.provider import Accelerator, IBMQBackend


def test_accelerator_run() -> None:
    backend = IBMQBackend.BELEM
    accelerator = Accelerator(backend)
    circuit = create_ghz(3)
    counts = accelerator.run_and_get_counts(circuit)
    assert len(counts) == 2**3
    assert counts["000"] / 1024 == approx(0.5, 0.2)
    assert counts["111"] / 1024 == approx(0.5, 0.2)
