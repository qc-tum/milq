"""AccelertorGroup Tests."""
import pytest
from qiskit.transpiler.exceptions import TranspilerError

from src.circuits import create_ghz
from src.provider import Accelerator, AcceleratorGroup, IBMQBackend


def test_acceleratorgroup_run() -> None:
    """_summary_"""
    backend_belem = IBMQBackend.BELEM
    accelerator_belem = Accelerator(backend_belem)
    backend_nairobi = IBMQBackend.QUITO
    accelerator_nairobi = Accelerator(backend_nairobi)
    accelerator = AcceleratorGroup([accelerator_belem, accelerator_nairobi])
    with pytest.raises(TranspilerError):
        accelerator.run_and_get_counts([create_ghz(3), create_ghz(7)])
