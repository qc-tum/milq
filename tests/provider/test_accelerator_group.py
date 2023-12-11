"""AccelertorGroup Tests."""
import pytest
from qiskit import transpile
from qiskit.transpiler.exceptions import TranspilerError

from src.circuits import create_ghz
from src.provider import Accelerator, AcceleratorGroup, IBMQBackend


# @pytest.mark.skip(
#     reason="Error does not get raised if transpile is not done in accelerator."
# )
def test_acceleratorgroup_run() -> None:
    """_summary_"""
    backend_belem = IBMQBackend.BELEM
    accelerator_belem = Accelerator(backend_belem)
    backend_quito = IBMQBackend.QUITO
    accelerator_quito = Accelerator(backend_quito)
    accelerator = AcceleratorGroup([accelerator_belem, accelerator_quito])
    with pytest.raises(TranspilerError):
        accelerator.run_and_get_counts(
            [
                transpile(create_ghz(3), backend_belem.value()),
                transpile(create_ghz(7), backend_quito.value()),
            ]
        )
