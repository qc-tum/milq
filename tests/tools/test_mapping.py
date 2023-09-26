""""""
from src.circuits import create_ghz
from src.provider import Accelerator, IBMQBackend
from src.tools import map_circuit


def test_map_circuit() -> None:
    """_summary_"""
    circuit = create_ghz(5)
    backend = IBMQBackend.BELEM
    accelerator = Accelerator(backend)
    circuit, _ = map_circuit(circuit, accelerator.backend)
    # TODO better assertion
    assert circuit.num_qubits == 5
