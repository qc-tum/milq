""""""
from src.circuits import create_ghz
from src.tools import assemble_circuit


def test_assemble_circuit() -> None:
    """_summary_"""
    circuits = [create_ghz(3), create_ghz(2)]
    circuit = assemble_circuit(circuits)
    assert circuit.num_qubits == 5
    # 5 Measure, 2 H, 3 CX
    assert sum(circuit.count_ops().values()) == 10
