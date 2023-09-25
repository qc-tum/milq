""""""
from src.circuits import create_quantum_only_ghz
from src.tools import cut_circuit


def test_cut_circuit() -> None:
    """_summary_"""
    circuit = create_quantum_only_ghz(5)
    experiments, _ = cut_circuit(circuit, [2, 3])
    assert len(experiments) == 2
    assert experiments[0].circuits[0].num_qubits == 2
    assert experiments[1].circuits[1].num_qubits == 3
