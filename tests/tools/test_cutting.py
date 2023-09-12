""""""
from src.circuits import create_quantum_only_ghz
from src.tools import cut_circuit


def test_cut_circuit() -> None:
    """_summary_"""
    circuit = create_quantum_only_ghz(5)
    cut_circuits = cut_circuit(circuit, [2, 3])
    assert len(cut_circuits) == 2
    assert cut_circuits[0].num_qubits == 2
    assert cut_circuits[1].num_qubits == 3
