"""_summary_"""

from src.circuits import create_quantum_only_ghz, generate_subcircuit
from src.scheduling.common import convert_circuits, partion_circuit


def test_subcircuit() -> None:
    """_summary_"""
    circuit = create_quantum_only_ghz(5)
    circuit.cz(1, 3)
    new_subcircuit = generate_subcircuit(circuit, [1, 3])
    assert len(new_subcircuit.data) == 1
    assert new_subcircuit.num_qubits == 2


def test_partition_circuit() -> None:
    """_summary_"""
    circuit = create_quantum_only_ghz(5)
    partions = [1, 0, 0, 0, 1]

    subcircuits = partion_circuit(convert_circuits([circuit], [])[0], partions)
    assert len(subcircuits) == 72

    circuit = create_quantum_only_ghz(6)
    partions = [0, 0, 1, 1, 2, 2]

    subcircuits = partion_circuit(convert_circuits([circuit], [])[0], partions)
    assert (
        len(subcircuits) == 78
    )  # 6 for first patition 6*6 for second and third partition each
