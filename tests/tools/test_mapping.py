""""""

from src.provider import IBMQBackend
from src.tools import map_circuit
from tests.helpers import create_ghz


def test_map_circuit() -> None:
    """_summary_"""
    circuit = create_ghz(5)
    backend = IBMQBackend.BELEM
    circuit, _ = map_circuit(circuit, backend)

    # Check the mapped circuit against the backend's coupling map
    edges = backend.value().coupling_map.get_edges()
    for [_, qubits, _] in circuit.data:
        indices = [circuit.find_bit(qubit).index for qubit in qubits]
        if len(indices) == 1:
            # Single qubit gates are always possible
            continue
        elif len(indices) == 2:
            # Check if the two qubits are indeed physically connected
            [a, b] = indices
            assert (a, b) in edges
        else:
            # Don't know how multi-qubit gates are handled in the coupling map
            raise NotImplementedError
