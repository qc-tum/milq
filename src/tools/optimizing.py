"""Optimizing circuits using the Qiskit transpiler."""

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.common import IBMQBackend
from .mapping import map_circuit


def optimize_circuit_offline(
    circuit: QuantumCircuit, backend: IBMQBackend
) -> QuantumCircuit:
    """Optimization without hardware information.

    Should only run high-level optimization.
    Needs to do gate decomposition for cutting to work.
    For now, as placeholder init transpile pass from qiskit.

    Args:
        circuit (QuantumCircuit): _description_
        backend (IBMQBackend): _description_

    Returns:
        QuantumCircuit: _description_
    """
    pass_manager = generate_preset_pass_manager(
        3, backend.value()
    )  # TODO eventually remove dependency
    pass_manager.layout = None
    pass_manager.optimization = None
    pass_manager.routing = None
    pass_manager.scheduling = None
    pass_manager.translation = None
    return pass_manager.run(circuit)


def optimize_circuit_online(
    circuit: QuantumCircuit, backend: IBMQBackend
) -> QuantumCircuit:
    """Optimization with hardware information.

    Should only run low-level optimization.
    For now, as placeholder restricted transpile pass from qiskit.

    Args:
        circuit (QuantumCircuit): _description_
        backend (IBMQBackend): _description_

    Returns:
        QuantumCircuit: _description_
    """
    pass_manager = generate_preset_pass_manager(3, backend.value())
    pass_manager.init = None
    _, pass_manager.layout = map_circuit(circuit, backend)
    return pass_manager.run(circuit)
