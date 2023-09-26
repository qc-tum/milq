"""_summary_"""

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from src.common import IBMQBackend


def optimize_circuit_offline(
    circuit: QuantumCircuit, backend: IBMQBackend
) -> QuantumCircuit:
    """Optimization without hardware information.

    Should only run high-level optimization.
    Needs to do gate decomposition for cutting to work.
    For now as placeholder init transpile pass from qiskit.

    Args:
        circuit (QuantumCircuit): _description_
        backend (IBMQBackend): _description_

    Returns:
        QuantumCircuit: _description_
    """
    pass_manager = generate_preset_pass_manager(
        3, backend.value()
    )  # TODO eventually remove dependency
    pass_manager.routing = None
    pass_manager.translation = None
    pass_manager.optimization = None
    pass_manager.scheduling = None
    return pass_manager.run(circuit)


def optimize_circuit_online(
    circuit: QuantumCircuit, backend: IBMQBackend
) -> QuantumCircuit:
    """Optimization with hardware information.

    Should only run low-evel optimization.
    For now as placeholder restricted transpile pass from qiskit.

    Args:
        circuit (QuantumCircuit): _description_
        backend (IBMQBackend): _description_

    Returns:
        QuantumCircuit: _description_
    """
    pass_manager = generate_preset_pass_manager(3, backend.value())
    pass_manager.init = None
    pass_manager.layout = None
    return pass_manager.run(circuit)
