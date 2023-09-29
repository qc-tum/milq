"""_summary_"""

from src.circuits import create_ghz, create_quantum_only_ghz
from src.common import IBMQBackend
from src.provider import Accelerator, Scheduler
from src.tools import optimize_circuit_offline


def test_schedluer() -> None:
    """_summary_"""
    backend_belem = IBMQBackend.BELEM
    accelerator_belem = Accelerator(backend_belem)
    backend_quito = IBMQBackend.QUITO
    accelerator_quito = Accelerator(backend_quito)
    scheduler = Scheduler([accelerator_belem, accelerator_quito])
    circuits = [create_quantum_only_ghz(7), create_ghz(3)]
    circuits = [
        optimize_circuit_offline(circuit, backend_belem) for circuit in circuits
    ]
    jobs = scheduler.generate_schedule(circuits)
    # should be:
    # 1. qpu0 -> 5 qubits, qpu1 -> 5 qubits
    # 2. qpu0 -> 5 qubits, qpu1 -> 5 qubits
    # 3. qpu0 -> 5 qubits, qpu1 -> 5 qubits
    # 4. qpu0 -> 3,2 qubits, qpu1 -> 2,2 qubits
    # 5. qpu0 -> 2,2 qubits, qpu1 -> 2 qubits
    assert len(jobs) == 10
