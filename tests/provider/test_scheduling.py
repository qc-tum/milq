"""_summary_"""

from src.circuits import create_ghz, create_quantum_only_ghz
from src.common import IBMQBackend
from src.provider import Accelerator, Scheduler
from src.tools import optimize_circuit_offline


def test_generate_schedule() -> None:
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
    jobs_per_qpu = {}
    for job in jobs:
        jobs_per_qpu.setdefault(job.qpu, []).append(job.job)
    assert jobs_per_qpu.keys() == {0, 1}
    jobs_0 = jobs_per_qpu[0]
    jobs_1 = jobs_per_qpu[1]
    assert len(jobs_0) == 5
    assert len(jobs_1) == 5
    qubits_0 = [
        [slice(0, 5)],
        [slice(0, 5)],
        [slice(0, 5)],
        [slice(0, 3), slice(3, 5)],
        [slice(0, 2), slice(2, 4)],
    ]
    for job, qubits in zip(jobs_0, qubits_0):
        assert job.mapping == qubits
    qubits_1 = [
        [slice(0, 5)],
        [slice(0, 5)],
        [slice(0, 5)],
        [slice(0, 2), slice(2, 4)],
        [slice(0, 2)],
    ]
    for job, qubits in zip(jobs_1, qubits_1):
        assert job.mapping == qubits


def test_run_circuits() -> None:
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
    jobs = scheduler.run_circuits(circuits)
    for job in jobs:
        assert job.result_counts is not None
