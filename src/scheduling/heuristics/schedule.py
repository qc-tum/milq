"""Schedule wrapper for population-based heuristics."""
from uuid import uuid4

from qiskit import QuantumCircuit

from src.common import CircuitJob, ScheduledJob
from src.provider import Accelerator

from .search import scatter_search
from ..types import JobResultInfo


def generate_heuristic_info_schedule(
    circuits: list[QuantumCircuit],
    accelerators: dict[str, int],
) -> list[JobResultInfo]:
    jobs = [
        CircuitJob(
            uuid=uuid4(),
            circuit=job,
            coefficient=None,
            cregs=1,
            index=0,
            n_shots=1024,
            observable="",
            partition_label="",
            result_counts={},
        )
        for job in circuits
    ]
    schedule = scatter_search(jobs, accelerators)
    # TODO create scheduled jobs from schedule
    return []


def generate_heuristic_exec_schedule(
    jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[ScheduledJob]:
    schedule = scatter_search(jobs, {str(acc.uuid): acc.qubits for acc in accelerators})
    # TODO create scheduled jobs from schedule
    return []
