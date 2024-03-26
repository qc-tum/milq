"""_summary_"""

from qiskit import QuantumCircuit

from src.common import UserCircuit
from src.provider import Accelerator
from src.scheduling.common import evaluate_final_solution

from .train import run_model
from ..types import JobResultInfo


def generate_rl_info_schedule(
    circuits: list[QuantumCircuit | UserCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> tuple[tuple[float, float, float], list[JobResultInfo]]:
    """Generates a schedule for the given jobs and accelerators using a rl agent.

    Args:
         circuits (list[QuantumCircuit | UserCircuit]): List of circuits (jobs) to schedule.
        accelerators (list[Accelerator]): List of accelerators to schedule on.

    Returns:
        tuple[tuple[float, float, float] list[JobResultInfo]]: 
            The list of jobs with their assigned machine and
            the makespan, score and noise of the schedule.
    """
    setting = {"accelerators": accelerators, "circuits": circuits}
    schedule = run_model(setting)
    combined_jobs = []
    for machine in schedule.machines:
        for idx, bucket in enumerate(machine.buckets):
            for job in bucket.jobs:
                if job is None:
                    continue
                combined_jobs.append(
                    JobResultInfo(
                        name=str(job.uuid),
                        machine=machine.id,
                        start_time=idx,
                        completion_time=-1.0,
                        capacity=job.num_qubits,
                    )
                )
    result = evaluate_final_solution(schedule, accelerators, circuits)
    return result, combined_jobs
