"""Schedule wrapper for population-based heuristics."""

from qiskit import QuantumCircuit

from src.common import ScheduledJob, UserCircuit
from src.provider import Accelerator
from src.tools import assemble_job
from src.scheduling.common import evaluate_final_solution

from .search import scatter_search
from ..types import JobResultInfo


def generate_heuristic_exec_schedule(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> tuple[list[ScheduledJob], float]:
    """Generates a schedule for the given jobs and accelerators using a scatter search heuristic.

    TODO:
    - Adapt to the existing interface of info/exec schedule
    - (Parallelize scatter search?)
    - Find meta-parameters for scatter search
    - Improve the heuristic (temperature, tabu list)
    - Find a good way to implement init options

    Args:
        circuits (list[QuantumCircuit]): List of circuits (jobs) to schedule.
        accelerators (list[Accelerator]): List of accelerators to schedule on.


    Returns:
        tuple[list[ScheduledJob], float]: The list of jobs with their assigned machine and
            the makespan of the schedule.
    """
    schedule = scatter_search(circuits, accelerators, **kwargs)
    combined_jobs = []
    for machine in schedule.machines:

        machin_idx = next(
            idx for idx, acc in enumerate(accelerators) if str(acc.uuid) == machine.id
        )
        for bucket in machine.buckets:
            combined_jobs.append(ScheduledJob(assemble_job(bucket.jobs), machin_idx))
    return combined_jobs, schedule.makespan


def generate_heuristic_info_schedule(
    circuits: list[QuantumCircuit | UserCircuit],
    accelerators: list[Accelerator],
    **kwargs,
) -> tuple[tuple[float, float, float], list[JobResultInfo]]:
    """tmp workaround"""
    schedule = scatter_search(circuits, accelerators, **kwargs)
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
