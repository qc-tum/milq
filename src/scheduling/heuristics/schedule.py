"""Schedule wrapper for population-based heuristics."""

from qiskit import QuantumCircuit

from src.common import ScheduledJob
from src.provider import Accelerator
from src.tools import assemble_job

from .search import scatter_search


def generate_heuristic_info_schedule(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    num_iterations: int = 100,
    num_elite_solutions: int = 10,
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
        num_iterations (int, optional): Number of search iterations. Defaults to 100.
        num_elite_solutions (int, optional): Max number of solutions to keep each round.
            Defaults to 10.

    Returns:
        tuple[list[ScheduledJob], float]: The list of jobs with their assigned machine and
            the makespan of the schedule.
    """
    schedule = scatter_search(
        circuits, accelerators, num_iterations, num_elite_solutions
    )
    combined_jobs = []
    for machine in schedule.machines:

        machind_idx = next(
            idx for idx, acc in enumerate(accelerators) if str(acc.uuid) == machine.id
        )
        for bucket in machine.buckets:
            combined_jobs.append(ScheduledJob(assemble_job(bucket.jobs), machind_idx))
    return combined_jobs, schedule.makespan
