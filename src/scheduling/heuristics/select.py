"""Evaluation and selection of solutions."""

from src.provider import Accelerator

from .types import Schedule, Bucket, MakespanInfo


def select_elite_solutions(
    population: list[Schedule], num_solutions: int, accelerators: list[Accelerator]
) -> list[Schedule]:
    """Selects the #num_solutions best solutions from the population by makespan.

    Args:
        population (list[Schedule]): List of schedules to select from.
        num_solutions (int): Number of solutions to select.
        accelerators (list[Accelerator]): Reference to the accelerators for makespan calculation.

    Raises:
        ValueError: If the population is empty.

    Returns:
        list[Schedule]: The #num_solutions best schedules with lowest makespan.
    """
    if len(population) == 0:
        raise ValueError("Population must not be empty.")

    population = [_evaluate_solution(schedule, accelerators) for schedule in population]
    return sorted(population, key=lambda x: x.makespan)[:num_solutions]


def select_best_solution(
    population: list[Schedule], accelerators: list[Accelerator]
) -> Schedule:
    """Selects the best solution from the population by makespan.

    Args:
        population (list[Schedule]): List of schedules to select from.
        accelerators (list[Accelerator]): Reference to the accelerators for makespan calculation.

    Returns:
        Schedule: The schedule with the lowest makespan.
    """
    return select_elite_solutions(population, 1, accelerators)[0]


def _evaluate_solution(schedule: Schedule, accelerators: list[Accelerator]) -> Schedule:
    makespans = []
    for machine in schedule.machines:
        accelerator = next(acc for acc in accelerators if str(acc.uuid) == machine.id)
        makespans.append(_calc_machine_makespan(machine.buckets, accelerator))
    schedule.makespan = max(makespans)
    return schedule


def _calc_machine_makespan(buckets: list[Bucket], accelerator: Accelerator) -> float:
    makespan = 0
    jobs: list[MakespanInfo] = []
    for idx, bucket in enumerate(buckets):
        # assumption: jobs take the longer of both circuits to execute and to set up
        jobs += [
            MakespanInfo(
                job=job,
                start_time=idx,
                completion_time=-1.0,
                capacity=job.num_qubits,
            )
            for job in bucket.jobs
        ]

    assigned_jobs = jobs.copy()
    for job in jobs:
        last_completed = max(
            (job for job in assigned_jobs), key=lambda x: x.completion_time
        )
        if job.start_time == 0.0:
            last_completed = MakespanInfo(None, 0.0, 0.0, 0)
        job.start_time = last_completed.completion_time
        job.completion_time = (
            last_completed.completion_time
            + accelerator.compute_processing_time(job.job)
            + accelerator.compute_setup_time(last_completed.job, job.job)
        )

    return makespan
