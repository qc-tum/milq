"""Evaluation and selection of solutions."""

from uuid import UUID
import logging

from src.provider import Accelerator
from .types import Schedule, Machine, Bucket, MakespanInfo, is_feasible


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
    logging.info("Selecting elite solutions...")
    if len(population) == 0:
        raise ValueError("Population must not be empty.")

    population = [evaluate_solution(schedule, accelerators) for schedule in population]
    logging.info("Evaluation done.")
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
    logging.info("Selecting best solution.")
    for solution in select_elite_solutions(population, len(population), accelerators):
        if is_feasible(solution):
            return solution
    return population[0]


def select_diverse_solutions(
    population: list[Schedule], num_solutions: int
) -> list[Schedule]:
    """Selects the #num_solutions most diverse solutions from the population.

    Args:
        population (list[Schedule]): List of schedules to select from.
        num_solutions (int): Number of solutions to select.

    Returns:
        list[Schedule]: The #num_solutions most diverse schedules.
    """
    return sorted(
        population,
        key=lambda x: sum(
            _hamming_proxy(x, other) for other in population if other != x
        ),
    )[-num_solutions:]


def evaluate_solution(schedule: Schedule, accelerators: list[Accelerator]) -> Schedule:
    """Calculates and updates the makespan of a schedule.

    Args:
        schedule (Schedule): A schedule to evaluate.
        accelerators (list[Accelerator]): The list of accelerators to schedule on.

    Returns:
        Schedule: The schedule with updated makespan and machine makespans.
    """
    logging.info("Evaluating makespan...")
    makespans = []
    for machine in schedule.machines:
        accelerator = next(acc for acc in accelerators if str(acc.uuid) == machine.id)
        makespans.append(_calc_machine_makespan(machine.buckets, accelerator))
        machine.makespan = makespans[-1]
    schedule.makespan = max(makespans)
    return schedule


def _calc_machine_makespan(buckets: list[Bucket], accelerator: Accelerator) -> float:
    jobs: list[MakespanInfo] = []
    for idx, bucket in enumerate(buckets):
        # assumption: jobs take the longer of both circuits to execute and to set up
        jobs += [
            MakespanInfo(
                job=job.circuit,
                start_time=idx,
                completion_time=-1.0,
                capacity=job.circuit.num_qubits,
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
    if len(jobs) == 0:
        return 0.0
    return max(jobs, key=lambda j: j.completion_time).completion_time


def _hamming_proxy(schedule: Schedule, other: Schedule) -> int:
    """Hamming distance proxy function for schedules."""
    num_buckets = max(len(schedule.machines), len(other.machines))
    distance = 0
    # should be same order
    for machine1, machine2 in zip(schedule.machines, other.machines):
        num_buckets = max(len(machine1.buckets), len(machine2.buckets))
        jobs1 = _helper(machine1)
        jobs2 = _helper(machine2)
        for job in jobs1:
            if job in jobs2:
                distance += abs(jobs1[job] - jobs2[job])
            else:
                distance += num_buckets

    return distance


def _helper(machine: Machine) -> dict[UUID, int]:
    return {
        job.uuid: idx
        for idx, bucket in enumerate(machine.buckets)
        for job in bucket.jobs
    }
