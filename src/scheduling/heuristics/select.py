"""Evaluation and selection of solutions."""

from uuid import UUID
import logging

from src.scheduling.common import Schedule, Machine, is_feasible, evaluate_solution


def select_elite_solutions(
    population: list[Schedule], num_solutions: int
) -> list[Schedule]:
    """Selects the #num_solutions best solutions from the population by makespan.

    Args:
        population (list[Schedule]): List of schedules to select from.
        num_solutions (int): Number of solutions to select.

    Raises:
        ValueError: If the population is empty.

    Returns:
        list[Schedule]: The #num_solutions best schedules with lowest makespan.
    """
    logging.debug("Selecting elite solutions...")
    if len(population) == 0:
        raise ValueError("Population must not be empty.")

    population = [evaluate_solution(schedule) for schedule in population]
    logging.debug("Evaluation done.")
    return sorted(population, key=lambda x: x.makespan)[:num_solutions]


def select_best_solution(population: list[Schedule]) -> Schedule:
    """Selects the best solution from the population by makespan.

    Args:
        population (list[Schedule]): List of schedules to select from.

    Returns:
        Schedule: The schedule with the lowest makespan.
    """
    logging.debug("Selecting best solution.")
    for solution in select_elite_solutions(population, len(population)):
        if is_feasible(solution):
            logging.debug("Feasible solution found.")
            return solution
    logging.debug("No feasible solution found.")
    return population[-1]


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
