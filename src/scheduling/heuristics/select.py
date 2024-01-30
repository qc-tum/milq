"""Evaluation and selection of solutions."""

from .types import Schedule


def select_elite_solutions(
    population: list[Schedule], num_solutions: int
) -> list[Schedule]:
    return population[:num_solutions]


def select_best_solution(population) -> Schedule:
    return select_elite_solutions(population, 1)[0]


def _evaluate_solution(schedule: Schedule) -> float:
    return schedule.makespan
