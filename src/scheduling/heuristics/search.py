"""Scatter search heuristic for scheduling problems."""

from functools import partial
from multiprocessing import Pool, cpu_count, current_process
import logging

from qiskit import QuantumCircuit

from src.provider import Accelerator

from .diversify import generate_new_solutions
from .improve import improve_solutions
from .initialize import initialize_population
from .select import (
    select_best_solution,
    select_elite_solutions,
    select_diverse_solutions,
)
from .types import Schedule


def scatter_search(
    circuits: list[QuantumCircuit],
    accelerators: list[Accelerator],
    num_iterations: int = 100,
    num_elite_solutions: int = 10,
    **kwargs,
) -> Schedule:
    """Scatter search heuristic for scheduling problems.

    Args:
        circuits (list[QuantumCircuit]): Batch of circuits to schedule.
        accelerators (list[Accelerator]): List of accelerators to schedule on.
        num_iterations (int, optional): Number of search iterations. Defaults to 100.
        num_elite_solutions (int, optional): Max number of solutions to keep each round.
        Defaults to 10.

    Returns:
        Schedule: The approximate best schedule found by the heuristic.
    """
    # TODO maybe decrease num_elite_solutions/diversificaiton over time? (similar to SA)
    num_cores = kwargs.get("num_cores", cpu_count())
    population = initialize_population(circuits, accelerators, **kwargs)
    kwargs["num_iterations"] = num_iterations // num_cores
    kwargs["num_elite_solutions"] = num_elite_solutions
    with Pool(processes=num_cores) as pool:
        work = partial(
            _task,
            accelerators=accelerators,
            **kwargs,
        )
        solutions = pool.map(work, [population for _ in range(num_cores)])

    return select_best_solution(solutions, accelerators)


def _task(
    population: list[Schedule],
    accelerators: list[Accelerator],
    num_iterations: int,
    num_elite_solutions: int,
    **kwargs,
) -> Schedule:
    best_solution = select_best_solution(population, accelerators)
    for _ in range(num_iterations):
        # Diversification
        new_solutions = generate_new_solutions(population)
        improved_population = improve_solutions(population, accelerators)

        # ensure we don't add duplicates
        population = _combine_solutions(population, new_solutions, improved_population)

        # Intensification
        elite_solutions = select_elite_solutions(
            population, num_elite_solutions, accelerators
        )
        diverse_solutions = select_diverse_solutions(population, num_elite_solutions)
        population = _combine_solutions(elite_solutions, diverse_solutions)

        # Update best solution
        current_best_solution = select_best_solution(population, accelerators)
        if current_best_solution.makespan < best_solution.makespan:
            best_solution = current_best_solution
            logging.debug(
                "Update best solution on process %s: New value %d ",
                current_process().name,
                current_best_solution.makespan,
            )
    return best_solution


def _combine_solutions(
    population: list[Schedule],
    *args: list[Schedule],
) -> list[Schedule]:
    """Combines solutions and removes duplicates."""
    combined_solution = []
    for solution in population + [schedule for other in args for schedule in other]:
        if solution not in combined_solution:
            combined_solution.append(solution)

    return combined_solution
