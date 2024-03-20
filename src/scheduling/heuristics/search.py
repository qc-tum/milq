"""Scatter search heuristic for scheduling problems."""

from functools import partial
from multiprocessing import Pool, cpu_count, current_process
import logging

from qiskit import QuantumCircuit
import tqdm

from src.common import UserCircuit
from src.provider import Accelerator

from src.scheduling.common.types import Schedule
from .diversify import generate_new_solutions
from .improve import improve_solutions
from .initialize import initialize_population
from .select import (
    select_best_solution,
    select_elite_solutions,
    select_diverse_solutions,
)


def scatter_search(
    circuits: list[QuantumCircuit | UserCircuit],
    accelerators: list[Accelerator],
    num_iterations: int = 100,
    num_elite_solutions: int = 10,
    **kwargs,
) -> Schedule:
    """Scatter search heuristic for scheduling problems.

    Args:
        circuits (list[QuantumCircuit | UserCircuit]): Batch of circuits to schedule.
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
    logging.info("Starting scatter search with %d cores", num_cores)
    if num_cores == 1:
        return _task(population, **kwargs)
    with Pool(processes=num_cores) as pool:
        work = partial(
            _task,
            **kwargs,
        )
        solutions = pool.map(work, [population for _ in range(num_cores)])

    return select_best_solution(solutions)


def _task(
    population: list[Schedule],
    num_iterations: int,
    num_elite_solutions: int,
    **kwargs,
) -> Schedule:
    logging.info("Starting new task on process %s", current_process().name)
    best_solution = select_best_solution(population)
    for idx in range(num_iterations):
        logging.info("Starting iteration %d on process %s", idx, current_process().name)
        # Diversification
        new_solutions = generate_new_solutions(population)
        improved_population = improve_solutions(population)

        # ensure we don't add duplicates
        population = _combine_solutions(population, improved_population, new_solutions)

        # Intensification
        elite_solutions = select_elite_solutions(population, num_elite_solutions)
        diverse_solutions = select_diverse_solutions(population, num_elite_solutions)
        population = _combine_solutions(
            elite_solutions,
            diverse_solutions,
            new_solutions,
        )

        # Update best solution
        current_best_solution = select_best_solution(population)
        if current_best_solution.makespan < best_solution.makespan:
            best_solution = current_best_solution
            logging.info(
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
