"""Scatter search heuristic for scheduling problems."""

from qiskit import QuantumCircuit

from src.provider import Accelerator

from .diversify import generate_new_solutions
from .initialize import initialize_population
from .select import select_best_solution, select_elite_solutions
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
    population = initialize_population(circuits, accelerators, **kwargs)
    best_solution = select_best_solution(population, accelerators)

    # Main loop
    for _ in range(num_iterations):
        # Diversification
        new_solutions = generate_new_solutions(population)
        population += new_solutions

        # Intensification
        elite_solutions = select_elite_solutions(
            population, num_elite_solutions, accelerators
        )
        population = elite_solutions

        # Update best solution
        current_best_solution = select_best_solution(population, accelerators)
        if current_best_solution.makespan < best_solution.makespan:
            best_solution = current_best_solution

    return best_solution
