"""Scatter search heuristic for scheduling problems."""


from src.common import CircuitJob

from .diversify import generate_new_solutions
from .initialize import initialize_population
from .select import select_best_solution, select_elite_solutions
from .types import Schedule


def scatter_search(
    circuits: list[CircuitJob],
    accelerator_capacities: dict[str, int],
    num_iterations: int = 100,
    num_elite_solutions: int = 10,
) -> Schedule:
    # TODO find good default values for parameters
    # TODO maybe decrease num_elite_solutions/diversificaiton over time? (similar to SA)
    population = initialize_population(circuits, accelerator_capacities)
    best_solution = select_best_solution(population)

    # Main loop
    for _ in range(num_iterations):
        # Diversification
        new_solutions = generate_new_solutions(population)
        population += new_solutions

        # Intensification
        elite_solutions = select_elite_solutions(population, num_elite_solutions)
        population = elite_solutions

        # Update best solution
        current_best_solution = select_best_solution(population)
        if current_best_solution.makespan < best_solution.makespan:
            best_solution = current_best_solution

    return best_solution
