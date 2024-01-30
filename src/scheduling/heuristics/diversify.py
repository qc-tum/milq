"""Diversify population by generation new local and global solutions."""

from .types import Schedule


def generate_new_solutions(
    population: list[Schedule], num_solutions: int = 1
) -> list[Schedule]:
    # TODO: Implement logic to generate new solutions

    local_candidates = _local_search(population)
    global_candidates = _diversify(population)
    return local_candidates + global_candidates + population


def _local_search(population: list[Schedule]) -> list[Schedule]:
    # TODO: generate new solutions by swapping jobs on one machine
    # TODO: maybe add tabu list here?
    return population


def _diversify(population: list[Schedule]) -> list[Schedule]:
    # TODO: generate new solutions by swapping jobs between machines
    return population
