"""Diversify population by generation new local and global solutions."""

import numpy as np

from .types import Schedule, Bucket


def generate_new_solutions(population: list[Schedule], **kwargs) -> list[Schedule]:
    """Generates new solutions by local search and diversification.

    Local search swaps jobs or buckets within the same machine.
    Diversification swaps jobs between different machines.

    Args:
        population (list[Schedule]): List of schedules to generate new solutions from.
        num_solutions (int, optional): Controls the number of solutions to add. Defaults to 1.
            Not used yet.

    Returns:
        list[Schedule]: local and global candidates for the next generation.
    """

    local_candidates = _local_search(population)
    global_candidates = _diversify(population)
    return local_candidates + global_candidates


def _local_search(population: list[Schedule]) -> list[Schedule]:
    """Swaps jobs within the same machine."""
    # TODO: maybe add tabu list here?
    local_population = population.copy()
    for schedule in local_population:
        for machine in schedule.machines:
            if len(machine.buckets) == 0:
                continue
            swap_buckets = np.random.randint(1, dtype=bool)
            number_of_swaps = (
                np.random.randint(1, len(machine.buckets))
                if len(machine.buckets) > 1
                else 1
            )
            for _ in range(number_of_swaps):
                idx1, idx2 = np.random.choice(len(machine.buckets), 2)
                if swap_buckets:
                    machine.buckets[idx1], machine.buckets[idx2] = (
                        machine.buckets[idx2],
                        machine.buckets[idx1],
                    )
                else:
                    _swap_jobs(
                        machine.buckets[idx1], machine.buckets[idx2], machine.capacity
                    )

    return local_population


def _swap_jobs(
    bucket1: Bucket,
    bucket2: Bucket,
    machine1_capacity: int,
    machine2_capacity: int | None = None,
):
    candidates: list[tuple[int, int]] = []
    if machine2_capacity is None:
        machine2_capacity = machine1_capacity
    bucket1_capacity = sum(job.circuit.num_qubits for job in bucket1.jobs)
    bucket2_capacity = sum(job.circuit.num_qubits for job in bucket2.jobs)
    for idx1, job1 in enumerate(bucket1.jobs):
        for idx2, job2 in enumerate(bucket2.jobs):
            if (
                bucket1_capacity - job1.circuit.num_qubits + job2.circuit.num_qubits
                <= machine1_capacity
                and (
                    bucket2_capacity - job2.circuit.num_qubits + job1.circuit.num_qubits
                    <= machine2_capacity
                )
            ):
                candidates.append((idx1, idx2))
    if len(candidates) > 0:
        idx1, idx2 = candidates[np.random.choice(len(candidates))]
        bucket1.jobs[idx1], bucket2.jobs[idx2] = bucket2.jobs[idx2], bucket1.jobs[idx1]


def _diversify(population: list[Schedule]) -> list[Schedule]:
    """Swaps jobs between different machines."""
    local_population = population.copy()
    for schedule in population:
        number_of_swaps = np.random.randint(1, len(schedule.machines))
        for _ in range(number_of_swaps):
            idx1, idx2 = np.random.choice(len(schedule.machines), 2)
            machine1, machine2 = schedule.machines[idx1], schedule.machines[idx2]
            _swap_jobs(
                np.random.choice(machine1.buckets),
                np.random.choice(machine2.buckets),
                machine1.capacity,
                machine2.capacity,
            )
    return local_population
