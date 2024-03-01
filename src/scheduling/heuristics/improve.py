"""Active solution improvement heuristics."""

import numpy as np

from src.provider import Accelerator

from src.scheduling.common.types import Schedule
from .select import evaluate_solution


def improve_solutions(population: list[Schedule]) -> list[Schedule]:
    """Improve solutions by also allowing invalid solutions.

    Improves the solution by taking the worst machine and removing a bucket from it.

    Args:
        schedules (list[Schedule]): The list of schedules to improve.

    Returns:
        list[Schedule]: An improved list of schedules, protentially with invalid solutions.
    """
    # Find the machine with the longest makespan
    for schedule in population:
        schedule = evaluate_solution(schedule)
        worst_machine = max(schedule.machines, key=lambda m: m.makespan)
        bucket = worst_machine.buckets.pop(
            np.random.randint(len(worst_machine.buckets))
        )
        # Remove bucket from worst machine
        for job in bucket.jobs:
            for machine in sorted(schedule.machines, key=lambda m: m.makespan):
                if machine == worst_machine:
                    continue
                # Find the bucket with the biggest remaining capacity
                smallest_bucket = min(
                    machine.buckets,
                    key=lambda b: sum(job.num_qubits for job in b.jobs),
                )
                smallest_bucket.jobs.append(job)

    return population
