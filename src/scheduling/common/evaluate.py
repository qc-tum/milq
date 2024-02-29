"""Evaluation and selection of solutions."""

import logging

from src.provider import Accelerator
from src.scheduling.common.types import Schedule, Bucket, MakespanInfo


def evaluate_solution(schedule: Schedule, accelerators: list[Accelerator]) -> Schedule:
    """Calculates and updates the makespan of a schedule.

    Args:
        schedule (Schedule): A schedule to evaluate.
        accelerators (list[Accelerator]): The list of accelerators to schedule on.

    Returns:
        Schedule: The schedule with updated makespan and machine makespans.
    """
    logging.debug("Evaluating makespan...")
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
