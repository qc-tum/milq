"""__summary__"""
from collections import defaultdict
from copy import deepcopy

import pulp

from .types import JobResultInfo, LPInstance, PTimes, STimes


def calculate_makespan(
    lp_instance: LPInstance,
    jobs: list[JobResultInfo],
    process_times: PTimes,
    setup_times: STimes,
) -> float:
    """Calculates the actual makespan from the list of results.

    Executes the schedule with the corret p_ij and s_ij values.

    Args:
        lp_instance (LPInstance): The base LP instance.
        jobs (list[JobResultInfo]): The list of job results.
        process_times (PTimes): The correct  p_ij.
        setup_times (STimes) The correct s_ij.

    Returns:
        float: The makespan of the schedule.
    """
    s_times = pulp.makeDict(
        [lp_instance.jobs, lp_instance.jobs, lp_instance.machines],
        setup_times,
        0,
    )
    p_times = pulp.makeDict(
        [lp_instance.jobs[1:], lp_instance.machines],
        process_times,
        0,
    )

    assigned_machines: defaultdict[str, list[JobResultInfo]] = defaultdict(list)
    for job in jobs:
        assigned_machines[job.machine].append(job)
    makespans = []
    for machine, assigned_jobs in assigned_machines.items():
        assigned_jobs_copy = deepcopy(assigned_jobs)
        for job in sorted(assigned_jobs, key=lambda x: x.start_time):
            # Find the last predecessor that is completed before the job starts
            # this can technically change the correct predecessor to a wrong one
            # because completion times are updated in the loop
            # I'm not sure if copying before the loop corrects this
            last_completed = _find_last_completed(job.name, assigned_jobs_copy, machine)

            if job.start_time == 0.0:
                last_completed = JobResultInfo("0", machine, 0.0, 0.0, 0)
            job.start_time = next(
                (
                    j.completion_time
                    for j in assigned_jobs
                    if last_completed.name == j.name
                ),
                0.0,
            )
            # calculate p_j + s_ij
            completion_time = (  # check if this order is correct
                job.start_time
                + p_times[job.name][machine]
                + s_times[last_completed.name][job.name][machine]
            )
            job.completion_time = completion_time
        makespans.append(max(job.completion_time for job in assigned_jobs))

    return max(makespans)


def _find_last_completed(
    job_name: str, jobs: list[JobResultInfo], machine: str
) -> JobResultInfo:
    """Finds the last completed job before the given job from the original schedule."""
    original_starttime = next(
        (j.start_time for j in jobs if job_name == j.name),
        0,
    )
    completed_before = [j for j in jobs if j.completion_time <= original_starttime]
    if len(completed_before) == 0:
        return JobResultInfo("0", machine, 0.0, 0.0, 0)

    completed_before = sorted(
        completed_before,
        key=lambda x: x.completion_time,
        reverse=True,
    )
    return completed_before[0]


# TODO Simplify?
def calculate_bin_makespan(
    jobs: list[JobResultInfo],
    process_times: PTimes,
    setup_times: STimes,
    accelerators: dict[str, int],
) -> float:
    """Calculates the actual makespan from the list of jobs.
    By executing the schedule with the corret p_ij and s_ij values.
    TODO see generate_bin_info_schedule
    """
    lp_jobs = ["0"] + [job.name for job in jobs]  # TODO
    machines = list(accelerators.keys())
    p_times = pulp.makeDict(
        [lp_jobs[1:], machines],
        process_times,
        0,
    )
    s_times = pulp.makeDict(
        [lp_jobs, lp_jobs, machines],
        setup_times,
        0,
    )

    assigned_machines: defaultdict[str, list[JobResultInfo]] = defaultdict(list)
    for job in jobs:
        assigned_machines[job.machine].append(job)
    makespans = []
    for machine, assigned_jobs in assigned_machines.items():
        for job in sorted(assigned_jobs, key=lambda x: x.start_time):
            # Find the last predecessor that is completed before the job starts
            # this can technically change the correct predecessor to a wrong one
            # because completion times are updated in the loop
            # I'm not sure if copying before the loop corrects this
            last_completed = max(
                (job for job in assigned_jobs), key=lambda x: x.completion_time
            )
            if job.start_time == 0.0:
                last_completed = JobResultInfo("0", machine, 0.0, 0.0, 0)
            job.start_time = last_completed.completion_time
            # calculate p_j + s_ij
            completion_time = (  # check if this order is correct
                last_completed.completion_time
                + p_times[job.name][machine]
                + s_times[last_completed.name][job.name][machine]
            )
            job.completion_time = completion_time
        makespans.append(max(job.completion_time for job in assigned_jobs))

    return max(makespans)
