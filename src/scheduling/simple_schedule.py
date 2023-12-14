import numpy as np
import pulp

from src.common import CircuitJob, ScheduledJob
from src.provider import Accelerator

from .calculate_makespan import calculate_makespan
from .extract_schedule import extract_info_schedule, extract_executable_schedule
from .setup_lp import set_up_base_lp
from .solve_lp import solve_lp
from .types import JobResultInfo, LPInstance, PTimes, STimes


def generate_simple_lp(
    lp_instance: LPInstance,
    process_times: PTimes,
    setup_times: STimes,
) -> LPInstance:
    """"""
    p_times = pulp.makeDict(
        [lp_instance.jobs[1:], lp_instance.machines],
        process_times,
        0,
    )
    s_times = pulp.makeDict(
        [lp_instance.jobs[1:], lp_instance.machines],
        _get_simple_setup_times(setup_times),
        0,
    )

    for job in lp_instance.jobs[1:]:
        lp_instance.problem += lp_instance.c_j[job] >= lp_instance.s_j[  # (7)
            job
        ] + pulp.lpSum(
            lp_instance.x_ik[job][machine]
            * (p_times[job][machine] + s_times[job][machine])
            for machine in lp_instance.machines
        )
    return lp_instance


def generate_simple_schedule(
    lp_instance: LPInstance,
    process_times: PTimes,
    setup_times: STimes,
) -> tuple[float, list[JobResultInfo]]:
    """Generates a simple schedule for the given jobs and accelerators using a simple MILP.

    First generates the schedule using MILP  and then calculates the makespan
    by executing the schedule with the correct p_ij and s_ij values.
    The MILP uses setup times depending on the maximum over all possible values.

    Args:
        lp_instance (LPInstance): The base LP instance.
        process_times (PTimes): The process times for each job on each machine.
        setup_times (STimes): The setup times for each job on each machine.

    Returns:
        tuple[float, list[JobResultInfo]]: List of jobs with their assigned machine and
            start and completion times.
    """
    lp_instance = generate_simple_lp(lp_instance, process_times, setup_times)
    _, jobs = extract_info_schedule(solve_lp(lp_instance))
    return calculate_makespan(lp_instance, jobs, process_times, setup_times), jobs


def _get_simple_setup_times(
    setup_times: STimes,
) -> list[list[float]]:
    """Overestimates the actual setup times for the simple LP."""
    new_times = [
        list(
            np.max(
                times[[t not in [0, idx] for t, _ in enumerate(times)]].transpose(),
                axis=1,
            )
        )
        for idx, times in enumerate(np.array(setup_times))
    ]
    # remove job 0
    del new_times[0]
    for times in new_times:
        del times[0]
    return new_times


def generate_simple_schedule_provider(
    jobs: list[CircuitJob],
    accelerators: list[Accelerator],
    big_m: int = 1000,
    t_max: int = 2**7,
) -> list[ScheduledJob]:
    """Calclulate a schedule for the given jobs and accelerators based on the simple MILP.

    The simple MILP includes the machine dependent setup times.
    Args:
        jobs (list[CircuitJob]): The list of jobs to run.
        accelerators (list[Accelerator]): The list of available accelerators.
        big_m (int, optional): M hepler for LP. Defaults to 1000.
        t_max (int, optional): Max number of Timesteps. Defaults to 2**7.
    Returns:
        list[ScheduledJob]: A list of Jobs scheduled to accelerators.
    """
    lp_instance = set_up_base_lp(jobs, accelerators, big_m, list(range(t_max)))
    # (4) - (7), (9)
    lp_instance = generate_simple_lp(
        lp_instance,
        _get_processing_times(jobs, accelerators),
        _get_simple_setup_times_provider(jobs, accelerators),
    )

    return extract_executable_schedule(solve_lp(lp_instance), jobs, accelerators)


def _get_simple_setup_times_provider(
    base_jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[list[float]]:
    return [
        [qpu.compute_setup_time(job_i.circuit, circuit_to=None) for qpu in accelerators]
        for job_i in base_jobs
        if job_i.circuit is not None
    ]


def _get_processing_times(
    base_jobs: list[CircuitJob],
    accelerators: list[Accelerator],
) -> list[list[float]]:
    return [
        [qpu.compute_processing_time(job.circuit) for qpu in accelerators]
        for job in base_jobs
        if job.circuit is not None
    ]
