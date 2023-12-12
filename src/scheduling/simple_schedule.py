import numpy as np
import pulp

from .calculate_makespan import calculate_makespan
from .solve_lp import solve_lp
from .types import JobResultInfo, LPInstance, PTimes, STimes


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
    _, jobs = solve_lp(lp_instance)
    s_times = pulp.makeDict(
        [lp_instance.jobs, lp_instance.jobs, lp_instance.machines],
        setup_times,
        0,
    )
    return calculate_makespan(jobs, p_times, s_times), jobs


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
