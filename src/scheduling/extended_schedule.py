import pulp

from .calculate_makespan import calculate_makespan
from .solve_lp import solve_lp
from .types import JobResultInfo, LPInstance, PTimes, STimes


def generate_extended_schedule(
    lp_instance: LPInstance,
    process_times: PTimes,
    setup_times: STimes,
    big_m: int = 1000,
) -> tuple[float, list[JobResultInfo]]:
    """Generates the extended schedule for the given jobs and accelerators using a complex MILP.

    First generates the schedule using MILP  and then calculates the makespan
    by executing the schedule with the correct p_ij and s_ij values.

    Args:
        lp_instance (LPInstance): The base LP instance.
        process_times (PTimes): The process times for each job on each machine.
        setup_times (STimes): The setup times for each job on each machine.
        big_m (int, optional): Metavariable for the LP. Defaults to 1000.

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
        [lp_instance.jobs, lp_instance.jobs, lp_instance.machines],
        setup_times,
        0,
    )
    # decision variables
    y_ijk = pulp.LpVariable.dicts(
        "y_ijk",
        (lp_instance.jobs, lp_instance.jobs, lp_instance.machines),
        cat="Binary",
    )
    a_ij = pulp.LpVariable.dicts(
        "a_ij", (lp_instance.jobs, lp_instance.jobs), cat="Binary"
    )  # a: Job i ends before job j starts
    b_ij = pulp.LpVariable.dicts(
        "b_ij", (lp_instance.jobs, lp_instance.jobs), cat="Binary"
    )  # b: Job i ends before job j ends
    d_ijk = pulp.LpVariable.dicts(
        "d_ijk",
        (lp_instance.jobs, lp_instance.jobs, lp_instance.machines),
        cat="Binary",
    )  # d: Job i and  j run on the same machine
    e_ijlk = pulp.LpVariable.dicts(
        "e_ijlk",
        (lp_instance.jobs, lp_instance.jobs, lp_instance.jobs, lp_instance.machines),
        cat="Binary",
    )

    for job in lp_instance.jobs[1:]:
        lp_instance.problem += (  # (4)
            pulp.lpSum(
                y_ijk[job_j][job][machine]
                for machine in lp_instance.machines
                for job_j in lp_instance.jobs
            )
            >= 1  # each job has a predecessor
        )
        lp_instance.problem += lp_instance.c_j[job] >= lp_instance.s_j[  # (7)
            job
        ] + pulp.lpSum(
            lp_instance.x_ik[job][machine] * p_times[job][machine]
            for machine in lp_instance.machines
        ) + pulp.lpSum(
            y_ijk[job_j][job][machine] * s_times[job_j][job][machine]
            for machine in lp_instance.machines
            for job_j in lp_instance.jobs
        )
        for machine in lp_instance.machines:
            lp_instance.problem += (  # predecessor (6)
                lp_instance.x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job_j][job][machine] for job_j in lp_instance.jobs)
                / big_m
            )
            lp_instance.problem += (  # successor
                lp_instance.x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job][job_j][machine] for job_j in lp_instance.jobs)
                / big_m
            )
            lp_instance.problem += (  # (5)
                lp_instance.z_ikt[job][machine][0] == y_ijk["0"][job][machine]
            )
        for job_j in lp_instance.jobs:
            lp_instance.problem += (
                lp_instance.c_j[job_j]
                + (
                    pulp.lpSum(
                        y_ijk[job_j][job][machine] for machine in lp_instance.machines
                    )
                    - 1
                )
                * big_m
                <= lp_instance.s_j[job]
            )

    # Extended constraints
    for job in lp_instance.jobs[1:]:
        for job_j in lp_instance.jobs[1:]:
            if job == job_j:
                lp_instance.problem += a_ij[job][job_j] == 0
                lp_instance.problem += b_ij[job][job_j] == 0
                continue
            lp_instance.problem += (
                a_ij[job][job_j]
                >= (lp_instance.s_j[job_j] - lp_instance.c_j[job]) / big_m
            )
            lp_instance.problem += (
                b_ij[job][job_j]
                >= (lp_instance.c_j[job_j] - lp_instance.c_j[job]) / big_m
            )
            for machine in lp_instance.machines:
                lp_instance.problem += (
                    d_ijk[job][job_j][machine]
                    >= lp_instance.x_ik[job][machine]
                    + lp_instance.x_ik[job_j][machine]
                    - 1
                )
                for job_l in lp_instance.jobs[1:]:
                    lp_instance.problem += (
                        e_ijlk[job][job_j][job_l][machine]
                        >= b_ij[job][job_l]
                        + a_ij[job_l][job_j]
                        + d_ijk[job][job_j][machine]
                        + d_ijk[job][job_l][machine]
                        - 3
                    )

    for job in lp_instance.jobs[1:]:
        for job_j in lp_instance.jobs[1:]:
            for machine in lp_instance.machines:
                lp_instance.problem += (
                    y_ijk[job][job_j][machine]
                    >= a_ij[job][job_j]
                    + (
                        pulp.lpSum(
                            e_ijlk[job][job_j][job_l][machine]
                            for job_l in lp_instance.jobs[1:]
                        )
                        / big_m
                    )
                    + d_ijk[job][job_j][machine]
                    - 2
                )
    _, jobs = solve_lp(lp_instance)
    return calculate_makespan(jobs, p_times, s_times), jobs