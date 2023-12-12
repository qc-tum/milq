"""Helpers to generate MILP based schedules."""
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pulp
from qiskit import QuantumCircuit

from src.scheduling import JobHelper, JobResultInfo, LPInstance, PTimes, STimes


def set_up_base_lp(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    big_m: int,
    timesteps: list[int],
) -> LPInstance:
    """Sets up the base LP instance.

    Generates a base LP instance with the given jobs and accelerators.
    It contains all the default constraints and variables.
    Does not contain the constraints regarding the successor relationship.

    Args:
        base_jobs (list[QuantumCircuit]): The list of quantum cirucits (jobs).
        accelerators (dict[str, int]): The list of available accelerators (machines).
        big_m (int): Metavariable for the LP.
        timesteps (list[int]): Meta variable for the LP, big enough to cover largest makespan.

    Returns:
        LPInstance: The LP instance object.
    """
    # Set up input params
    jobs = ["0"] + [str(idx + 1) for idx, _ in enumerate(base_jobs)]
    job_capacities = {str(idx + 1): job.num_qubits for idx, job in enumerate(base_jobs)}
    job_capacities["0"] = 0
    machines = list(accelerators.keys())
    machine_capacities = accelerators

    # set up problem variables
    x_ik = pulp.LpVariable.dicts("x_ik", (jobs, machines), cat="Binary")
    z_ikt = pulp.LpVariable.dicts("z_ikt", (jobs, machines, timesteps), cat="Binary")

    c_j = pulp.LpVariable.dicts("c_j", (jobs), 0, cat="Continuous")
    s_j = pulp.LpVariable.dicts("s_j", (jobs), 0, cat="Continuous")
    c_max = pulp.LpVariable("makespan", 0, cat="Continuous")

    problem = pulp.LpProblem("Scheduling", pulp.LpMinimize)
    # set up problem constraints
    problem += pulp.lpSum(c_max)  # (obj)
    problem += c_j["0"] == 0  # (8)
    for job in jobs[1:]:
        problem += c_j[job] <= c_max  # (1)
        problem += pulp.lpSum(x_ik[job][machine] for machine in machines) == 1  # (3)
        problem += c_j[job] - s_j[job] + 1 == pulp.lpSum(  # (11)
            z_ikt[job][machine][timestep]
            for timestep in timesteps
            for machine in machines
        )
        for machine in machines:
            problem += (  # (12)
                pulp.lpSum(z_ikt[job][machine][timestep] for timestep in timesteps)
                <= x_ik[job][machine] * big_m
            )

        for timestep in timesteps:
            problem += (  # (13)
                pulp.lpSum(z_ikt[job][machine][timestep] for machine in machines)
                * timestep
                <= c_j[job]
            )
            problem += s_j[job] <= pulp.lpSum(  # (14)
                z_ikt[job][machine][timestep] for machine in machines
            ) * timestep + big_m * (
                1 - pulp.lpSum(z_ikt[job][machine][timestep] for machine in machines)
            )
    for timestep in timesteps:
        for machine in machines:
            problem += (  # (15)
                pulp.lpSum(
                    z_ikt[job][machine][timestep] * job_capacities[job]
                    for job in jobs[1:]
                )
                <= machine_capacities[machine]
            )
    return LPInstance(
        problem=problem,
        jobs=jobs,
        machines=machines,
        x_ik=x_ik,
        z_ikt=z_ikt,
        c_j=c_j,
        s_j=s_j,
        named_circuits=[JobHelper("0", None)]
        + [JobHelper(str(idx + 1), job) for idx, job in enumerate(base_jobs)],
    )


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
    _, jobs = _solve_lp(lp_instance)
    s_times = pulp.makeDict(
        [lp_instance.jobs, lp_instance.jobs, lp_instance.machines],
        setup_times,
        0,
    )
    return calculate_makespan(jobs, p_times, s_times), jobs


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
    _, jobs = _solve_lp(lp_instance)
    return calculate_makespan(jobs, p_times, s_times), jobs


def _solve_lp(lp_instance: LPInstance) -> tuple[float, list[JobResultInfo]]:
    """Solves a LP using gurobi and generates the results."""
    solver_list = pulp.listSolvers(onlyAvailable=True)
    gurobi = "GUROBI_CMD"
    if gurobi in solver_list:
        solver = pulp.getSolver(gurobi)
        lp_instance.problem.solve(solver)
    else:
        lp_instance.problem.solve()
    return _generate_results(lp_instance)


def _generate_results(lp_instance: LPInstance) -> tuple[float, list[JobResultInfo]]:
    assigned_jobs = {
        job.name: JobResultInfo(name=job.name, capacity=job.circuit.num_qubits)
        if job.circuit is not None
        else JobResultInfo(name=job.name)
        for job in lp_instance.named_circuits
    }
    for var in lp_instance.problem.variables():
        if var.name.startswith("x_") and var.varValue > 0.0:
            name = var.name.split("_")[2:]
            assigned_jobs[name[0]].machine = name[1]
        elif var.name.startswith("s_"):
            name = var.name.split("_")[2]
            assigned_jobs[name].start_time = float(var.varValue)
        elif var.name.startswith("c_"):
            name = var.name.split("_")[2]
            assigned_jobs[name[0]].completion_time = float(var.varValue)
    del assigned_jobs["0"]
    return lp_instance.problem.objective.value(), list(assigned_jobs.values())


def calculate_makespan(
    jobs: list[JobResultInfo],
    p_times: defaultdict[str, defaultdict[str, float]],
    s_times: defaultdict[str, defaultdict[str, defaultdict[str, float]]],
) -> float:
    """Calculates the actual makespan from the list of results.

    Executes the schedule with the corret p_ij and s_ij values.

    Args:
        jobs (list[JobResultInfo]): The list of job results.
        p_times (defaultdict[str, defaultdict[str, float]]): The correct  p_ij.
        s_times (defaultdict[str, defaultdict[str, defaultdict[str, float]]]): The correct s_ij.

    Returns:
        float: The makespan of the schedule.
    """
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
                last_completed = JobResultInfo("0", machine, 0.0, 0.0)
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
        return JobResultInfo("0", machine, 0.0, 0.0)

    completed_before = sorted(
        completed_before,
        key=lambda x: x.completion_time,
        reverse=True,
    )
    return completed_before[0]
