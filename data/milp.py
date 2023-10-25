import json

import pulp
import numpy as np

np.random.seed(42)


def get_process_time(job_i, machine_k) -> int:
    if job_i == 0:
        return 0
    return job_i + np.random.randint(-2, 3) + machine_k


def get_setup_time(job_i, job_j_, machine_k) -> int:  # change to float
    if job_j_ == 0:
        return 0
    return (job_i + job_j_) // 2 + np.random.randint(-2, 3) + machine_k


# Meta Variables
BIG_M = 1000
TIMESTEPS = 2**6

# Inputs
jobs = ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I"]
job_capacities = {
    "0": 0,  # dummy job
    "A": 5,
    "B": 5,
    "C": 5,
    "D": 5,
    "E": 3,
    "F": 2,
    "G": 2,
    "H": 2,
    "I": 2,
}
machines = ["QUITO", "BELEM"]
machine_capacities = {"QUITO": 5, "BELEM": 5}
timesteps = list(range(TIMESTEPS + 1))  # Big enough to cover all possible timesteps


def generate_lp() -> pulp.LpProblem:
    # params
    processing_times = [
        [
            get_process_time(job_capacities[job], machine_capacities[machine])
            for machine in machines
        ]
        for job in jobs
    ]
    setup_times = [
        [
            [
                50  # BIG!
                if job_i in [job_j, "0"]
                else get_setup_time(
                    job_capacities[job_i],
                    job_capacities[job_j],
                    machine_capacities[machine],
                )
                for machine in machines
            ]
            for job_i in jobs
        ]
        for job_j in jobs
    ]
    p_times = pulp.makeDict([jobs, machines], processing_times, 0)
    s_times = pulp.makeDict([jobs, jobs, machines], setup_times, 0)

    # decision variables
    x_ik = pulp.LpVariable.dicts(
        "x_ik", (jobs, machines), cat=pulp.const.LpBinary
    )  # x: Job i is assigned to machine k
    y_ijk = pulp.LpVariable.dicts(
        "y_ijk", (jobs, jobs, machines), cat=pulp.const.LpBinary
    )  # y: Job i is assigned before job j
    z_ikt = pulp.LpVariable.dicts(
        "z_ikt", (jobs, machines, timesteps), cat=pulp.const.LpBinary
    )  # z: Job i is assigned to machine k at timestep t

    a_ij = pulp.LpVariable.dicts(
        "a_ij", (jobs, jobs), cat=pulp.const.LpBinary
    )  # a: Job i ends before job j starts
    b_ij = pulp.LpVariable.dicts(
        "b_ij", (jobs, jobs), cat=pulp.const.LpBinary
    )  # b: Job i ends before job j ends
    d_ijk = pulp.LpVariable.dicts(
        "d_ijk", (jobs, jobs, machines), cat=pulp.const.LpBinary
    )  # d: Job i and  j run on the same machine

    e_ijlk = pulp.LpVariable.dicts(
        "e_ijlk", (jobs, jobs, jobs, machines), cat=pulp.const.LpBinary
    )

    c_max = pulp.LpVariable("makespan", 0, cat="Continuous")  # c: makespan
    c_j = pulp.LpVariable.dicts(
        "c_j", (jobs), 0, cat="Continuous"
    )  # c: completion time
    s_j = pulp.LpVariable.dicts("s_j", (jobs), 0, cat="Continuous")  # s: start time

    # Problem
    problem = pulp.LpProblem("Scheduling", pulp.LpMinimize)
    # Objective function
    problem += pulp.lpSum(c_max)

    # Constraints

    # makespan constraint (1)
    for job in jobs[1:]:
        problem += c_j[job] <= c_max

    # job assignment constraint (3)
    for job in jobs[1:]:
        problem += pulp.lpSum(x_ik[job][machine] for machine in machines) == 1

    # replaced (4) - (6):  jobs can have multiple predecessors and successors
    for job in jobs[1:]:
        problem += (
            pulp.lpSum(
                y_ijk[job_j][job][machine] for machine in machines for job_j in jobs
            )
            >= 1  # each job has a predecessor
        )
    # if the job has a predecessor or successor
    # on a machine it also has to run on this machine
    for job in jobs[1:]:
        for machine in machines:
            problem += (  # predecessor
                x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job_j][job][machine] for job_j in jobs) / BIG_M
            )
            problem += (  # successor
                x_ik[job][machine]
                >= pulp.lpSum(y_ijk[job][job_j][machine] for job_j in jobs) / BIG_M
            )

    # only if job runs at t=0 it can have predecessor 0
    for job in jobs[1:]:
        for machine in machines:
            problem += z_ikt[job][machine][0] == y_ijk["0"][job][machine]

    # completion time for each job (7)
    for job in jobs[1:]:
        problem += c_j[job] >= s_j[job] + pulp.lpSum(
            x_ik[job][machine] * p_times[job][machine] for machine in machines
        ) + pulp.lpSum(
            y_ijk[job_j][job][machine] * s_times[job_j][job][machine]
            for machine in machines
            for job_j in jobs
        )

    # completion time for dummy job (8)
    problem += c_j["0"] == 0

    # order constraint (9)
    for job in jobs[1:]:
        for job_j in jobs:
            problem += (
                c_j[job_j]
                + (pulp.lpSum(y_ijk[job_j][job][machine] for machine in machines) - 1)
                * BIG_M
                <= s_j[job]
            )

    # (10) we don't need this constraint
    # job is combleted (11)
    for job in jobs[1:]:
        problem += c_j[job] - s_j[job] + 1 == pulp.lpSum(
            z_ikt[job][machine][timestep]
            for timestep in timesteps
            for machine in machines
        )

    # z fits machine assignment (12)
    for job in jobs[1:]:
        for machine in machines:
            problem += (
                pulp.lpSum(z_ikt[job][machine][timestep] for timestep in timesteps)
                <= x_ik[job][machine] * BIG_M
            )

    # z fits time assignment (13) - (14)
    for job in jobs[1:]:
        for timestep in timesteps:
            problem += (
                pulp.lpSum(z_ikt[job][machine][timestep] for machine in machines)
                * timestep
                <= c_j[job]
            )

    for job in jobs[1:]:
        for timestep in timesteps:
            problem += s_j[job] <= pulp.lpSum(
                z_ikt[job][machine][timestep] for machine in machines
            ) * timestep + BIG_M * (
                1 - pulp.lpSum(z_ikt[job][machine][timestep] for machine in machines)
            )

    # capacity constraint (15)
    for timestep in timesteps:
        for machine in machines:
            problem += (
                pulp.lpSum(
                    z_ikt[job][machine][timestep] * job_capacities[job]
                    for job in jobs[1:]
                )
                <= machine_capacities[machine]
            )
    # These constraints encode the specific behavior we want to achieve with y_ijk:
    # y_ijk == 1 <=>
    # c_i < c_j and not exist l in J: c_i < c_l < s_j and i,j,l run on the same machine
    for job in jobs[1:]:
        for job_j in jobs[1:]:
            if job == job_j:
                problem += a_ij[job][job_j] == 0
                problem += b_ij[job][job_j] == 0
                continue
            problem += a_ij[job][job_j] >= (s_j[job_j] - c_j[job]) / BIG_M
            problem += b_ij[job][job_j] >= (c_j[job_j] - c_j[job]) / BIG_M
            for machine in machines:
                problem += (
                    d_ijk[job][job_j][machine]
                    >= x_ik[job][machine] + x_ik[job_j][machine] - 1
                )
                for job_l in jobs[1:]:
                    problem += (
                        e_ijlk[job][job_j][job_l][machine]
                        >= b_ij[job][job_l]
                        + a_ij[job_l][job_j]
                        + d_ijk[job][job_j][machine]
                        + d_ijk[job][job_l][machine]
                        - 3
                    )

    for job in jobs[1:]:
        for job_j in jobs[1:]:
            for machine in machines:
                problem += (
                    y_ijk[job][job_j][machine]
                    >= a_ij[job][job_j]
                    + (
                        1
                        - pulp.lpSum(
                            e_ijlk[job][job_j][job_l][machine] for job_l in jobs[1:]
                        )
                    )
                    + d_ijk[job][job_j][machine]
                    - 2
                )

    # (16) - (20) already encoded in vars
    problem.writeLP("scheduling.lp")
    return problem


def solve_and_print_lp(filename: str, problem: pulp.LpProblem) -> None:
    solver_list = pulp.listSolvers(onlyAvailable=True)
    if len(solver_list) == 2:
        solver = pulp.getSolver("GUROBI_CMD")
        problem.solve(solver)
    else:
        problem.solve()
    print("Status:", pulp.LpStatus[problem.status])

    with open(filename, "w+", encoding="utf-8") as f:
        json.dump(
            {
                "params": {
                    "jobs": jobs,
                    "machines": machines,
                    "job_capcities": job_capacities,
                    "machine_capacities": machine_capacities,
                    "timesteps": timesteps,
                    "processing_times": processing_times,
                    "setup_times": s_times,
                },
                "status": pulp.LpStatus[problem.status],
                "objective": pulp.value(problem.objective),
                "variables": {
                    var.name: var.varValue
                    for var in problem.variables()
                    if var.varValue > 0
                },
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    problem = generate_lp()
    solve_and_print_lp("scheduling.json", problem)
