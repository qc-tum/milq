import json

import pulp
import numpy as np

np.random.seed(42)


def get_process_time(job_i, machine_k) -> int:  # change to float
    if job_i == 0:
        return 0
    return job_i + np.random.randint(-2, 3) + machine_k


def get_setup_time(job_i, job_j_, machine_k) -> int:  # change to float
    if job_j_ == 0:
        return 0
    return (job_i + job_j_) / 2 + np.random.randint(-2, 3) + machine_k


# Meta Variables
BIG_M = 1000000
TIMESTEPS = 2**8
solver_list = pulp.listSolvers(onlyAvailable=True)
print(solver_list)

# Inputs
jobs = ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I"]
job_capcities = {
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
timesteps = list(range(TIMESTEPS))  # Big enough to cover all possible timesteps
# params
processing_times = [
    [
        get_process_time(job_capcities[job], machine_capacities[machine])
        for machine in machines
    ]
    for job in jobs
]
setup_times = [
    [
        [
            get_setup_time(
                job_capcities[job_i], job_capcities[job_j], machine_capacities[machine]
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
    "x_ik", (jobs, machines), cat="Binary"
)  # x: Job i is assigned to machine k
y_ijk = pulp.LpVariable.dicts(
    "y_ijk", (jobs, jobs, machines), cat="Binary"
)  # y: Job i is assigned before job j
z_ikt = pulp.LpVariable.dicts(
    "z_ikt", (jobs, machines, timesteps), cat="Binary"
)  # z: Job i is assigned to machine k at timestep t

c_max = pulp.LpVariable("makespan", 0, cat="Continuous")  # c: makespan
c_j = pulp.LpVariable.dicts("c_j", (jobs), 0, cat="Continuous")  # c: completion time
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
    for timestep in timesteps:
        problem += (
            pulp.lpSum(z_ikt[job][machine][timestep] for machine in machines) <= 1
        )
    problem += pulp.lpSum(x_ik[job][machine] for machine in machines) == 1

# (4) - (6) TODO jobs can have multiple predecessors and successors


# completion time for each job (7)
for job in jobs[1:]:  # maybe needs t_i
    problem += c_j[job] == s_j[job] + pulp.lpSum(
        x_ik[job][machine] * p_times[job][machine] for machine in machines
    ) + pulp.lpSum(
        y_ijk[job][job_j][machine] * s_times[job][job_j][machine]
        for machine in machines
        for job_j in jobs
    )
# completion time for dummy job (8)
problem += c_j["0"] == 0

# order constraint (9) # TODO can we change this to allow for jobs to be seperated in time ?
for job in jobs[1:]:
    for job_j in jobs:
        problem += (
            c_j[job_j]
            + pulp.lpSum(y_ijk[job_j][job][machine] - 1 for machine in machines) * BIG_M
            <= s_j[job]
        )

# (10) we don't need this constraint
# job is combleted (11) TODO can we relax this?
for job in jobs[1:]:
    problem += c_j[job] - s_j[job] == pulp.lpSum(
        z_ikt[job][machine][timestep] for machine in machines for timestep in timesteps
    )

# z fits machine assignment (12)
# TODO do we need this / or can we loosen this -> how to encode the setup times
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
                z_ikt[job][machine][timestep] * job_capcities[job] for job in jobs
            )
            <= machine_capacities[machine]
        )
# (16) - (20) already encoded in vars
problem.writeLP("scheduling.lp")
if len(solver_list) == 2:
    solver = pulp.getSolver("GUROBI_CMD")
    problem.solve(solver)
else:
    problem.solve()
print("Status:", pulp.LpStatus[problem.status])

with open("scheduling.json", "w+", encoding="utf-8") as f:
    json.dump(
        {
            "params": {
                "jobs": jobs,
                "machines": machines,
                "job_capcities": job_capcities,
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
