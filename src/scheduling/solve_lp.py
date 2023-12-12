import pulp
from .types import LPInstance, JobResultInfo


def solve_lp(lp_instance: LPInstance) -> tuple[float, list[JobResultInfo]]:
    """Solves a LP using gurobi and generates the results.

    Args:
        lp_instance (LPInstance): The input LP instance.

    Returns:
        tuple[float, list[JobResultInfo]]: The objective value and the list of jobs with their
            with their assigned machine and start and completion times.
    """
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
