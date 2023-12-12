"""Solves a LP using gurobi if available."""
import pulp

from .types import LPInstance


def solve_lp(lp_instance: LPInstance) -> LPInstance:
    """Solves a LP using gurobi.

    Args:
        lp_instance (LPInstance): The input LP instance.

    Returns:
        lp_instance (LPInstance): The LP instance with the solved problem object..
    """
    solver_list = pulp.listSolvers(onlyAvailable=True)
    gurobi = "GUROBI_CMD"
    if gurobi in solver_list:
        solver = pulp.getSolver(gurobi)
        lp_instance.problem.solve(solver)
    else:
        lp_instance.problem.solve()
    return lp_instance
