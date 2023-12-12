"""Module for setting up the base LP instance."""
from functools import singledispatch
from typing import Any, Iterable

from qiskit import QuantumCircuit
import pulp

from src.common import CircuitJob
from src.provider import Accelerator
from .types import LPInstance, JobHelper


@singledispatch
def set_up_base_lp(
    base_jobs: list[Any], accelerators: Iterable[Any], big_m: int, timesteps: list[int]
) -> LPInstance:
    """Sets up the base LP instance base method. This method is not implemented!

    Generates a base LP instance with the given jobs and accelerators.
    It contains all the default constraints and variables.
    Does not contain the constraints regarding the successor relationship.

    Args:
        base_jobs (list[Any]): The list of quantum cirucits (jobs).
        accelerators (Iterable[Any]): The list of available accelerators (machines).
        big_m (int): Metavariable for the LP.
        timesteps (list[int]): Meta variable for the LP, big enough to cover largest makespan.

    Raises:
        NotImplementedError: If the method is not implemented for the given types.
    """
    raise NotImplementedError


@set_up_base_lp.register
def set_up_base_lp(
    base_jobs: list[CircuitJob],
    accelerators: list[Accelerator],
    big_m: int,
    timesteps: list[int],
) -> LPInstance:
    """Sets up the base LP instance for use in the provider.

    Generates a base LP instance with the given jobs and accelerators.
    It contains all the default constraints and variables.
    Does not contain the constraints regarding the successor relationship.

    Args:
        base_jobs (list[CircuitJob]): The list of quantum cirucits (jobs).
        accelerators (list[Accelerator]): The list of available accelerators (machines).
        big_m (int): Metavariable for the LP.
        timesteps (list[int]): Meta variable for the LP, big enough to cover largest makespan.

    Returns:
        LPInstance: The LP instance object.
    """
    # Set up input params
    job_capacities = {
        str(job.uuid): job.circuit.num_qubits
        for job in base_jobs
        if job.circuit is not None
    }
    job_capacities["0"] = 0
    machine_capacities = {str(qpu.uuid): qpu.qubits for qpu in accelerators}

    return _set_up_base_lp(job_capacities, machine_capacities, timesteps, big_m)


@set_up_base_lp.register
def set_up_base_lp(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    big_m: int,
    timesteps: list[int],
) -> LPInstance:
    """Sets up the base LP instance for use outside of provider.

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
    job_capacities = {str(idx + 1): job.num_qubits for idx, job in enumerate(base_jobs)}
    job_capacities["0"] = 0

    machine_capacities = accelerators

    lp_instance = _set_up_base_lp(job_capacities, machine_capacities, timesteps, big_m)
    lp_instance.named_circuits = [JobHelper("0", None)] + [
        JobHelper(str(idx + 1), job) for idx, job in enumerate(base_jobs)
    ]
    return lp_instance


def _set_up_base_lp(
    job_capacities: dict[str, int],
    machine_capacities: dict[str, int],
    timesteps: list[int],
    big_m: int,
) -> LPInstance:
    jobs = list(job_capacities.keys())
    machines = list(machine_capacities.keys())
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
        named_circuits=[],
    )
