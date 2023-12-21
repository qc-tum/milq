"""Module for setting up the base LP instance."""
from qiskit import QuantumCircuit
import numpy as np
import pulp

from src.common import CircuitJob
from src.provider import Accelerator
from .types import LPInstance, JobHelper, PTimes, STimes


def set_up_base_lp(
    base_jobs: list[CircuitJob] | list[QuantumCircuit],
    accelerators: list[Accelerator] | dict[str, int],
    big_m: int,
    timesteps: int,
) -> LPInstance:
    """Wrapper to set up the base LP instance through one function.

    Generates a base LP instance with the given jobs and accelerators.
    It contains all the default constraints and variables.
    Does not contain the constraints regarding the successor relationship.

    Args:
        base_jobs (list[CircuitJob] | list[QuantumCircuit]): The list of quantum cirucits (jobs).
        accelerators (list[Accelerator] | dict[str, int]):
            The list of available accelerators (machines).
        big_m (int): Metavariable for the LP.
        timesteps (int): Meta variable for the LP, big enough to cover largest makespan.

    Returns:
        LPInstance: The LP instance object.

    Raises:
        NotImplementedError: If the input types are not supported.
    """
    if isinstance(accelerators, list):
        return _set_up_base_lp_exec(base_jobs, accelerators, big_m, timesteps)
    if isinstance(accelerators, dict):
        return _set_up_base_lp_info(base_jobs, accelerators, big_m, timesteps)

    raise NotImplementedError


def _set_up_base_lp_exec(
    base_jobs: list[CircuitJob],
    accelerators: list[Accelerator],
    big_m: int,
    timesteps: int,
) -> LPInstance:
    """Sets up the base LP instance for use in the provider.

    Generates a base LP instance with the given jobs and accelerators.
    It contains all the default constraints and variables.
    Does not contain the constraints regarding the successor relationship.

    Args:
        base_jobs (list[CircuitJob]): The list of quantum cirucits (jobs).
        accelerators (list[Accelerator]): The list of available accelerators (machines).
        big_m (int): Metavariable for the LP.
        timesteps (int): Meta variable for the LP, big enough to cover largest makespan.

    Returns:
        LPInstance: The LP instance object.
    """
    # Set up input params
    job_capacities = {
        str(job.uuid): job.circuit.num_qubits
        for job in base_jobs
        if job.circuit is not None
    }
    job_capacities = {"0": 0} | job_capacities
    machine_capacities = {str(qpu.uuid): qpu.qubits for qpu in accelerators}

    lp_instance = _define_lp(
        job_capacities, machine_capacities, list(range(timesteps)), big_m
    )
    lp_instance.named_circuits = [JobHelper("0", None)] + [
        JobHelper(str(job.uuid), job.circuit)
        for job in base_jobs
        if job.circuit is not None
    ]
    return lp_instance


def _set_up_base_lp_info(
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    big_m: int,
    timesteps: int,
) -> LPInstance:
    """Sets up the base LP instance for use outside of provider.

    Generates a base LP instance with the given jobs and accelerators.
    It contains all the default constraints and variables.
    Does not contain the constraints regarding the successor relationship.

    Args:
        base_jobs (list[QuantumCircuit]): The list of quantum cirucits (jobs).
        accelerators (dict[str, int]): The list of available accelerators (machines).
        big_m (int): Metavariable for the LP.
        timesteps (int): Meta variable for the LP, big enough to cover largest makespan.

    Returns:
        LPInstance: The LP instance object.
    """
    # Set up input params
    job_capacities = {str(idx + 1): job.num_qubits for idx, job in enumerate(base_jobs)}
    job_capacities = {"0": 0} | job_capacities

    machine_capacities = accelerators

    lp_instance = _define_lp(
        job_capacities, machine_capacities, list(range(timesteps)), big_m
    )
    lp_instance.named_circuits = [JobHelper("0", None)] + [
        JobHelper(str(idx + 1), job) for idx, job in enumerate(base_jobs)
    ]
    return lp_instance


def _define_lp(
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


def set_up_simple_lp(
    lp_instance: LPInstance,
    process_times: PTimes,
    setup_times: STimes,
) -> LPInstance:
    """Sets up the LP for the simple scheduling problem.

    Setup times are overestimated, and not depending on the sequence.

    Args:
        lp_instance (LPInstance): The base LP.
        process_times (PTimes): Original process times.
        setup_times (STimes): Original setup times.

    Returns:
        LPInstance: The updated LP instance.
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
    return lp_instance


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


def set_up_extended_lp(
    lp_instance: LPInstance,
    process_times: PTimes,
    setup_times: STimes,
    big_m: int = 1000,
) -> LPInstance:
    """Sets up the LP for the extended scheduling problem.

    This uses the complex successor relationship.

    Args:
        lp_instance (LPInstance): The base LP.
        process_times (PTimes): Original process times.
        setup_times (STimes): Original setup times.
        big_m (int, optional): Metavariable for the LP. Defaults to 1000.

    Returns:
        LPInstance: The updated LP instance.
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
    return lp_instance
