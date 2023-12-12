"""Methods for generating a schedule for a given provider."""
from bisect import insort
from collections import defaultdict

import pulp

from src.common import CircuitJob, ScheduledJob
from src.scheduling import Bin, JobResultInfo, LPInstance
from src.tools import assemble_job
from .accelerator import Accelerator


def generate_baseline_schedule(
    jobs: list[CircuitJob], accelerators: list[Accelerator], **kwargs
) -> list[ScheduledJob]:
    """Schedule jobs onto qpus.

    Each qpu represents a bin.
    Since all jobs are asumed to take the same amount of time, they are associated
    with a timestep (index).
    k-first fit bin means we keep track of all bins that still have space left.
    Once a qpu is full, we add a new bin for all qpus at the next timestep.
    We can't run circuits with one qubit, scheduling doesn't take this into account.
    Args:
        jobs (list[CircuitJob]): The list of jobs to run.
         accelerators (list[Accelerator]): The list of available accelerators.
    Returns:
        list[ScheduledJob]: A list of Jobs scheduled to accelerators.
    """
    # Use binpacking to combine circuits into qpu sized jobs
    # placeholder for propper scheduling
    # TODO set a flag when an experiment is done
    # TODO consider number of shots
    # Assumption: bins should be equally loaded and take same amount of time

    def find_fitting_bin(job: CircuitJob, bins: list[Bin]) -> int | None:
        for idx, b in enumerate(bins):
            if b.capacity >= job.instance.num_qubits:
                return idx
        return None

    open_bins = [
        Bin(index=0, capacity=qpu.qubits, qpu=idx)
        for idx, qpu in enumerate(accelerators)
    ]
    closed_bins = []
    index = 1
    for job in jobs:
        if job.instance is None:
            continue
        # Find the index of a fitting bin
        bin_idx = find_fitting_bin(job, open_bins)

        if bin_idx is None:
            # Open new bins
            new_bins = [
                Bin(index=index, capacity=qpu.qubits, qpu=idx)
                for idx, qpu in enumerate(accelerators)
            ]
            index += 1

            # Search for a fitting bin among the new ones
            bin_idx = find_fitting_bin(job, new_bins)
            assert bin_idx is not None, "Job doesn't fit onto any qpu"
            bin_idx += len(open_bins)
            open_bins += new_bins

        # Add job to selected bin
        selected_bin = open_bins[bin_idx]
        selected_bin.jobs.append(job)
        selected_bin.capacity -= job.instance.num_qubits

        # Close bin if full
        if selected_bin.capacity == 0:
            selected_bin.full = True
            closed_bins.append(selected_bin)
            del open_bins[bin_idx]

    # Close all open bins
    for obin in open_bins:
        if len(obin.jobs) > 0:
            closed_bins.append(obin)

    # Build combined jobs from bins
    combined_jobs = []
    for _bin in sorted(closed_bins, key=lambda x: x.index):
        combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
    return combined_jobs


def generate_simple_schedule(
    jobs: list[CircuitJob],
    accelerators: list[Accelerator],
    big_m: int = 1000,
    t_max: int = 2**7,
    **kwargs,
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
    lp_instance = _set_up_base_lp(jobs, accelerators, big_m, list(range(t_max)))
    # (4) - (7), (9)
    p_times = pulp.makeDict(
        [lp_instance.jobs[1:], lp_instance.machines],
        _get_processing_times(jobs, accelerators),
        0,
    )
    s_times = pulp.makeDict(
        [lp_instance.jobs[1:], lp_instance.machines],
        _get_simple_setup_times(jobs, accelerators),
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

    return _solve_lp(lp_instance, jobs, accelerators)


def generate_extended_schedule(
    jobs: list[CircuitJob],
    accelerators: list[Accelerator],
    big_m: int = 1000,
    t_max: int = 2**7,
    **kwargs,
) -> list[ScheduledJob]:
    """Calclulate a schedule for the given jobs and accelerators based on the extended MILP.

    The extended MILP includes the sequence dependent setup time between jobs.
    This depends on the unique pre- and successor condition described in the paper.
    Args:
        jobs (list[CircuitJob]): The list of jobs to run.
        accelerators (list[Accelerator]): The list of available accelerators.
        big_m (int, optional): M hepler for LP. Defaults to 1000.
        t_max (int, optional): Max number of Timesteps. Defaults to 2**7.
    Returns:
        list[ScheduledJob]: A list of Jobs scheduled to accelerators.
    """
    lp_instance = _set_up_base_lp(jobs, accelerators, big_m, list(range(t_max)))

    # additional parameters
    p_times = pulp.makeDict(
        [lp_instance.jobs, lp_instance.machines],
        _get_processing_times(jobs, accelerators),
        0,
    )
    # TODO check if this works correctly for job "0"
    s_times = pulp.makeDict(
        [lp_instance.jobs, lp_instance.jobs, lp_instance.machines],
        _get_setup_times(jobs, accelerators, kwargs.get("default_value", 50)),
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
    return _solve_lp(lp_instance, jobs, accelerators)


def _set_up_base_lp(
    base_jobs: list[CircuitJob],
    accelerators: list[Accelerator],
    big_m: int,
    timesteps: list[int],
) -> LPInstance:
    # Set up input params
    jobs = ["0"] + [str(job.uuid) for job in base_jobs]
    job_capacities = {
        str(job.uuid): job.instance.num_qubits
        for job in base_jobs
        if job.instance is not None
    }
    job_capacities["0"] = 0
    machines = [str(qpu.uuid) for qpu in accelerators]
    machine_capacities = {str(qpu.uuid): qpu.qubits for qpu in accelerators}

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
    )


def _get_processing_times(
    base_jobs: list[CircuitJob],
    accelerators: list[Accelerator],
) -> list[list[float]]:
    return [
        [qpu.compute_processing_time(job.instance) for qpu in accelerators]
        for job in base_jobs
        if job.instance is not None
    ]


def _get_setup_times(
    base_jobs: list[CircuitJob], accelerators: list[Accelerator], default_value: int
) -> list[list[list[float]]]:
    return [
        [
            [
                qpu.compute_setup_time(job_i.instance, job_j.instance)
                for qpu in accelerators
            ]
            for job_i in base_jobs
            if job_i.instance is not None
        ]
        for job_j in base_jobs
        if job_j.instance is not None
    ]


def _get_simple_setup_times(
    base_jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[list[float]]:
    return [
        [
            qpu.compute_setup_time(job_i.instance, circuit_to=None)
            for qpu in accelerators
        ]
        for job_i in base_jobs
        if job_i.instance is not None
    ]


def _solve_lp(
    lp_instance: LPInstance, jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[ScheduledJob]:
    solver_list = pulp.listSolvers(onlyAvailable=True)
    gurobi = "GUROBI_CMD"
    if gurobi in solver_list:
        solver = pulp.getSolver(gurobi)
        lp_instance.problem.solve(solver)
    else:
        lp_instance.problem.solve()
    return _generate_schedule_from_lp(lp_instance, jobs, accelerators)


def _generate_schedule_from_lp(
    lp_instance: LPInstance, jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[ScheduledJob]:
    assigned_jobs = {
        job: JobResultInfo(name=job, machine="", start_time=-1.0, completion_time=-1.0)
        for job in lp_instance.jobs
    }
    for var in lp_instance.problem.variables():
        if var.name.startswith("x_") and var.varValue > 0.0:
            name = var.name.split("_")[2:]
            assigned_jobs["-".join(name[:5])].machine = "-".join(name[-5:])
        elif var.name.startswith("s_"):
            name = "-".join(var.name.split("_")[2:])
            assigned_jobs[name].start_time = float(var.varValue)
        elif var.name.startswith("c_"):
            name = "-".join(var.name.split("_")[2:])
            assigned_jobs[name].completion_time = float(var.varValue)
    del assigned_jobs["0"]
    machine_assignments: dict[str, list[JobResultInfo]] = defaultdict(list)
    for job in assigned_jobs.values():
        if job.machine != "":
            machine_assignments[job.machine].append(job)

    closed_bins = []
    accelerator_uuids = [str(qpu.uuid) for qpu in accelerators]
    for machine, machine_jobs in machine_assignments.items():
        try:
            machine_idx = accelerator_uuids.index(machine)
        except ValueError:
            continue
        machine_capacity = accelerators[machine_idx].qubits
        closed_bins += _form_bins(machine_capacity, machine_idx, machine_jobs, jobs)
    combined_jobs = []

    for _bin in sorted(closed_bins, key=lambda x: x.index):
        if len(_bin.jobs) > 0:
            combined_jobs.append(
                ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu)
            )
    return combined_jobs


def _form_bins(
    machine_capacity: int,
    machine_id: int,
    assigned_jobs: list[JobResultInfo],
    jobs: list[CircuitJob],
) -> list[Bin]:
    # TODO: adapat number of shots
    bins: list[Bin] = []
    current_time = -1.0
    open_jobs: list[JobResultInfo] = []
    counter = -1
    current_bin = Bin(capacity=machine_capacity, index=counter, qpu=machine_id)

    for job in sorted(assigned_jobs, key=lambda x: x.start_time):
        if job.start_time == current_time:
            # s_i = s_j -> add to same bin
            _append_if_exists(job, current_bin, jobs, open_jobs=open_jobs)
            continue

        # s_i > s_j -> add to new bin
        counter += 1
        _bin = Bin(capacity=machine_capacity, index=counter, qpu=machine_id)
        if len(open_jobs) == 0:
            # no open jobs -> add simply add to new bin
            _append_if_exists(job, _bin, jobs, open_jobs=open_jobs)

        elif open_jobs[0].completion_time > job.start_time:
            # noone finishes before job starts -> add to new bin which includes all open jobs
            _append_if_exists(
                job, _bin, jobs, current_bin=current_bin, open_jobs=open_jobs
            )
        else:
            # someone finishes before job starts
            # -> add bin for each job that finishes before job starts
            open_jobs_copy = open_jobs.copy()
            for open_job in open_jobs_copy:
                if open_job.completion_time > job.start_time:
                    # found the first that is still running, can stop
                    _append_if_exists(
                        job, _bin, jobs, current_bin=current_bin, open_jobs=open_jobs
                    )
                    break
                if open_job not in open_jobs:
                    # has been removed in the meantime
                    continue
                # remove the last job and all that end at the same time
                _bin.jobs = current_bin.jobs
                for second_job in open_jobs_copy:
                    if second_job.completion_time == open_job.completion_time:
                        _append_if_exists(
                            second_job, _bin, jobs, open_jobs=open_jobs, do_remove=True
                        )
                current_bin = _bin
                counter += 1
                _bin = Bin(capacity=machine_capacity, index=counter, qpu=machine_id)

        bins.append(_bin)
        current_time = job.start_time
        current_bin = _bin

    return bins


def _append_if_exists(
    job: JobResultInfo,
    _bin: Bin,
    jobs: list[CircuitJob],
    current_bin: Bin | None = None,
    open_jobs: list[JobResultInfo] | None = None,
    do_remove: bool = False,
) -> None:
    if cjob := next((j for j in jobs if str(j.uuid) == job.name), None):
        if current_bin is not None:
            _bin.jobs = current_bin.jobs
        _bin.jobs.append(cjob)
        if open_jobs is not None:
            if do_remove:
                open_jobs.remove(job)
            else:
                insort(open_jobs, job, key=lambda x: x.completion_time)
