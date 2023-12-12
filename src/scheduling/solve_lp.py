from collections import defaultdict
from bisect import insort

import pulp

from src.common import CircuitJob, ScheduledJob
from src.provider import Accelerator
from src.tools import assemble_job

from .types import Bin, LPInstance, JobResultInfo


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
        combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
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
