"""Process the results of the LP solver to generate a schedule."""
from collections import defaultdict
from bisect import insort
from typing import Callable

import pulp

from src.common import CircuitJob, ScheduledJob
from src.provider import Accelerator
from src.tools import assemble_job

from .calculate_makespan import calculate_bin_makespan
from .types import Bin, LPInstance, JobResultInfo, PTimes, STimes


def generate_bin_info_schedule(
    jobs: list[JobResultInfo],
    process_times: PTimes,
    setup_times: STimes,
    accelerators: dict[str, int],
) -> tuple[float, list[JobResultInfo]]:
    """Generates a schedule for evaluation purposes from bin packing.

    TODO make in to a overloaded function, summarize input params
    Args:
        jobs (list[JobResultInfo]): The list of the scheduled jobs.
        process_times (PTimes): The original processing times.
        setup_times (STimes): The original setup times.
        accelerators (dict[str, int]): The list of used accelerators.

    Returns:
        tuple[float, list[JobResultInfo]]: The objective value and the schedule.
    """
    lp_jobs = ["0"] + [job.name for job in jobs]  # TODO
    machines = list(accelerators.keys())
    p_times = pulp.makeDict(
        [lp_jobs[1:], machines],
        process_times,
        0,
    )
    s_times = pulp.makeDict(
        [lp_jobs, lp_jobs, machines],
        setup_times,
        0,
    )
    return calculate_bin_makespan(jobs, p_times, s_times, accelerators), jobs


def extract_info_schedule(
    lp_instance: LPInstance,
) -> tuple[float, list[JobResultInfo]]:
    """Extracts a schedule for evaluation purposes.

    Args:
        lp_instance (LPInstance): A solved LP instance.

    Returns:
        tuple[float, list[JobResultInfo]]: The objective value and the list of jobs with their
            with their assigned machine and start and completion times.
    """
    # TODO check if _first_name_func is needed once we change to uuids
    assigned_jobs = _extract_gurobi_results(lp_instance, _first_name_func)
    return lp_instance.problem.objective.value(), list(assigned_jobs.values())


def extract_executable_schedule(
    lp_instance: LPInstance, jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[ScheduledJob]:
    """Extracts a schedule for execution purposes.

    Solves the problem and generates bins to execute simultaneous jobs.
    TODO still assumes that bins take same time. This is only done for acceleratorgroup.
    Args:
        lp_instance (LPInstance): A solved LP instance.
        jobs (list[CircuitJob]): The list of original jobs.
        accelerators (list[Accelerator]): The list of available accelerators.

    Returns:
        list[ScheduledJob]: _description_
    """
    assigned_jobs = _extract_gurobi_results(lp_instance, _second_name_func)
    return _generate_schedule_from_lp(assigned_jobs, jobs, accelerators)


def _extract_gurobi_results(
    lp_instance: LPInstance, name_function: Callable[[str], tuple[str, str]]
) -> dict[str, JobResultInfo]:
    assigned_jobs = {
        job.name: JobResultInfo(
            name=job.name,
            machine="",
            start_time=-1.0,
            completion_time=-1.0,
            capacity=job.circuit.num_qubits,
        )
        if job.circuit is not None
        else JobResultInfo(
            name=job.name,
            machine="",
            start_time=-1.0,
            completion_time=-1.0,
            capacity=0,
        )
        for job in lp_instance.named_circuits
    }
    for var in lp_instance.problem.variables():
        if var.name.startswith("x_") and var.varValue > 0.0:
            names = name_function(var.name)
            assigned_jobs[names[0]].machine = names[1]
        elif var.name.startswith("s_"):
            name = "-".join(var.name.split("_")[2:])
            assigned_jobs[name].start_time = float(var.varValue)
        elif var.name.startswith("c_"):
            name = "-".join(var.name.split("_")[2:])
            # TODO for some reason this was name[0] before
            assigned_jobs[name].completion_time = float(var.varValue)
    del assigned_jobs["0"]
    return assigned_jobs


def _generate_schedule_from_lp(
    assigned_jobs: dict[str, JobResultInfo],
    jobs: list[CircuitJob],
    accelerators: list[Accelerator],
) -> list[ScheduledJob]:
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
            current_bin = _bin
            current_time = job.start_time
        elif open_jobs[0].completion_time > job.start_time:
            # noone finishes before job starts -> add to new bin which includes all open jobs
            _append_if_exists(
                job, _bin, jobs, current_bin=current_bin, open_jobs=open_jobs
            )
            bins.append(current_bin)
            current_bin = _bin
            current_time = job.start_time
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
                    if second_job == open_job:
                        continue
                    if second_job.completion_time == open_job.completion_time:
                        _append_if_exists(
                            second_job, _bin, jobs, open_jobs=open_jobs, do_remove=True
                        )
                current_bin = _bin
                counter += 1
                _bin = Bin(capacity=machine_capacity, index=counter, qpu=machine_id)

            bins.append(current_bin)
            current_bin = _bin
            current_time = job.start_time

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


def _first_name_func(name: str) -> tuple[str, str]:
    # For single character jobs
    names = name.split("_")[2:]
    return names[0], names[1]


def _second_name_func(name: str) -> tuple[str, str]:
    # For UUIDS
    names = name.split("_")[2:]
    return "-".join(names[:5]), "-".join(names[-5:])
