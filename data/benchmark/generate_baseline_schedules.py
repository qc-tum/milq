"""Generate baseline schedules."""
from collections import defaultdict

import pulp
from qiskit import QuantumCircuit

from .types import Bin, JobHelper, JobResultInfo


def generate_baseline_schedule(
    jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    process_times: list[list[float]],
    setup_times: list[list[list[float]]],
) -> tuple[float, list[JobResultInfo]]:
    """Generate baseline schedule."""

    def find_fitting_bin(job: JobHelper, bins: list[Bin]) -> int | None:
        for idx, b in enumerate(bins):
            if b.capacity >= job.instance.num_qubits:
                return idx
        return None

    new_jobs = [JobHelper(str(idx + 1), job) for idx, job in enumerate(jobs)]
    open_bins = [
        Bin(index=0, capacity=qpu, qpu=idx)
        for idx, qpu in enumerate(accelerators.values())
    ]
    closed_bins = []
    index = 1
    for job in new_jobs:
        if job is None or job.instance is None:
            continue
        # Find the index of a fitting bin
        bin_idx = find_fitting_bin(job, open_bins)

        if bin_idx is None:
            # Open new bins
            new_bins = [
                Bin(index=index, capacity=qpu, qpu=idx)
                for idx, qpu in enumerate(accelerators.values())
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
    combined_jobs: list[JobResultInfo] = []
    for _bin in sorted(closed_bins, key=lambda x: x.index):
        # combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
        for job in _bin.jobs:
            if job is None or job.instance is None:
                continue
            combined_jobs.append(
                JobResultInfo(
                    name=job.name,
                    machine=list(accelerators.keys())[_bin.qpu],
                    start_time=_bin.index,
                    completion_time=-1.0,
                    capacity=job.instance.num_qubits,
                )
            )

    return _calculate_result_from_baseline(
        combined_jobs, process_times, setup_times, jobs, accelerators
    )


def _calculate_result_from_baseline(
    jobs: list[JobResultInfo],
    process_times: list[list[float]],
    setup_times: list[list[list[float]]],
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
) -> tuple[float, list[JobResultInfo]]:
    lp_jobs = ["0"] + [str(idx + 1) for idx, _ in enumerate(base_jobs)]
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

    return _calculate_makespan(jobs, p_times, s_times), jobs


def _calculate_makespan(
    jobs: list[JobResultInfo],
    p_times: defaultdict[str, defaultdict[str, float]],
    s_times: defaultdict[str, defaultdict[str, defaultdict[str, float]]],
) -> float:
    """Calculates the actual makespan from the list of jobs."""
    assigned_machines: defaultdict[str, list[JobResultInfo]] = defaultdict(list)
    for job in jobs:
        assigned_machines[job.machine].append(job)
    makespans = []
    for machine, assigned_jobs in assigned_machines.items():
        for job in sorted(assigned_jobs, key=lambda x: x.start_time):
            # Find the last predecessor that is completed before the job starts
            # this can technically change the correct predecessor to a wrong one
            # because completion times are updated in the loop
            # I'm not sure if copying before the loop corrects this
            last_completed = max(
                (job for job in assigned_jobs), key=lambda x: x.completion_time
            )
            if job.start_time == 0.0:
                last_completed = JobResultInfo("0", machine, 0.0, 0.0)
            job.start_time = last_completed.completion_time
            # calculate p_j + s_ij
            completion_time = (  # check if this order is correct
                last_completed.completion_time
                + p_times[job.name][machine]
                + s_times[last_completed.name][job.name][machine]
            )
            job.completion_time = completion_time
        makespans.append(max(job.completion_time for job in assigned_jobs))

    return max(makespans)
