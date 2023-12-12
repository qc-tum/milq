"""Generate baseline schedules."""
from collections import defaultdict

import pulp
from qiskit import QuantumCircuit

from src.scheduling import Bin, JobHelper, JobResultInfo, PTimes, STimes


def generate_baseline_schedule(
    jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
    process_times: PTimes,
    setup_times: STimes,
) -> tuple[float, list[JobResultInfo]]:
    """Generates a baseline schedule for the given jobs and accelerators using binpacking.

    First generates the schedule using binpacking and then calculates the makespan
    by executing the schedule with the correct p_ij and s_ij values.

    Args:
        jobs (list[QuantumCircuit]): The list of circuits (jobs) to schedule.
        accelerators (dict[str, int]): The list of accelerators to schedule on (bins).
        process_times (PTimes): The process times for each job on each machine.
        setup_times (STimes): The setup times for each job on each machine.

    Returns:
        tuple[float, list[JobResultInfo]]: List of jobs with their assigned machine and
            start and completion times.
    """

    def find_fitting_bin(job: JobHelper, bins: list[Bin]) -> int | None:
        if job.circuit is None:
            return None
        for idx, b in enumerate(bins):
            if b.capacity >= job.circuit.num_qubits:
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
        if job is None or job.circuit is None:
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
        selected_bin.capacity -= job.circuit.num_qubits

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
            if job is None or job.circuit is None:
                continue
            combined_jobs.append(
                JobResultInfo(
                    name=job.name,
                    machine=list(accelerators.keys())[_bin.qpu],
                    start_time=_bin.index,
                    completion_time=-1.0,
                    capacity=job.circuit.num_qubits,
                )
            )

    return _calculate_result_from_baseline(
        combined_jobs, process_times, setup_times, jobs, accelerators
    )


def _calculate_result_from_baseline(
    jobs: list[JobResultInfo],
    process_times: PTimes,
    setup_times: STimes,
    base_jobs: list[QuantumCircuit],
    accelerators: dict[str, int],
) -> tuple[float, list[JobResultInfo]]:
    """Converst the setup and process times to a format that can be used by the
    _calculate_makespan function.
    """
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

    return calculate_bin_makespan(jobs, p_times, s_times), jobs
