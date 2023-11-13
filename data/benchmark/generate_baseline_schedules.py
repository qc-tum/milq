"""Generate baseline schedules."""

from qiskit import QuantumCircuit

from .types import Bin, JobResultInfo, Result


def generate_baseline_schedule(
    jobs: list[QuantumCircuit], accelerators: dict[str, int]
) -> tuple[float, list[JobResultInfo]]:
    """Generate baseline schedule."""

    def find_fitting_bin(job: QuantumCircuit, bins: list[Bin]) -> int | None:
        for idx, b in enumerate(bins):
            if b.capacity >= job.num_qubits:
                return idx
        return None

    open_bins = [
        Bin(index=0, capacity=qpu, qpu=idx)
        for idx, qpu in enumerate(accelerators.values())
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
        selected_bin.capacity -= job.num_qubits

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
    # TODO: calclulate makespan and schedule
    # for _bin in sorted(closed_bins, key=lambda x: x.index):
    #     combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
    return 0, combined_jobs


def _calculate_result_from_baseline(
    jobs: list[JobResultInfo],
    p_times: list[list[float]],
    s_times: list[list[list[float]]],
) -> Result:
    return Result(0.0, [])
