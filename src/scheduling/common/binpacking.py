"""Actual bin packing algorithm."""

from src.common import CircuitJob

from ..types import Bin


def do_bin_pack(jobs: list[CircuitJob], accelerator_capacities: list[int]) -> list[Bin]:
    """Perform first-fit decreasing bin packing on the given jobs.

    Each accelerator is represented by a bin with a given capacity.

    Args:
        jobs (list[CircuitJob]): The list of jobs to bin pack.
        accelerator_capacities (list[int]): The list of accelerator capacities.

    Returns:
        list[Bin]: A list of bins with the packed jobs.
            Bin qpu attribute is the index of the accelerator in the list.
    """
    open_bins = [
        Bin(index=0, capacity=qpu, qpu=idx)
        for idx, qpu in enumerate(accelerator_capacities)
    ]
    closed_bins = []
    index = 1
    for job in jobs:
        if job.circuit is None:
            continue
        # Find the index of a fitting bin
        bin_idx = _find_fitting_bin(job, open_bins)

        if bin_idx is None:
            # Open new bins
            new_bins = [
                Bin(index=index, capacity=qpu, qpu=idx)
                for idx, qpu in enumerate(accelerator_capacities)
            ]
            index += 1

            # Search for a fitting bin among the new ones
            bin_idx = _find_fitting_bin(job, new_bins)
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
    return closed_bins


def _find_fitting_bin(job: CircuitJob, bins: list[Bin]) -> int | None:
    if job.circuit is None:
        raise ValueError("Job has no circuit")
    for idx, b in enumerate(bins):
        if b.capacity >= job.circuit.num_qubits:
            return idx
    return None
