"""Methods for generating a schedule for a given provider."""
from dataclasses import dataclass, field

from src.common import CircuitJob, ScheduledJob
from src.tools import assemble_job
from .accelerator import Accelerator


@dataclass
class Bin:
    """Helper to keep track of binning problem."""

    capacity: int = 0
    full: bool = False
    index: int = -1
    jobs: list[CircuitJob] = field(default_factory=list)
    qpu: int = -1


def generate_baseline_schedule(
    jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[ScheduledJob]:
    """Schedule jobs onto qpus.

    Each qpu represents a bin.
    Since all jobs are asumet to take the same amount of time, the are associated
    with a timestep (index).
    k-first fit bin means we keep track of all bins that still have space left.
    Once a qpu is full, we add a new bin for each qpu at the next timestep.
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
    # Assumption: beens should be equally loaded and take same amoutn of time
    open_bins = [
        Bin(index=0, capacity=qpu.qubits, qpu=idx)
        for idx, qpu in enumerate(accelerators)
    ]
    closed_bins = []
    index = 1
    for job in jobs:
        if job.instance is None:
            continue
        for obin in open_bins:
            # TODO consider 1 free qubit remaining
            if obin.capacity >= job.instance.num_qubits:
                obin.jobs.append(job)
                obin.capacity -= job.instance.num_qubits
                if obin.capacity <= 1:
                    obin.full = True
                    closed_bins.append(obin)
                    open_bins.remove(obin)
                break
        else:
            new_bins = [
                Bin(index=index, capacity=qpu.qubits, qpu=idx)
                for idx, qpu in enumerate(accelerators)
            ]
            index += 1
            for nbin in new_bins:
                # TODO consider 1 free qubit remaining
                if nbin.capacity >= job.instance.num_qubits:
                    nbin.jobs.append(job)
                    nbin.capacity -= job.instance.num_qubits
                    if nbin.capacity == 0:
                        nbin.full = True
                        closed_bins.append(nbin)
                        new_bins.remove(nbin)
                    break
            open_bins += new_bins
    for obin in open_bins:
        if len(obin.jobs) > 0:
            closed_bins.append(obin)
    combined_jobs = []
    for _bin in sorted(closed_bins, key=lambda x: x.index):
        combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
    return combined_jobs


def generate_simple_schedule(
    jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[ScheduledJob]:
    pass

def generate_extended_schedule(
    jobs: list[CircuitJob], accelerators: list[Accelerator]
) -> list[ScheduledJob]:
    pass  # MILP With setup times
