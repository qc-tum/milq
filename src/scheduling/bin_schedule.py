"""Generate baseline schedules."""
from uuid import uuid4

from qiskit import QuantumCircuit

from src.common import CircuitJob, ScheduledJob
from src.provider import Accelerator
from src.tools import assemble_job
from .types import Bin, JobResultInfo


def generate_bin_info_schedule(
    circuits: list[QuantumCircuit],
    accelerators: dict[str, int],
) -> list[JobResultInfo]:
    """Generates a baseline schedule for the given jobs and accelerators using binpacking.

    First generates the schedule using binpacking and then calculates the makespan
    by executing the schedule with the correct p_ij and s_ij values.

    Args:
        circuits (list[QuantumCircuit]): The list of circuits (jobs) to schedule.
        accelerators (dict[str, int]): The list of accelerators to schedule on (bins).

    Returns:
        tuple[float, list[JobResultInfo]]: List of jobs with their assigned machine and
            start and completion times.
    """
    jobs = [
        CircuitJob(
            uuid=uuid4(),
            circuit=job,
            coefficient=None,
            cregs=1,
            index=0,
            n_shots=1024,
            observable="",
            partition_label="",
            result_counts={},
        )
        for job in circuits
    ]
    # Build combined jobs from bins
    closed_bins = _do_bin_pack(jobs, list(accelerators.values()))
    combined_jobs: list[JobResultInfo] = []
    for _bin in sorted(closed_bins, key=lambda x: x.index):
        for job in _bin.jobs:
            if job is None or job.circuit is None:
                continue
            combined_jobs.append(
                JobResultInfo(
                    name=str(job.uuid),
                    machine=list(accelerators.keys())[_bin.qpu],
                    start_time=_bin.index,
                    completion_time=-1.0,
                    capacity=job.circuit.num_qubits,
                )
            )

    return combined_jobs


def generate_bin_executable_schedule(
    jobs: list[CircuitJob], accelerators: list[Accelerator]
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
    closed_bins = _do_bin_pack(jobs, [qpu.qubits for qpu in accelerators])
    # Build combined jobs from bins
    combined_jobs = []
    for _bin in sorted(closed_bins, key=lambda x: x.index):
        combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
    return combined_jobs


def _do_bin_pack(
    jobs: list[CircuitJob], accelerator_capacities: list[int]
) -> list[Bin]:
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
