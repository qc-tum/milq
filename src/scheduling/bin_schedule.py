"""Generate baseline schedules."""

from uuid import uuid4

from qiskit import QuantumCircuit

from src.common import CircuitJob, ScheduledJob
from src.provider import Accelerator
from src.scheduling.common import do_bin_pack
from src.tools import assemble_job
from .types import JobResultInfo


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
    closed_bins = do_bin_pack(jobs, list(accelerators.values()))
    combined_jobs: list[JobResultInfo] = []
    for _bin in sorted(closed_bins, key=lambda x: x.index):
        for job in _bin.jobs:
            if job is None or job.circuit is None:
                continue
            combined_jobs.append(
                JobResultInfo(
                    name=job.circuit.name,
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
    closed_bins = do_bin_pack(jobs, [qpu.qubits for qpu in accelerators])
    # Build combined jobs from bins
    combined_jobs = []
    for _bin in sorted(closed_bins, key=lambda x: x.index):
        combined_jobs.append(ScheduledJob(job=assemble_job(_bin.jobs), qpu=_bin.qpu))
    return combined_jobs
