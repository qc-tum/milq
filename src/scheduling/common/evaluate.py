"""Evaluation and selection of solutions."""

from dataclasses import dataclass
import logging

from qiskit import QuantumCircuit

from src.common import UserCircuit
from src.tools import cut_according_to_partition
from src.provider import Accelerator
from src.scheduling.common.types import Schedule, Bucket, MakespanInfo, CircuitProxy


def evaluate_final_solution(
    schedule: Schedule,
    accelerators: list[Accelerator],
    circuits: list[QuantumCircuit | UserCircuit],
) -> tuple[float, float, float]:
    """Calculates and updates the makespan of a schedule.

    Uses the values provided by the accelerators to calculate the makespan.
    Cuts the circuits according to the schedule.
    Args:
        schedule (Schedule): A schedule to evaluate.
        accelerators (list[Accelerator]): The list of accelerators to schedule on.
        circuits (list[QuantumCircuit | UserCircuit]): The orginal list of circuits to schedule.

    Returns:
        tuple[float, float, float]: The makespan, custom metric, and noise.
    """
    logging.debug("Evaluating makespan...")
    metrics, makespans, noises = [], [], []
    _cut_according_to_schedule(schedule, circuits)
    for machine in schedule.machines:
        accelerator = next(acc for acc in accelerators if str(acc.uuid) == machine.id)
        makespan, metric = _calc_machine_makespan(machine.buckets, accelerator)
        makespans.append(makespan)
        metrics.append(metric)
        noises.append(_calc_noise(machine.buckets, accelerator))
        machine.makespan = makespans[-1]

    schedule.makespan = max(makespans)
    return makespan, max(metrics), sum(noises)


def _calc_machine_makespan(
    buckets: list[Bucket], accelerator: Accelerator
) -> tuple[float, float]:
    jobs: list[MakespanInfo] = []
    for idx, bucket in enumerate(buckets):
        # assumption: jobs take the longer of both circuits to execute and to set up
        jobs += [
            MakespanInfo(
                job=circuit,
                start_time=idx,
                completion_time=-1.0,
                capacity=circuit.num_qubits,
                preselection=job.preselection,
                priority=job.priority,
                strictness=job.strictness,
            )
            for circuit, job in zip(bucket.circuits, bucket.jobs)
        ]
    if len(jobs) == 0:
        return 0.0
    assigned_jobs = jobs.copy()
    for job in jobs:
        last_completed = max(
            (job for job in assigned_jobs), key=lambda x: x.completion_time
        )
        if job.start_time == 0.0:
            last_completed = MakespanInfo(None, 0.0, 0.0, 0)
        job.start_time = last_completed.completion_time
        job.completion_time = (
            last_completed.completion_time
            + accelerator.compute_processing_time(job.job)
            + accelerator.compute_setup_time(last_completed.job, job.job)
        )
    return max(
        jobs, key=lambda j: j.completion_time
    ).completion_time, makespan_function(jobs, str(accelerator.uuid))


@dataclass
class BucketHelper:
    """Keeps track of the bucket and job index.
    Used of reconstructing circuits and cutting them properly"""

    bucket: Bucket
    job: CircuitProxy
    idx: int
    new_circuit: QuantumCircuit | None = None


def _cut_according_to_schedule(schedule: Schedule, circuits: list[UserCircuit]) -> None:
    """Cuts the circuits according to the schedule."""
    for circuit in circuits:
        helpers: list[BucketHelper] = []
        for machine in schedule.machines:
            for bucket in machine.buckets:
                for job_index, job in enumerate(bucket.jobs):
                    if isinstance(job, QuantumCircuit):
                        continue
                    if job.uuid == circuit.name:
                        helpers.append(BucketHelper(bucket, job, job_index))
        partition = [0] * circuit.num_qubits
        for idx, helper in enumerate(helpers):
            for i in helper.job.indices:
                partition[i] = idx
        new_circuits = cut_according_to_partition(circuit.circuit, partition)
        for idx, helper in enumerate(helpers):
            new_circuit = next(
                (
                    circ
                    for circ in new_circuits
                    if circ.num_qubits == helper.job.num_qubits
                ),
                None,
            )

            if new_circuit is not None:
                helper.bucket.circuits.append(new_circuit)


def evaluate_solution(schedule: Schedule) -> Schedule:
    """Calculates and updates the makespan and noise of a schedule using the proxy values.

    Args:
        schedule (Schedule): A schedule to evaluate.

    Returns:
        Schedule: The schedule with updated makespan, noise and machine makespans.
    """
    logging.debug("Evaluating proxy makespan...")
    makespans = []
    noises = []
    for machine in schedule.machines:
        makespans.append(
            machine.queue_length + _calc_proxy_makespan(machine.buckets, machine.id)
        )
        machine.makespan = makespans[-1]
        noises.append(_calc_proxy_noise(machine.buckets))
    schedule.makespan = max(makespans)
    schedule.noise = sum(noises)
    return schedule


def _calc_proxy_makespan(
    buckets: list[Bucket],
    machine_id: str,
    set_up_values: tuple[int, int] = (1, 10),
) -> float:
    # set_up_values: use cheap set up if circuits are from the same cut
    jobs: list[MakespanInfo] = []
    for idx, bucket in enumerate(buckets):
        # assumption: jobs take the longer of both circuits to execute and to set up
        jobs += [
            MakespanInfo(
                job=job,
                start_time=idx,
                completion_time=-1.0,
                capacity=job.num_qubits,
                preselection=job.preselection,
                priority=job.priority,
                strictness=job.strictness,
                n_shots=job.n_shots,
            )
            for job in bucket.jobs
        ]
    if len(jobs) == 0:
        return 0.0

    assigned_jobs = jobs.copy()
    for job in jobs:
        last_completed = max(
            (job for job in assigned_jobs), key=lambda x: x.completion_time
        )
        if job.start_time == 0.0:
            last_completed = MakespanInfo(None, 0.0, 0.0, 0)
        job.start_time = last_completed.completion_time
        set_up_time = set_up_values[1]
        if (
            last_completed.job is not None
            and job.job.origin == last_completed.job.origin
            and job.job.indices == last_completed.job.indices
        ):
            set_up_time = set_up_values[0]
        job.completion_time = (
            last_completed.completion_time + job.job.processing_time + set_up_time
        )

    return makespan_function(jobs, machine_id)


def _calc_proxy_noise(buckets: list[Bucket]) -> float:
    return sum(job.noise for bucket in buckets for job in bucket.jobs)


def _calc_noise(buckets: list[Bucket], accelerator: Accelerator) -> float:
    noises = []
    for bucket in buckets:
        noises.append(sum(accelerator.compute_noise(circ) for circ in bucket.circuits))
    return sum(noises)


def makespan_function(
    jobs: list[MakespanInfo], machine: str, alpha: float = 1.0, beta: float = 1.0
) -> float:
    """Calsulcates for the makespan function."""
    makespan = 0.0
    for job in jobs:
        makespan += job.completion_time * job.priority * alpha
        makespan += 0 if job.preselection == machine else job.strictness * beta
    return makespan
