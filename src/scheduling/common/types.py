"""Data structures for population-based heuristics."""

from dataclasses import dataclass, field
from uuid import UUID

from qiskit import QuantumCircuit

from src.common import CircuitJob, CombinedJob


@dataclass
class CircuitProxy:
    """A proxy for a quantum circuit to be used in the population-based heuristics."""

    origin: QuantumCircuit
    processing_time: float
    num_qubits: int
    indices: list[int] | None = None
    uuid: UUID


@dataclass
class Bucket:
    """A bucket is a list of jobs that are performed on the same machine at one timestep."""

    # All
    jobs: list[CircuitProxy] = field(default_factory=list)

    # max_duration: int
    # start_time: int
    # end_time: int
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Bucket):
            return False
        return sorted([job.uuid for job in self.jobs]) == sorted(
            [job.uuid for job in __value.jobs]
        )


@dataclass
class Machine:
    """A machine has a list of jobs, which are performed in buckets over several timesteps.
    One bucket represents one timestep.
    """

    capacity: int
    id: str
    buckets: list[Bucket]  # Has to be ordered
    makespan: float = 0.0

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Machine) or self.id != __value.id:
            return False
        for bucket_self, bucket_other in zip(self.buckets, __value.buckets):
            if bucket_self != bucket_other:
                return False
        return True


@dataclass
class Schedule:
    """A schedule is a list of machines, and their jobs."""

    machines: list[Machine]
    makespan: float

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Schedule):
            return False
        other_machines = {machine.id: machine for machine in __value.machines}
        for machine in self.machines:
            if machine.id not in other_machines:
                return False
            if machine != other_machines[machine.id]:
                return False
        return True


@dataclass
class MakespanInfo:
    """Dataclass to track job completion times for makespan calc"""

    job: CircuitProxy | None
    start_time: float
    completion_time: float
    capacity: int


def is_feasible(schedule: Schedule) -> bool:
    """Checks if a schedule is feasible."""
    return all(
        sum(job.circuit.num_qubits for job in bucket.jobs) <= machine.capacity
        for machine in schedule.machines
        for bucket in machine.buckets
    )
