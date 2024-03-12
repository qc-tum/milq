"""Data structures for population-based heuristics."""

from dataclasses import dataclass, field
from uuid import UUID

from qiskit import QuantumCircuit


@dataclass
class CircuitProxy:
    """A proxy for a quantum circuit to be used in the population-based heuristics."""

    origin: QuantumCircuit
    processing_time: float
    num_qubits: int
    uuid: UUID
    indices: list[int] | None = None
    n_shots: int = 1024  # Not sure if needed
    noise: float = 0.0
    # Classical scheduling information
    priority: int = 1
    strictness: int = 1
    preselection: str | None = None


@dataclass
class Bucket:
    """A bucket is a list of jobs that are performed on the same machine at one timestep."""

    # All
    jobs: list[CircuitProxy] = field(default_factory=list)
    # For the final schedule
    circuits: list[QuantumCircuit] = field(default_factory=list)

    # max_duration: int
    # start_time: int
    # end_time: int
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Bucket):
            return False
        return sorted([(job.uuid, job.indices) for job in self.jobs]) == sorted(
            [(job.uuid, job.indices) for job in __value.jobs]
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
    queue_length: float

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
    noise: float = 0.0

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

    def is_feasible(self) -> bool:
        """Checks if a schedule is feasible."""
        return all(
            sum(job.num_qubits for job in bucket.jobs) <= machine.capacity
            for machine in self.machines
            for bucket in machine.buckets
        )


@dataclass
class MakespanInfo:
    """Dataclass to track job completion times for makespan calc"""

    job: QuantumCircuit | None
    start_time: float
    completion_time: float
    capacity: int
    priority: int = 1
    strictness: int = 1
    preselection: str | None = None
