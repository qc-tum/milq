"""Data structures for population-based heuristics."""

from dataclasses import dataclass, field

from qiskit import QuantumCircuit

from src.common import CircuitJob


@dataclass
class Bucket:
    """A bucket is a list of jobs that are performed on the same machine at one timestep."""

    # All
    jobs: list[CircuitJob] = field(default_factory=list)
    # max_duration: int
    # start_time: int
    # end_time: int


@dataclass
class Machine:
    """A machine has a list of jobs, which are performed in buckets over several timesteps.
    One bucket represents one timestep.
    """

    capacity: int
    id: str
    buckets: list[Bucket]  # Has to be ordered
    makespan: float = 0.0


@dataclass
class Schedule:
    """A schedule is a list of machines, and their jobs."""

    machines: list[Machine]
    makespan: float


@dataclass
class MakespanInfo:
    """Dataclass to track job completion times for makespan calc"""

    job: QuantumCircuit | None
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
