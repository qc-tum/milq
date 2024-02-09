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
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Bucket):
            return False
        return [job.uuid for job in self.jobs] == [job.uuid for job in __value.jobs]
    
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

    # def __eq__(self, __value: object) -> bool:
    #     if not isinstance(__value, Schedule):
    #         return False
    #     other_machiens = {machine.id: machine for machine in __value.machines}
    #     for machine in self.machines:
    #         if machine.id not in other_machiens:
    #             return False
    #         for bucket_self, bucket_other in zip(
    #             machine.buckets, other_machiens[machine.id].buckets
    #         ):
    #             if [job.uuid for job in bucket_self.jobs] != [
    #                 job.uuid for job in bucket_other.jobs
    #             ]:
    #                 return False
    #     return True


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
