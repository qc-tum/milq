"""Helper Classes for Scheduling Tasks."""
from dataclasses import dataclass, field
from enum import auto, Enum

from qiskit import QuantumCircuit
import pulp


class SchedulerType(Enum):
    """The type of scheduler to use."""

    BASELINE = auto()
    SIMPLE = auto()
    EXTENDED = auto()


@dataclass
class Bin:
    """Helper to keep track of binning problem."""

    capacity: int
    full: bool = False
    index: int
    jobs: list[QuantumCircuit] = field(default_factory=list)
    qpu: int


@dataclass
class JobHelper:
    """Helper to keep track of job names."""

    name: str
    circuit: QuantumCircuit | None  # TODO optional necessary?


@dataclass
class LPInstance:
    """Helper to keep track of LP problem."""

    problem: pulp.LpProblem
    jobs: list[str]
    machines: list[str]
    x_ik: dict[str, dict[str, pulp.LpVariable]]
    z_ikt: dict[str, dict[str, dict[int, pulp.LpVariable]]]
    c_j: dict[str, pulp.LpVariable]
    s_j: dict[str, pulp.LpVariable]
    named_circuits: list[JobHelper]


@dataclass
class JobResultInfo:
    """Helper to keep track of job results."""

    name: str
    machine: str
    start_time: float
    completion_time: float
    capacity: int


@dataclass
class Result:
    """Benchmark result for one instance of setting+jobs."""

    makespan: float
    jobs: list[JobResultInfo]
    time: float


# Typedef
PTimes = list[list[float]]
STimes = list[list[list[float]]]
Benchmark = list[
    dict[str, dict[str, int] | list[dict[str, PTimes | STimes | dict[str, Result]]]]
]
