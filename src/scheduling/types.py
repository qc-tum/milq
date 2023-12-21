"""Helper Classes for Scheduling Tasks."""
from dataclasses import dataclass, field
from enum import auto, Enum

from qiskit import QuantumCircuit
import pulp

from src.common import CircuitJob
from src.provider import Accelerator


class SchedulerType(Enum):
    """The type of scheduler to use."""

    BASELINE = auto()
    SIMPLE = auto()
    EXTENDED = auto()


@dataclass
class Bin:
    """Helper to keep track of binning problem."""

    capacity: int
    index: int
    qpu: int
    jobs: list[QuantumCircuit] = field(default_factory=list)
    full: bool = False


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
Benchmark = list[  # TODO should we move this?
    dict[str, dict[str, int] | list[dict[str, PTimes | STimes | dict[str, Result]]]]
]


@dataclass
class ExecutableProblem:
    """Defines an executable problem.

    This calculates setup and process times based on the accelerators.
    """

    base_jobs: list[CircuitJob]
    accelerators: list[Accelerator]
    big_m: int
    timesteps: int


@dataclass
class InfoProblem:
    """Defines an "InfoProblem" whis is used for evaluation purposes.

    This requires setup and process times to be defined as they are
    not calculated from the accelerators.
    """

    base_jobs: list[QuantumCircuit]
    accelerators: dict[str, int]
    big_m: int
    timesteps: int
    process_times: PTimes
    setup_times: STimes
