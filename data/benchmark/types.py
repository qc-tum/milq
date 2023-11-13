from dataclasses import dataclass, field

from qiskit import QuantumCircuit
import pulp


@dataclass
class Bin:
    """Helper to keep track of binning problem."""

    capacity: int = 0
    full: bool = False
    index: int = -1
    jobs: list[QuantumCircuit] = field(default_factory=list)
    qpu: int = -1


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


@dataclass
class JobResultInfo:
    """Helper to keep track of job results."""

    name: str
    machine: str = ""
    start_time: float = -1.0
    completion_time: float = -1.0


@dataclass
class Result:
    """Benchmark result for one instance of setting+jobs."""

    makespan: float
    jobs: list[JobResultInfo]
    time: float
