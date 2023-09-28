"""Data object for Experiments belonging to one cut circuit."""
from dataclasses import dataclass
from uuid import UUID

from circuit_knitting.cutting.qpd import WeightType
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList


@dataclass
class Experiment:
    """Data class for cut results."""

    circuits: list[QuantumCircuit]
    coefficients: list[tuple[float, WeightType]]
    n_shots: int
    observables: PauliList | dict[str, PauliList]
    partition_label: str
    result_counts: list[dict[str, int]] | None
    uuid: UUID


@dataclass
class CircuitJob:
    """Data class for single cicruit"""

    index: int
    instance: QuantumCircuit
    coefficient: tuple[float, WeightType]
    n_shots: int
    observable: PauliList  # Should be single pauli
    partition_lable: str
    result_counts: dict[str, int] | None
    uuid: UUID
