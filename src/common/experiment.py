"""Data object for Experiments belonging to one cut circuit."""
from dataclasses import dataclass, field
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
    instance: QuantumCircuit | None
    coefficient: tuple[float, WeightType]
    n_shots: int
    observable: PauliList  # Should be single pauli
    partition_lable: str
    result_counts: dict[str, int] | None
    uuid: UUID
    cregs: int


@dataclass
class CombinedJob:
    """Data class for combined circuit object.
    Order of the lists has to be correct for all!
    """

    indices: list[int] = field(default_factory=list)
    instance: QuantumCircuit | None = None
    coefficients: list[tuple[float, WeightType]] = field(default_factory=list)
    mapping: list[slice] = field(default_factory=list)
    n_shots: int = 0
    observable: PauliList | None = None
    partition_lables: list[str] = field(default_factory=list)
    result_counts: dict[str, int] | None = None
    uuids: list[UUID] = field(default_factory=list)
    cregs: list[int] = field(default_factory=list)


def create_jobs_from_experiment(experiment: Experiment) -> list[CircuitJob]:
    """_summary_

    Args:
        experiment (Experiment): _description_

    Returns:
        list[CircuitJob]: _description_
    """
    return [
        CircuitJob(
            idx,
            circuit,
            experiment.coefficients[idx],
            experiment.n_shots,
            experiment.observables,  # TODO this might need to change for proper observables
            experiment.partition_label,
            None,
            experiment.uuid,
            len(circuit.cregs),
        )
        for idx, circuit in enumerate(experiment.circuits)
    ]
