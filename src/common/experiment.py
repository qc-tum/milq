"""Data object for Experiments belonging to one cut circuit."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from uuid import UUID

from circuit_knitting.cutting.qpd import WeightType
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList


@dataclass
class Experiment:
    """Data class for cut results."""

    circuits: List[QuantumCircuit]
    coefficients: List[Tuple[float, WeightType]]
    n_shots: int
    observables: PauliList | Dict[str, PauliList]
    partition_label: str
    result_counts: List[Dict[str, int]] | None
    uuid: UUID
