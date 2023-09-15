"""Circuit cutting using the CTK library."""
from dataclasses import dataclass
from typing import Dict, List, Tuple
from uuid import UUID, uuid4

from circuit_knitting.cutting import (
    partition_problem,
    generate_cutting_experiments,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList
import numpy as np


@dataclass
class Experiment:
    """Data class for cut results."""

    circuits: List[QuantumCircuit]
    coeficients: List[float]
    n_shots: int
    partition_lable: str
    result_counts: List[Dict[str, int]] | None
    uuid: UUID


def cut_circuit(
    circuit: QuantumCircuit,
    partitions: List[int],
    observables: (PauliList | None) = None,
) -> Tuple[List[Experiment], UUID]:
    """_summary_

    Args:
        circuit (QuantumCircuit): _description_
        partitions (List[int]): _description_
        observables (PauliList  |  None, optional): _description_. Defaults to None.

    Returns:
        List[Experiment]: _description_
    """
    if observables is None:
        observables = PauliList("Z" * circuit.num_qubits)
    partitions = _generate_partition_lables(partitions)
    partitioned_problem = partition_problem(circuit, partitions, observables)
    experiments, coefficients = generate_cutting_experiments(
        partitioned_problem.subcircuits,
        partitioned_problem.subobservables,
        num_samples=np.inf,
    )
    uuid = uuid4()
    return [
        Experiment(
            circuits,
            [coeff for coeff, _ in coefficients],
            2**12,  # TODO Calculate somehow
            partition_lable,
            None,
            uuid,
        )
        for partition_lable, circuits in experiments.items()
    ], uuid


def _generate_partition_lables(partitions: List[int]) -> str:
    # TODO find a smart way to communicate partition information
    return "".join(str(i) * value for i, value in enumerate(partitions))
