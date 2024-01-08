"""Circuit cutting using the CTK library."""
from uuid import UUID, uuid4

from circuit_knitting.cutting import (
    partition_problem,
    generate_cutting_experiments,
)
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList
import numpy as np

from src.common import Experiment


def cut_circuit(
    circuit: QuantumCircuit,
    partitions: list[int],
    observables: (PauliList | None) = None,
) -> tuple[list[Experiment], UUID]:
    """Cut a circuit into multiple subcircuits.

    Args:
        circuit (QuantumCircuit): The circuit to cut
        partitions (list[int]): The partitions to cut the circuit into (given as a list of qubits)
        observables (PauliList  |  None, optional): The observables for each qubit.
            Defaults to None (= Z measurements).

    Returns:
        tuple[list[Experiment], UUID]: _description_
    """
    if observables is None:
        observables = PauliList("Z" * circuit.num_qubits)
    gen_partitions = _generate_partition_labels(partitions)
    partitioned_problem = partition_problem(circuit, gen_partitions, observables)
    experiments, coefficients = generate_cutting_experiments(
        partitioned_problem.subcircuits,
        partitioned_problem.subobservables,
        num_samples=np.inf,
    )
    uuid = uuid4()
    return [
        Experiment(
            circuits,
            coefficients,  # split up by order?
            2**12,  # TODO Calculate somehow
            partitioned_problem.subobservables[partition_label],
            partition_label,
            None,
            uuid,
        )
        for partition_label, circuits in experiments.items()
    ], uuid


def _generate_partition_labels(partitions: list[int]) -> str:
    # TODO find a smart way to communicate partition information
    return "".join(str(i) * value for i, value in enumerate(partitions))
