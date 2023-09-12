"""Circuit cutting using the CTK library."""
from typing import List

from circuit_knitting.cutting import (
    partition_problem,
    execute_experiments,
    reconstruct_expectation_values,
)
from qiskit import QuantumCircuit


def cut_circuit(circuit: QuantumCircuit, partitions: List[int]) -> List[QuantumCircuit]:
    """_summary_

    Args:
        circuit (QuantumCircuit): _description_
        partitions (str): _description_

    Returns:
        List[QuantumCircuit]: _description_
    """
    # TODO find a way to communicate cutting information for reassembly at a later time
    partitions = _generate_partition_lables(partitions)
    circuits = partition_problem(circuit, partitions)
    return list(circuits.subcircuits.values())


def _generate_partition_lables(partitions: List[int]) -> str:
    # TODO find a smart way to communicate partition information
    return "".join(str(i) * value for i, value in enumerate(partitions))
