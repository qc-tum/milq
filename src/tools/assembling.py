"""Assemble a single circuit from multiple independent ones."""
from qiskit import QuantumCircuit
from qiskit.quantum_info import PauliList

from src.common import CircuitJob, CombinedJob


def assemble_circuit(circuits: list[QuantumCircuit]) -> QuantumCircuit:
    """_summary_

    Args:
        circuits (list[QuantumCircuit]): _description_

    Returns:
        QuantumCircuit: _description_
    """
    qubits, clbits = [
        sum(x)
        for x in zip(
            *[(circuit.num_qubits, circuit.num_clbits) for circuit in circuits]
        )
    ]
    composed_circuit = QuantumCircuit(qubits, clbits)
    qubits, clbits = 0, 0
    for circuit in circuits:
        composed_circuit.compose(
            circuit,
            qubits=list(range(qubits, qubits + circuit.num_qubits)),
            clbits=list(range(clbits, clbits + circuit.num_clbits)),
            inplace=True,
        )
        qubits += circuit.num_qubits
        clbits += circuit.num_clbits
    return composed_circuit


def assemble_job(circuit_jobs: list[CircuitJob]) -> CombinedJob:
    """_summary_

    Args:
        circuit_jobs (list[CircuitJob]): _description_

    Returns:
        CombinedJob: _description_
    """
    combined_job = CombinedJob(n_shots=circuit_jobs[0].n_shots)
    circuits = []
    qubit_count = 0
    observable = PauliList("")
    for job in circuit_jobs:
        combined_job.indices.append(job.index)
        circuits.append(job.instance)
        combined_job.coefficients.append(job.coefficient)
        combined_job.mapping.append(
            slice(qubit_count, qubit_count + job.instance.num_qubits)
        )
        qubit_count += job.instance.num_qubits
        observable = observable.expand(job.observable)
        combined_job.partition_lables.append(job.partition_lable)
    combined_job.instance = assemble_circuit(circuits)
    combined_job.observable = observable
    return combined_job
