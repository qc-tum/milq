"""Assemble a single circuit from multiple independent ones."""
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import PauliList

from src.common import CircuitJob, CombinedJob


def assemble_circuit(circuits: list[QuantumCircuit]) -> QuantumCircuit:
    """Assemble a single circuit from stacking multiple independent circuits on top
    of each other.

    Args:
        circuits (list[QuantumCircuit]): The individual circuits

    Returns:
        QuantumCircuit: The combined circuit
    """
    composed_circuit = QuantumCircuit()
    for idx, circuit in enumerate(circuits):
        for creg in circuit.cregs:
            composed_circuit.add_register(
                ClassicalRegister(creg.size, f"{idx}_{creg.name}")
            )
        for qreg in circuit.qregs:
            composed_circuit.add_register(
                QuantumRegister(qreg.size, f"{idx}_{qreg.name}")
            )

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
    """Assembles multiple circuit jobs into a single combined job.

    Args:
        circuit_jobs (list[CircuitJob]): The individual circuit jobs

    Returns:
        CombinedJob: The combined job
    """
    if len(circuit_jobs) == 0:
        return CombinedJob()
    combined_job = CombinedJob(n_shots=circuit_jobs[0].n_shots)
    circuits = []
    qubit_count = 0
    observable = PauliList("")
    for job in circuit_jobs:
        combined_job.indices.append(job.index)
        circuits.append(job.circuit)
        combined_job.coefficients.append(job.coefficient)
        combined_job.mapping.append(
            slice(qubit_count, qubit_count + job.circuit.num_qubits)
        )
        qubit_count += job.circuit.num_qubits
        observable = observable.expand(job.observable)
        combined_job.partition_lables.append(job.partition_label)
        combined_job.uuids.append(job.uuid)
        combined_job.cregs.append(job.cregs)
    combined_job.circuit = assemble_circuit(circuits)
    combined_job.observable = observable
    return combined_job
