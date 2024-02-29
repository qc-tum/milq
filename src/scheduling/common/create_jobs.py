from qiskit import QuantumCircuit

from src.common import CircuitJob, jobs_from_experiment, job_from_circuit
from src.tools import cut_circuit

from .types import CircuitProxy


def convert_to_jobs(
    circuits: list[QuantumCircuit], partitions: list[list[int]]
) -> list[CircuitProxy]:
    jobs = []
    for idx, circuit in enumerate(
        sorted(circuits, key=lambda circ: circ.num_qubits, reverse=True)
    ):
        if len(partitions[idx]) > 1:
            experiments, _ = cut_circuit(circuit, partitions[idx])
            jobs += [
                job
                for experiment in experiments
                for job in jobs_from_experiment(experiment)
            ]
        else:
            # assumption for now dont cut to any to smaller
            circuit = job_from_circuit(circuit)
            jobs.append(circuit)
    return jobs
