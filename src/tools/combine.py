"""Tool for appending a single circuit to a combined job."""

from src.common import CombinedJob, UserCircuit, job_from_circuit
from .assembling import assemble_circuit


def combine(combined_job: CombinedJob, user_circuit: UserCircuit) -> None:
    """Appends a single circuit job to a combined job.

    Args:
        combined_job (CombinedJob): The combined job, which will be extended.
        user_circuit (CircuitJob): The new circuit job

    """
    circuit_job = job_from_circuit(user_circuit.circuit)
    combined_job.indices.append(circuit_job.index)
    combined_job.coefficients.append(circuit_job.coefficient)
    combined_job.mapping.append(
        slice(
            combined_job.mapping[-1].stop,
            combined_job.mapping[-1].stop + circuit_job.circuit.num_qubits,
        )
    )
    combined_job.observable = combined_job.observable.expand(circuit_job.observable)
    combined_job.partition_lables.append(circuit_job.partition_label)
    combined_job.uuids.append(circuit_job.uuid)
    combined_job.cregs.append(circuit_job.cregs)
    combined_job.circuit = assemble_circuit([combined_job.circuit, circuit_job.circuit])
