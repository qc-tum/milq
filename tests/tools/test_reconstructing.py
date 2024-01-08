"""_summary"""
from pytest import approx

from src.circuits import create_quantum_only_ghz
from src.common import jobs_from_experiment, IBMQBackend
from src.provider import Accelerator
from src.tools import (
    assemble_job,
    cut_circuit,
    optimize_circuit_offline,
    reconstruct_counts_from_job,
    reconstruct_experiments_from_circuits,
    reconstruct_expvals,
)


def test_reconstruct_expvals() -> None:
    """_summary_"""
    backend = IBMQBackend.BELEM
    accelerator = Accelerator(backend)
    circuit = create_quantum_only_ghz(7)
    circuit = optimize_circuit_offline(circuit, backend)
    experiments, uuid = cut_circuit(circuit, [3, 4])
    for experiment in experiments:
        experiment.result_counts = [
            accelerator.run_and_get_counts(circuit) for circuit in experiment.circuits
        ]
    exp_vals = reconstruct_expvals(list(filter(lambda x: x.uuid == uuid, experiments)))
    assert len(exp_vals) == 1
    assert exp_vals[0] == approx(0, abs=0.05)


def test_reconstruct_counts_from_job() -> None:
    """_summary_"""
    backend = IBMQBackend.BELEM
    accelerator = Accelerator(backend)
    circuit = create_quantum_only_ghz(5)
    circuit = optimize_circuit_offline(circuit, backend)
    experiments, _ = cut_circuit(circuit, [2, 3])
    jobs = []
    for experiment in experiments:
        jobs += jobs_from_experiment(experiment)

    combined_job = assemble_job([jobs[2], jobs[10]])
    combined_job.result_counts = accelerator.run_and_get_counts(combined_job.circuit)
    jobs = reconstruct_counts_from_job(combined_job)
    assert len(jobs[0].result_counts) == 8
    assert len(jobs[1].result_counts) == 16


def test_reconstruct_experiments_from_circuits() -> None:
    """_summary_"""
    backend = IBMQBackend.BELEM
    accelerator = Accelerator(backend)
    circuit = create_quantum_only_ghz(5)
    circuit = optimize_circuit_offline(circuit, backend)
    experiments, _ = cut_circuit(circuit, [2, 3])
    jobs = []
    for experiment in experiments:
        jobs += jobs_from_experiment(experiment)
    job_results = []
    for job_1, job_2 in zip(jobs[:6], jobs[6:]):
        combined_job = assemble_job([job_1, job_2])
        combined_job.result_counts = accelerator.run_and_get_counts(
            combined_job.circuit
        )
        job_results += reconstruct_counts_from_job(combined_job)
    # TODO how do I know that all circuits from an experiment have run?
    reconstructed_experiments = reconstruct_experiments_from_circuits(job_results)
    assert len(reconstructed_experiments) == 2
    assert reconstructed_experiments[0].coefficients == experiments[0].coefficients
