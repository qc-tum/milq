"""Result reconstruction using CKT toolbox."""
from collections import Counter

from circuit_knitting.cutting import reconstruct_expectation_values
from qiskit.primitives.sampler import SamplerResult

from src.common import CircuitJob, CombinedJob, Experiment


def reconstruct_counts_from_job(job: CombinedJob) -> list[CircuitJob]:
    """_summary_

    Args:
        job (CombinedJob): _description_

    Returns:
        list[CircuitJob]: _description_
    """
    circuit_jobs = []
    offset = 0
    for idx, _ in enumerate(job.indices):
        mapping = job.mapping[idx]
        observable = job.observable[0][mapping]
        counts = _get_partial_counts(job.result_counts, offset, job.cregs[idx])
        circuit_jobs.append(
            CircuitJob(
                job.indices[idx],
                None,
                job.coefficients[idx],
                job.n_shots,
                observable,
                job.partition_lables[idx],
                counts,
                job.uuids[idx],
                job.cregs[idx],
            )
        )
        offset += job.cregs[idx]
    return circuit_jobs


def _get_partial_counts(
    counts: dict[str, int], offset: int, cregs: int
) -> dict[str, int]:
    partial_counts = Counter()
    for bits, count in counts.items():
        current_bits = " ".join(bits[::-1].split(" ")[offset : offset + cregs][::-1])
        partial_counts[current_bits] += count
    return dict(partial_counts)


def reconstruct_expvals(experiments: list[Experiment]) -> list[float]:
    """_summary_

    Args:
        experiments (list[Experiment]): _description_

    Returns:
        list[float]: _description_
    """
    coefficients = experiments[0].coefficients
    subobservables = {
        experiment.partition_label: experiment.observables for experiment in experiments
    }

    results = {}
    for experiment in experiments:
        quasi_distributions = []
        for counts in experiment.result_counts:
            quasi_distribution = {}
            metadata = []
            for bits, count in counts.items():
                quasi_distribution[int(bits.replace(" ", ""), 2)] = (
                    count / experiment.n_shots
                )
            quasi_distributions.append(quasi_distribution)
            for circuit in experiment.circuits:
                metadata.append({"num_qpd_bits": len(circuit.cregs[0])})
        result = SamplerResult(quasi_distributions, metadata)
        results[experiment.partition_label] = result

    return reconstruct_expectation_values(
        results,
        coefficients,
        subobservables,
    )
