"""Result reconstruction using CKT toolbox."""
from collections import Counter
from functools import partial

from circuit_knitting.cutting import reconstruct_expectation_values
from qiskit.primitives.sampler import SamplerResult

from src.common import CircuitJob, CombinedJob, Experiment


def reconstruct_experiments_from_circuits(jobs: list[CircuitJob]) -> list[Experiment]:
    """_summary_

    Args:
        jobs (list[CircuitJob]): _description_

    Returns:
        list[Experiment]: _description_
    """
    uuids = set(job.uuid for job in jobs)
    experiments = []
    for uuid in uuids:
        uuid_jobs = list(filter(partial(lambda j, u: j.uuid == u, u=uuid), jobs))
        partitions = set(job.partition_label for job in uuid_jobs)
        for partition in partitions:
            partition_jobs = sorted(
                filter(
                    partial(lambda j, p: j.partition_label == p, p=partition), uuid_jobs
                ),
                key=lambda x: x.index,
            )

            experiments.append(
                Experiment(
                    None,
                    [job.coefficient for job in partition_jobs],
                    0,
                    partition_jobs[0].observable,  # TODO fix for multiple
                    partition,
                    [job.result_counts for job in partition_jobs],
                    uuid,
                )
            )
    return experiments


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
        # Get the observable to be measured for this circuit.
        mapping = job.mapping[idx]
        observable = job.observable[0][mapping]

        # Get the counts for the results of this circuit.
        counts = _get_partial_counts(job.result_counts, offset, job.cregs[idx])

        # Create a new CircuitJob for this circuit.
        circuit_jobs.append(
            CircuitJob(
                coefficient=job.coefficients[idx],
                cregs=job.cregs[idx],
                index=job.indices[idx],
                circuit=None,
                n_shots=job.n_shots,
                observable=observable,
                partition_label=job.partition_lables[idx],
                result_counts=counts,
                uuid=job.uuids[idx],
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
