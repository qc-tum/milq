"""Result reconstruction using CKT toolbox."""
from circuit_knitting.cutting import reconstruct_expectation_values
from qiskit.primitives.sampler import SamplerResult

from src.common import Experiment


def reconstruct_counts(experiment: Experiment) -> None:
    ...


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
