"""Test the resource estimation."""

from src.circuits import create_quantum_only_ghz
from src.resource_estimation import ResourceEstimator


def test_resource_estimator() -> None:
    """Test the sampling overhead.
    We cut a single CNOT gate from a 10-qubit GHZ state and estimate the resources

    """
    circuit = create_quantum_only_ghz(10)
    estimator = ResourceEstimator(circuit)
    partition = [0] * 5 + [1] * 5
    resource = estimator.resource(
        binary=partition, epsilon=0.1, delta=0.1, method="simple"
    )
    assert resource.n_circuit_pairs * 2 == 12
    assert resource.n_samples // (2 * resource.n_circuit_pairs) == 898
