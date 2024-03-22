from dataclasses import dataclass
import numpy as np

from .running import estimate_runtime
from .noise import estimate_noise
from .device import predict_device

try:
    from .cutting.src import ResourceEstimator
except ImportError:

    @dataclass
    class Resource:
        """Dataclass for the resources required to evaluate a cut quantum circuit."""

        kappa: int
        gate_groups: dict[str, list[int]]
        n_samples: int
        n_circuits: int

        def __repr__(self) -> str:
            if 1 == self.n_circuits:
                return (
                    f"{self.n_circuits} job with "
                    + f"{round(self.n_samples / self.n_circuits)} shots."
                )
            return (
                f"{self.n_circuits} jobs with "
                + f"{round(self.n_samples / self.n_circuits)} shots each."
            )

    class ResourceEstimator:
        def __init__(self, circuit) -> None:
            self.circuit = circuit

        def resource(self, binary, epsilon, delta) -> Resource:
            counter = 0
            indices = [idx for idx, value in enumerate(binary) if value == 0]
            hoefdings = 2 / epsilon**2 * np.log(2 / delta)
            for gate in self.circuit.data:

                if not all(
                    self.circuit.find_bit(qubit).index in indices for qubit in gate[1]
                ):
                    counter += 1
            return Resource(3, {}, 3**2 * hoefdings, 6**counter)
