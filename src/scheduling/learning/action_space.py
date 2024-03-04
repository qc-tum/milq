"""Custom action space for the scheduling environment."""

from gymnasium.spaces import Discrete, Dict, MultiDiscrete
from qiskit import QuantumCircuit

from src.scheduling.common import Schedule


class ActionSpace(Dict):
    """The action space for the scheduling environment.
    It contains the spaces for the following actions:"""

    def __init__(self, circuits: list[QuantumCircuit], schedule: Schedule) -> None:
        n_circuits = len(circuits)
        n_buckets = sum(
            len(machine.buckets) + 1 for machine in schedule.machines
        )  # +1 for allowing new bucket
        # 0: cut, 1: move, 2: swap ## removed 1: combine
        super().__init__(
            {
                "action": Discrete(3),
                "params": MultiDiscrete([n_circuits, n_circuits, n_buckets]),
            }
        )

    def update_actions(self, schedule: Schedule) -> None:
        n_circuits = 0
        n_buckets = 0

        for machine in schedule.machines:
            n_circuits += sum(len(bucket.jobs) for bucket in machine.buckets)
            n_buckets += len(machine.buckets) + 1  # +1 for allowing new bucket
        self.spaces["params"] = MultiDiscrete([n_circuits, n_circuits, n_buckets])

    def enable_terminate(self) -> None:
        self.spaces["action"] = Discrete(4)  # 0: cut, 1: move, 2: swap, 3: terminate

    def disable_terminate(self) -> None:
        self.spaces["action"] = Discrete(3)
