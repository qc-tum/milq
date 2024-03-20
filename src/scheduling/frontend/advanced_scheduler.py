"""A scheduler for quantum circuits."""

from collections import deque
from uuid import UUID

from src.common import UserCircuit
from src.provider import Accelerator
from src.tools import combine
from ..types import SchedulerType


class AdvancedScheduler:
    """Classical scheduler for quantum circuits.

    TODO:
    - Fix assembling of circuits
    - "Listen" to job sobmussions
    - Start up accelerators
    - Select scheduling type
    - Use scheduling algorithms

    Args:
        accelerators (list[Accelerator]): _description_
        stype (SchedulerType, optional): _description_. Defaults to SchedulerType.BASELINE.
        allow_backfilling (bool, optional): _description_. Defaults to True.
    """

    def __init__(
        self,
        accelerators: list[Accelerator],
        stype: SchedulerType = SchedulerType.BASELINE,
        allow_backfilling: bool = True,
        batch_size: int = 5,
    ) -> None:

        self.accelerators = {str(acc.uuid): acc for acc in accelerators}

        self.uuids: list[UUID] = []
        self.stype = stype
        self._queue = deque([])
        self.allow_backfilling = allow_backfilling
        self._stop = False
        self.batch_size = batch_size

        def submit_circuit(self, circuit: UserCircuit) -> None:
            """Submits a new circuit to the scheduler."""
            if not self.allow_backfilling:
                self._queue.append(circuit)
                return

            if (
                circuit.machine_preference is not None
                and circuit.machine_preference in self.accelerators
                and len(self.accelerators[circuit.machine_preference].queue) == 0
            ):
                self.accelerators[circuit.machine_preference].queue.append(circuit)
                return
            # TODO: define Max value
            if circuit.strictness > 10:
                self._queue.append(circuit)
                return
            # backfilling
            for acc in self.accelerators.values():
                if len(acc.queue) == 0:
                    acc.queue.append(circuit)
                    return

            for acc in self.accelerators.values():
                for job in acc.queue:
                    if (
                        job.circuit.num_qubits + circuit.circuit.num_qubits
                        <= acc.qubits
                    ):
                        combine(job, circuit)
                        return
            self._queue.append(circuit)

        def stop(self) -> None:
            self._stop = True

        def run(self) -> None:
            """Starts the scheduling process."""
            while not self._stop:
                circuits = []
                for _ in range(self.batch_size):
                    if len(self._queue) == 0:
                        continue
                    circuit = self._queue.popleft()
                    circuits.append(circuit)
                self._schedule(circuits)

        def _schedule(self, circuits: list[UserCircuit]) -> None:
            pass
