""""""
from typing import Dict, List

from qiskit import QuantumCircuit

from .accelerator import Accelerator


class AcceleratorGroup:
    """_summary_"""

    def __init__(self, accelerators: List[Accelerator]) -> None:
        self.accelerators = accelerators
        self._qubits = sum(acc.qubits for acc in accelerators)

    @property
    def qubits(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self._qubits

    def run_and_get_counts(
        self, circuits: List[QuantumCircuit]
    ) -> List[Dict[int, int]]:
        """_summary_

        Args:
            circuits (List[QuantumCircuit]): _description_

        Returns:
            List[Dict[int, int]]: _description_
        """
        counts = []
        for circuit, accelerator in zip(circuits, self.accelerators):
            # TODO in parallel!
            counts.append(accelerator.run_and_get_counts(circuit))
        # TODO do some magic to figure out which counts belong to which circuit
        return counts
