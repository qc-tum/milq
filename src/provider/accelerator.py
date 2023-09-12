"""Wrapper for IBMs backend simulator."""
from enum import Enum
from typing import Dict

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import FakeBelem, FakeNairobi, FakeQuito


class IBMQBackend(Enum):
    """_summary_

    Args:
        Enum (_type_): _description_
    """

    BELEM = FakeBelem
    NAIROBI = FakeNairobi
    QUITO = FakeQuito


class Accelerator:
    """_summary_"""

    def __init__(self, backend: IBMQBackend) -> None:
        self.simulator = AerSimulator.from_backend(backend.value())
        self._backend = backend
        self._qubits = len(self.simulator.properties().qubits)

    @property
    def qubits(self) -> int:
        """_summary_

        Returns:
            int: _description_
        """
        return self._qubits

    @property
    def backend(self) -> IBMQBackend:
        """_summary_

        Returns:
            IBMQBackend: _description_
        """
        return self._backend

    def run_and_get_counts(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """_summary_

        Args:
            circuit (QuantumCircuit): _description_

        Returns:
            Dict[str, int]: _description_
        """
        # TODO check qubit size
        # TODO check if transpile here is necessary / needs to be moved somewhere else
        result = self.simulator.run(transpile(circuit, self.simulator)).result()
        return result.get_counts(0)
