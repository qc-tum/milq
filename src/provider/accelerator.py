"""Wrapper for IBMs backend simulator."""
from typing import Dict

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit

from src.common import IBMQBackend
from src.tools import optimize_circuit_online


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
        opt_circuit = optimize_circuit_online(circuit, self.simulator)
        result = self.simulator.run(opt_circuit).result()
        return result.get_counts(0)
