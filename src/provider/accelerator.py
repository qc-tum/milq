"""Wrapper for IBMs backend simulator."""

from collections import deque
from uuid import UUID, uuid4
import logging

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.common import IBMQBackend, CombinedJob
from src.resource_estimation import estimate_runtime, estimate_noise
from src.tools import optimize_circuit_online, generate_subcircuit
from utils.helpers import time_conversion


class Accelerator:
    """Wrapper for a single backend simulator."""

    def __init__(
        self, backend: IBMQBackend, shot_time: int = 1, reconfiguration_time: int = 0
    ) -> None:
        self.simulator = AerSimulator.from_backend(backend.value())
        self._backend = backend
        self._qubits = len(self.simulator.properties().qubits)
        self._shot_time = shot_time
        self._reconfiguration_time = reconfiguration_time
        self._uuid = uuid4()
        self.queue: deque[CombinedJob] = deque([])

    def compute_processing_time(self, circuit: QuantumCircuit) -> float:
        """Computes the processing time for the circuit for a single shot.

        Args:
            circuit (QuantumCircuit): The circuit to analyze.

        Returns:
            float: The processing time in µs.
        """
        logging.debug("Computing processing time for circuit...")
        # TODO: should define real gate params in resource estimation
        time_in_ns = estimate_runtime(circuit)

        logging.debug("Done.")
        return time_conversion(time_in_ns, "ns", target_unit="us")

    def compute_setup_time(
        self, circuit_from: QuantumCircuit | None, circuit_to: QuantumCircuit | None
    ) -> float:
        """Computes the set up time by switching between one circuit to another.

        # TODO curretly only the constant reconfiguration time is returned.
        Args:
            circuit_from (QuantumCircuit): Ending circuit.
            circuit_to (QuantumCircuit): Starting circuit.

        Returns:
            float: Set up time from circuit_from to circuit_to in µs.
        """
        logging.debug("Computing setup time for circuit...")
        if circuit_from is None:
            return 0
        if circuit_to is None:
            return self._reconfiguration_time
        return self._reconfiguration_time

    def compute_noise(self, quantum_circuit: QuantumCircuit) -> float:
        """Estimates the noise of a circuit on an accelerator.

        Args:
            circuit (QuantumCircuit): The circuit to estimate the noise for.

        Returns:
            float: The estimated noise of the circuit on the accelerator.
        """
        if quantum_circuit.num_qubits > self.qubits:
            sub_quantum_circuit = generate_subcircuit(
                quantum_circuit, list(range(self.qubits))
            )
            return (
                estimate_noise(sub_quantum_circuit, self.simulator)
                * quantum_circuit.num_qubits
                / self.qubits
            )
        return estimate_noise(quantum_circuit, self.simulator)

    @property
    def shot_time(self) -> int:
        """Time factor for each shot.

        Returns:
            int: The time one shot takes.
        """
        return self._shot_time

    @property
    def reconfiguration_time(self) -> int:
        """Additional time penalty for reconfiguration.

        Returns:
            int: The recongiguration time.
        """
        return self._reconfiguration_time

    @property
    def qubits(self) -> int:
        """Number of qubits.

        Returns:
            int: The number of qubits.
        """
        return self._qubits

    @property
    def backend(self) -> IBMQBackend:
        """The backend, which is simulated.

        Returns:
            IBMQBackend: The backend.
        """
        return self._backend

    @property
    def uuid(self) -> UUID:
        """_summary_

        Returns:
            UUID: _description_
        """
        return self._uuid

    def run_and_get_counts(
        self, circuit: QuantumCircuit, n_shots: int = 2**10
    ) -> dict[str, int]:
        """Run a circuit and get the measurment counts.

        The circuit is optimized before running, using the now available backend information.
        Args:
            circuit (QuantumCircuit): The circuit to run.
            n_shots (int, optional): Number of shots. Defaults to 2**10.

        Returns:
            dict[str, int]: Measurment counts.
        """
        # TODO check qubit size
        # opt_circuit = optimize_circuit_online(circuit, self._backend)
        # TODO For some reason the above line blocks
        result = self.simulator.run(circuit, shots=n_shots).result()
        return result.get_counts(0)
