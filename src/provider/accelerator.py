"""Wrapper for IBMs backend simulator."""
from uuid import UUID, uuid4

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from src.common import IBMQBackend
from src.tools import optimize_circuit_online


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

    @staticmethod
    def _time_conversion(
        time: float, unit: str, target_unit: str = "us", dt: float | None = None
    ) -> float:
        """Converts a time from one unit to another.

        Args:
            time (float): The time to convert.
            unit (str): The unit of the time.
            target_unit (str, optional): The target unit. Defaults to "us".
            dt (float | None, optional): The duration in seconds of the device-dependent
            time. Must be set if unit is in dt but target isn't. Defaults to None.

        Returns:
            float: _description_
        """
        if unit == target_unit:
            return time

        units = ["s", "ms", "us", "ns", "ps"]

        # target_unit must be a SI unit
        assert target_unit in units

        # Convert dt (device-dependent time) to SI unit
        if unit == "dt":
            assert dt is not None
            time *= dt
            unit = "s"

        target_shift = units.index(target_unit)
        current_shift = units.index(unit)
        required_shift = 3 * (target_shift - current_shift)
        return time * 10**required_shift

    def compute_processing_time(self, circuit: QuantumCircuit) -> float:
        """Computes the processing time for the circuit for a single shot.

        Args:
            circuit (QuantumCircuit): The circuit to analyze.

        Returns:
            float: The processing time in µs.
        """
        # TODO: doing a full hardware-aware compilation just to get the processing
        # time is not efficient. An approximation would be better.
        be = self._backend.value()
        transpiled_circuit = transpile(circuit, be, scheduling_method="alap")
        return Accelerator._time_conversion(
            transpiled_circuit.duration, transpiled_circuit.unit, dt=be.dt
        )

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
        if circuit_from is None:
            return self._reconfiguration_time
        if circuit_to is None:
            return self._reconfiguration_time
        return self._reconfiguration_time

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
