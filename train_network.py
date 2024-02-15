"""Generates the benchmark data."""

from mqt.bench import get_benchmark
from qiskit import QuantumCircuit
import numpy as np

from src.provider import Accelerator, IBMQBackend
from src.scheduling.learning import train_for_settings


def _generate_batch(max_qubits: int, circuits_per_batch: int) -> list[QuantumCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(2, max_qubits + 1)
        circuit = get_benchmark(benchmark_name="random", level=0, circuit_size=size)
        batch.append(circuit)

    return batch


# Define different settings for training


ACCELERATORS = [
    [
        Accelerator(IBMQBackend.BELEM, shot_time=5, reconfiguration_time=12),
        Accelerator(IBMQBackend.NAIROBI, shot_time=7, reconfiguration_time=12),
    ],
    [
        Accelerator(IBMQBackend.BELEM, shot_time=5, reconfiguration_time=12),
        Accelerator(IBMQBackend.NAIROBI, shot_time=7, reconfiguration_time=12),
        Accelerator(IBMQBackend.QUITO, shot_time=2, reconfiguration_time=16),
    ],
]

CIRC_PER_BATCH = 5
MAX_QUBITS = 25

if __name__ == "__main__":
    settings = [
        {"accelerators": accs, "circuits": _generate_batch(MAX_QUBITS, CIRC_PER_BATCH)}
        for accs in ACCELERATORS
    ]
    train_for_settings(settings)
