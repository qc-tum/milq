"""Generates the benchmark data."""

import logging

from mqt.bench import get_benchmark
from qiskit import QuantumCircuit
import numpy as np

from src.provider import Accelerator, IBMQBackend
from src.scheduling.learning import train_for_settings


def _generate_batch(max_qubits: int, circuits_per_batch: int) -> list[QuantumCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(1, max_qubits + 1)
        circuit = get_benchmark(benchmark_name="random", level=2, circuit_size=size)
        circuit.remove_final_measurements(inplace=True)
        batch.append(circuit)

    return batch


# Define different settings for training


ACCELERATORS = [
    [
        Accelerator(IBMQBackend.BELEM, shot_time=5, reconfiguration_time=12),
        Accelerator(IBMQBackend.NAIROBI, shot_time=7, reconfiguration_time=12),
        None,
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
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("circuit_knitting").setLevel(logging.WARNING)
    logging.getLogger("stevedore").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    settings = [
        {"accelerators": accs, "circuits": _generate_batch(MAX_QUBITS, CIRC_PER_BATCH)}
        for accs in ACCELERATORS
    ]
    train_for_settings(settings)
