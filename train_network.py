"""Generates the benchmark data."""

import logging

from mqt.bench import get_benchmark
import numpy as np

from src.common import UserCircuit
from src.provider import Accelerator, IBMQBackend
from src.scheduling.learning import train_for_settings


def _generate_batch(
    max_qubits: int, circuits_per_batch: int, accelerators: list[Accelerator]
) -> list[UserCircuit]:
    # Generate a random circuit
    batch = []
    for _ in range(circuits_per_batch):
        size = np.random.randint(2, max_qubits + 1)
        circuit = get_benchmark(benchmark_name="random", level=2, circuit_size=size)
        circuit.remove_final_measurements(inplace=True)
        user_circuit = UserCircuit(
            circuit,
            size,
            np.random.randint(1, 10),
            str(accelerators[np.random.randint(len(accelerators))].uuid),
            np.random.randint(1, 3),
        )
        batch.append(user_circuit)

    return batch


# Define different settings for training


ACCELERATORS: list[list[Accelerator | None]] = [
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
    logging.basicConfig(
        level=logging.INFO,
        filename="train.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        encoding="utf-8",
    )
    for setting in ACCELERATORS:
        for acc in setting:
            if acc is not None:
                acc.queue.extend([0] * np.random.randint(0, 10))
    logging.getLogger("qiskit").setLevel(logging.WARNING)
    logging.getLogger("circuit_knitting").setLevel(logging.WARNING)
    logging.getLogger("stevedore").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    settings = [
        {
            "accelerators": accs,
            "circuits": _generate_batch(
                MAX_QUBITS, CIRC_PER_BATCH, [acc for acc in accs if acc is not None]
            ),
        }
        for accs in ACCELERATORS
    ]
    train_for_settings(settings, 10**5)
    logging.info("Training done.")
