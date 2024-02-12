"""_summary_"""

from src.scheduling.heuristics import generate_heuristic_info_schedule

from src.circuits import create_quantum_only_ghz
from src.common import IBMQBackend
from src.provider import Accelerator


def test_generate_heuristic_info_schedule():
    """Test for generate_heuristic_info_schedule."""
    circuits = [create_quantum_only_ghz(qubits) for qubits in range(8, 9)]
    accelerators = [
        Accelerator(IBMQBackend.BELEM, shot_time=1, reconfiguration_time=1),
        Accelerator(IBMQBackend.NAIROBI, shot_time=1, reconfiguration_time=1),
    ]
    schedule, makespan = generate_heuristic_info_schedule(
        circuits, accelerators, num_iterations=2, partition_size=3
    )
    assert 5 < len(schedule) < 13
    assert 45 < makespan < 100
